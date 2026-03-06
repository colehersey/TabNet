# Neural Sequence Generator

A character-level recurrent neural network running bare-metal on an ATmega32U4 microcontroller. It autocompletes Linux shell commands in real time using only fixed-point integer math — no floating point, no OS, no frameworks. Type a command prefix over serial, press a button, and the device predicts what comes next.

The entire model — 96 hidden units, 17KB of weights, piecewise-linear tanh, int32 accumulators — fits in 24KB of flash and 549 bytes of RAM, and runs inference in under 15ms per keystroke at 8 MHz.

## Demo

```
> make c        -> lean        (predicts "make clean")
> make f        -> lash        (predicts "make flash")
> git d         -> iff         (predicts "git diff")
> git s         -> tatus       (predicts "git status")
> grep -        -> r           (predicts "grep -r")
```

Each prediction is generated token-by-token with a confidence bar displayed on 8 LEDs. The user can accept predictions into the context, deny them to roll back, or keep pressing predict to extend the sequence.

## Project Structure

```
firmware/                       Bare-metal C for ATmega32U4
  main.c                        Main loop, state machine, button/serial dispatch
  rnn.c / rnn.h                 Fixed-point RNN inference engine
  glcd.c / glcd.h               ST7565 128x32 SPI LCD driver
  display.c / display.h         Font rendering and UI layout
  font5x7.h                     5x7 bitmap font in PROGMEM
  leds.c / leds.h               8-LED confidence bar
  serial.c / serial.h           UART1 serial (9600 baud, U2X mode)
  controls.c / controls.h       4-button input with debounce
  config.h                      Pin definitions, model dimensions, display layout
  weights.h                     Quantized weight matrices (generated)
  vocab.h                       Character/token mapping (generated)
  Makefile                      avr-gcc build (alternative to Atmel Studio)

trainer/                        Python training and export pipeline
  train.py                      PyTorch char-level RNN trainer
  export.py                     Quantizer and C header generator
  benchmark.py                  Float32-vs-firmware fidelity benchmark
  corpus.txt                    Training data (~619 lines of shell commands)
  model.pt                      Trained model weights (generated)
  meta.json                     Vocabulary and model metadata (generated)

CHANGELOG.md                    Detailed engineering log of all findings
```

## Requirements

**Training (PC/WSL):**
- Python 3.8+
- PyTorch >= 2.0
- NumPy >= 1.24

**Firmware:**
- avr-gcc toolchain
- avrdude (or Atmel Studio)
- ATmega32U4 board with ST7565 128x32 GLCD, 8 LEDs, 4 buttons, UART serial

## Build and Flash

```bash
# Train the model
cd trainer
pip install -r requirements.txt
python train.py         # trains RNN, saves model.pt + meta.json
python export.py        # quantizes weights, writes firmware/weights.h + vocab.h

# Build and flash firmware
cd ../firmware
make clean && make flash
```

The Makefile uses `avr109` (Caterina bootloader). Put the board in bootloader mode (double-tap reset) before flashing. Adjust `PORT` in the Makefile to match your system.

## How It Works

### User Interaction

The device has four buttons and a serial terminal interface:

| Button | Action |
|--------|--------|
| **Predict** | Generate the next predicted character (can press repeatedly to chain predictions) |
| **Accept** | Commit the prediction into the context — the RNN keeps its current state |
| **Deny** | Discard the prediction and roll the RNN back to the pre-prediction state |
| **Reset** | Clear everything and start fresh |

The LCD shows two lines: `C:` for the current context (what you've typed and accepted) and `P:` for the pending prediction. LEDs fill cumulatively as a confidence bar (0-8) based on the margin between the top two logits.

Characters typed over serial are fed to the RNN one at a time. Each character updates the hidden state. When Predict is pressed, the RNN generates one token, displays it, and advances a *working* copy of the hidden state. The *base* state is preserved so Deny can roll back cleanly. Accept promotes the working state to base.

### The Model

A single-layer Elman RNN trained at the character level:

```
h_t = tanh(Wxh * x_t + Whh * h_{t-1})
logits = Why * h_t
prediction = argmax(logits)
```

| Parameter | Value |
|-----------|-------|
| Hidden size | 96 |
| Vocabulary | 41 characters (lowercase + digits + space, newline, `-`, `.`, `/`) |
| Wxh (input->hidden) | int8, 41x96 = 3,936 bytes |
| Whh (hidden->hidden) | int8, 96x96 = 9,216 bytes |
| Why (hidden->output) | int16, 41x96 = 7,872 bytes |
| Total weight storage | 21,024 bytes in flash (PROGMEM) |
| Hidden state | int16 (Q15 fixed-point), 192 bytes in RAM |
| Accumulators | int32, 384 bytes on stack |

The model is trained on ~619 lines of Linux shell commands (ls, cd, git, make, grep, rm, cat, etc.) with heavy repetition of common prefixes so the model learns strong prefix-to-completion associations.

### Quantization

The core technical challenge of this project: taking a float32 PyTorch model and making it produce identical predictions in pure integer arithmetic on an 8-bit microcontroller with no FPU.

**Per-matrix int8/int16 quantization.** Each weight matrix has its own scale factor. Wxh and Whh are quantized to int8 ([-127, 127]). Why is quantized to int16 ([-32767, 32767]) because it directly determines the argmax ranking — small logit margins are decisive, so the output layer benefits most from higher precision.

**WXH_SCALE reconciles the per-matrix scales.** Since Wxh and Whh have different float ranges, their int8 representations are on different scales. The firmware accumulator computes:

```
acc[i] = Wxh_q[tok][i] * WXH_SCALE + sum_j(Whh_q[j][i] * h[j])
```

WXH_SCALE = round(S_wxh * 32767 / S_whh) brings the input contribution onto the same integer scale as the recurrent contribution. No per-product shift is needed — int8 * int16 products summed over 96 terms fit comfortably in int32.

**Q_SHIFT aligns the tanh input scale.** After accumulation, the result is right-shifted before entering the piecewise tanh:

```c
h[i] = tanh_q15((acc[i] + (1L << (Q_SHIFT - 1))) >> Q_SHIFT);
```

Q_SHIFT is chosen so the firmware's tanh breakpoint (8192 in Q15) corresponds to the same activation level as the training breakpoint (0.25 in float). The formula is `Q_SHIFT = closest_to_1(log2(127 / S_whh))` — picking the integer shift whose scale factor `127 / (S_whh * 2^Q_SHIFT)` is nearest to 1.0. The `+ (1L << (Q_SHIFT - 1))` term adds rounding instead of truncation, eliminating a systematic negative bias.

**WHY_PROD_SHIFT prevents int32 overflow in the output layer.** Since Why is int16 and h is int16, each product can reach 32767^2 ~ 1.07 billion. Summed over 96 hidden units this would overflow int32. Each product is right-shifted by WHY_PROD_SHIFT=6 before accumulation. Since all logits shift equally, argmax is preserved — the shift is lossless for prediction.

**Piecewise-linear tanh in Q15.** The activation function is a three-segment approximation:

```
|x| < 8192:     identity (slope 1.0)
8192 <= |x| < 32767: interpolated (slope 0.75)
|x| >= 32767:    saturated at +/-32767
```

This runs in ~10 cycles on AVR with no lookup table. The key insight (discovered through systematic debugging — see below) is that the model must be *trained* with this same piecewise function, not with the standard `tanh`.

### Training with Firmware Tanh (USE_FW_TANH)

The single most impactful decision in the project. When `USE_FW_TANH=True`, the PyTorch training loop uses a differentiable version of the firmware's piecewise tanh instead of `torch.tanh`:

```python
def fw_tanh(x):
    bp = 8192.0 / 32767.0   # ~0.25
    x_c = torch.clamp(x, -1.0, 1.0)
    return torch.where(x_c.abs() < bp, x_c,
                       torch.sign(x_c) * (bp + (x_c.abs() - bp) * 0.75))
```

This is fully differentiable (gradients: 1.0 in linear region, 0.75 in interpolated, 0.0 at saturation) and lets the model learn weights that are adapted to the approximation it will actually run with. Without this, the model trains against `tanh(x)` but runs against a fundamentally different function — the firmware's piecewise tanh returns ~37% of the true tanh output for moderate activations (e.g., pre-activation 1.0: firmware gives 0.28, true tanh gives 0.76). Hidden states are systematically too small, weakening the recurrent connection and causing prediction failures.

### Benchmark and Diagnostic Framework

**A/B/C/D stage comparison.** To systematically attribute failures, each prefix is evaluated under four conditions:

| Stage | Weights | Activation | Purpose |
|-------|---------|------------|---------|
| A | float32 | torch.tanh | Theoretical baseline |
| B | float32 | piecewise tanh | True reference (same activation as training) |
| C | dequantized int8/16 | torch.tanh | Isolates weight quantization error |
| D | int8/16 quantized | Q15 piecewise tanh | Full firmware simulation |

When the model is trained with `USE_FW_TANH=True`, **B vs D is the primary metric** — both use the same activation function, so the comparison isolates the fixed-point arithmetic gap. A vs D conflates tanh shape difference with quantization error and is tracked as a secondary metric.

Break attribution: if A != B, the tanh approximation is the cause. If A != C, weight quantization is the cause. If A = B = C but C != D, the fixed-point state arithmetic is the cause.

**benchmark.py** runs 150 deterministic prefixes (drawn from corpus line starts, lengths 1-12) through all four stages and reports: top-1 match rates, top-3 containment, margin-weighted error, high/low-margin failure counts, hidden state drift, and hardware resource estimates. It auto-classifies the result as "WE ARE THERE" (B vs D >= 90%, MWE < 5.0, <= 2 high-margin failures) or "FURTHER IMPROVEMENT POSSIBLE" with specific diagnostics.

## Hardware

### Pinout

| Function | Pin(s) |
|----------|--------|
| SPI MOSI | PB2 |
| SPI SCLK | PB1 |
| SPI CS | PB0 |
| GLCD D/C | PF1 |
| GLCD Reset | PF0 |
| Backlight | PC7 |
| LED bar [0-4] | PB3-PB7 |
| LED bar [5-7] | PF4-PF6 |
| UART1 RX | PD2 |
| UART1 TX | PD3 |
| Predict button | PD6 |
| Accept button | PD5 |
| Deny button | PD4 |
| Reset button | PD7 |

All buttons are active-low with internal pull-ups, debounced at 20ms. The LCD is an ST7565-based 128x32 SPI GLCD running in SPI mode 3 with COM reverse (0xC8) so the LSB-first font renders correctly with page 0 at the top.

### Resource Usage (actual build)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| Flash | 24,760 bytes | 28,672 bytes (32KB - 4KB bootloader) | 86% |
| SRAM | 549 bytes | 2,560 bytes | 21% |

### Performance

| Operation | Estimated cycles | Time @ 8 MHz |
|-----------|-----------------|--------------|
| rnn_step (feed one character) | ~85,000 | ~10.6 ms |
| rnn_predict (compute argmax) | ~40,000 | ~5.0 ms |
| Total per keystroke | ~125,000 | ~15.6 ms |

## Engineering History

The path from "model outputs garbage" to "model predicts `make clean` correctly" involved 11 phases of systematic debugging. The full record is in [CHANGELOG.md](CHANGELOG.md). Key milestones:

**Phase 1-2: Capacity scaling.** Started with HIDDEN_SIZE=16 on a 165-line AVR assembly corpus. The model could only learn bigram statistics. Expanded to 96 hidden units and 619 lines of shell commands. Float32 model worked well (loss ~0.33), but quantized predictions were random.

**Phase 3-4: Quantization root cause.** A hardcoded `<< 7` shift gave the input (Wxh) contribution ~16K but the recurrent (Whh) contribution ~400M — a 950x imbalance. Input signal was completely drowned. Fixed with per-matrix int8 scales and WXH_SCALE to reconcile them in the accumulator.

**Phase 5: Why int16 upgrade.** The output layer in int8 caused close-call prediction flips. Upgrading Why to int16 cost +3.8KB flash but gave the argmax layer the precision it needs. Each product is right-shifted by WHY_PROD_SHIFT to prevent overflow; the shift is lossless for ranking.

**Phase 6-7: Firmware tanh training (key breakthrough).** Built the A/B/C/D diagnostic framework. Found that 7/10 failures were in the fixed-point path, 0 were from the tanh shape itself. The root cause: the firmware's piecewise tanh gives ~37% of the float tanh output for moderate activations. The model was trained with the wrong activation function. Training with `USE_FW_TANH=True` moved firmware match from 2/10 to 5/10 immediately.

**Phase 8-9: Q_SHIFT calibration.** Corpus-based calibration confirmed Q_SHIFT=9 was reasonable for dynamic range. Added rounded right-shift (`+ (1 << (Q_SHIFT-1))`) to eliminate systematic negative bias.

**Phase 10-11: Q_SHIFT scale-matching (critical breakthrough).** Built a 150-prefix benchmark and discovered B vs D was only 57%. Root cause: the old Q_SHIFT formula optimized dynamic range but created a 2.5x scale mismatch. The firmware's effective tanh breakpoint was at 0.63 instead of the training value of 0.25. Switching to scale-matching (`Q_SHIFT = closest_to_1(log2(127/S_whh))`) moved B vs D from 57% to 87% with no retraining. After retraining: 9/10 on the diagnostic prefix set, ~83% on the 150-prefix benchmark, 96% top-3 containment.

### Key Lessons

1. **Test float32 vs quantized separately.** Knowing which layer breaks saves enormous debugging time.
2. **Train with the activation function you deploy.** Real tanh != piecewise Q15 tanh. This single change had the largest impact.
3. **The output layer benefits most from higher precision** because it directly determines argmax and small logit margins are decisive.
4. **Q_SHIFT must match the training activation scale, not just dynamic range.** The firmware tanh breakpoint must correspond to the same real-valued activation as the training breakpoint. Getting this wrong creates a systematic scale mismatch that dominates all other error sources.
5. **Saturation % is the key diagnostic for Q_SHIFT quality.** < 1% is ideal. > 20% means values are binary (destructive). < 0.01% means dynamic range is wasted.
6. **Benchmark must compare the same activation families.** When the model is trained with fw_tanh, B vs D (both piecewise) is the correct fidelity metric. A vs D conflates tanh shape error with quantization error.

## License

Educational project. No license specified.
