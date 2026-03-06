#!/usr/bin/env python3
"""
Export trained RNN weights to C header files for ATmega32U4.

Reads model.pt and meta.json from train.py, quantizes weights:
  - Wxh, Whh: int8 per-matrix scales (WXH_SCALE reconciles in accumulator)
  - Why: int16 for maximum output-layer precision (argmax accuracy)
and emits:
  - ../firmware/weights.h  (Wxh, Whh, Why arrays in PROGMEM + constants)
  - ../firmware/vocab.h    (token-char mapping)

Usage:
    python export.py
"""

import json
import math
import os
import numpy as np
import torch

MODEL_FILE = "model.pt"
META_FILE = "meta.json"
FIRMWARE_DIR = os.path.join(os.path.dirname(__file__), "..", "firmware")

def format_array_2d(name, arr, dtype="int8_t", comment=""):
    """Format a 2D array as a C PROGMEM initializer."""
    rows, cols = arr.shape
    width = 4 if dtype == "int8_t" else 6
    lines = []
    if comment:
        lines.append(f"/* {comment} */")
    lines.append(f"static const {dtype} {name}[{rows}][{cols}] PROGMEM = {{")
    for r in range(rows):
        vals = ", ".join(f"{int(v):{width}d}" for v in arr[r])
        comma = "," if r < rows - 1 else ""
        lines.append(f"    {{{vals}}}{comma}")
    lines.append("};")
    return "\n".join(lines)

def compute_q_shift(S_whh, Wxh_q, Whh_q, wxh_scale, data, hidden_size,
                     n_seqs=300, warmup=8, collect=12):
    """
    Compute Q_SHIFT using scale-matching: Q_SHIFT = round(log2(127 / S_whh)).

    This ensures the firmware's tanh_q15 input is in correct Q15 scale so
    the effective activation breakpoint matches the training fw_tanh breakpoint
    (0.25 in float domain ↔ 8192 in Q15).

    Also runs a corpus diagnostic pass to report accumulator statistics and
    saturation percentage for the chosen Q_SHIFT.
    """
    # Pick Q_SHIFT whose scale factor (127 / (S_whh * 2^Q_SHIFT)) is closest
    # to 1.0. This matches the firmware tanh_q15 breakpoint to the training
    # fw_tanh breakpoint, avoiding both saturation (overscale) and excessive
    # linearity (underscale).
    ideal = math.log2(127.0 / S_whh)
    q_lo = max(1, int(math.floor(ideal)))
    q_hi = q_lo + 1
    scale_lo = 127.0 / (S_whh * (2 ** q_lo))
    scale_hi = 127.0 / (S_whh * (2 ** q_hi))
    q_shift = q_lo if abs(scale_lo - 1.0) < abs(scale_hi - 1.0) else q_hi

    def _tanh_q15(x):
        if x >= 32767:  return 32767
        if x <= -32768: return -32768
        xs = int(x)
        if xs >= 0:
            if xs < 8192: return xs
            else:         return 8192 + (((xs - 8192) * 3) >> 2)
        else:
            if xs > -8192: return xs
            else:          return -8192 + (((xs + 8192) * 3) >> 2)

    rng = np.random.default_rng(42)
    max_start = max(1, len(data) - warmup - collect)
    starts = rng.integers(0, max_start, size=n_seqs)

    acc_abs = []
    for start in starts:
        h = np.zeros(hidden_size, dtype=np.int16)
        for step in range(warmup + collect):
            pos = int(start) + step
            if pos >= len(data): break
            tok = int(data[pos])
            acc = np.zeros(hidden_size, dtype=np.int64)
            for i in range(hidden_size):
                acc[i] = int(Wxh_q[tok, i]) * wxh_scale
                for j in range(hidden_size):
                    acc[i] += int(Whh_q[j, i]) * int(h[j])
            if step >= warmup:
                acc_abs.extend(int(abs(acc[i])) for i in range(hidden_size))
            for i in range(hidden_size):
                h[i] = np.int16(_tanh_q15((int(acc[i]) + (1 << (q_shift - 1))) >> q_shift))

    if acc_abs:
        arr = np.array(acc_abs, dtype=np.float64)
        p50  = np.percentile(arr, 50)
        p90  = np.percentile(arr, 90)
        p95  = np.percentile(arr, 95)
        sat_thresh = 32767 * (1 << q_shift)
        pct_sat = float(np.mean(arr > sat_thresh) * 100)
        scale_factor = 127.0 / (S_whh * (2 ** q_shift))
        eff_bp = 0.25 / scale_factor

        print(f"  Q_SHIFT={q_shift}  (scale-match: log2(127/{S_whh:.4f}) = {math.log2(127.0/S_whh):.2f})")
        print(f"  Scale factor: {scale_factor:.3f}  (1.0 = perfect)")
        print(f"  Effective breakpoint: {eff_bp:.3f}  (training: 0.250)")
        print(f"  Corpus |acc|: p50={p50:,.0f}  p90={p90:,.0f}  p95={p95:,.0f}")
        print(f"  Saturation: {pct_sat:.1f}%")

    return q_shift


def export_weights():
    # Load metadata
    with open(META_FILE, "r") as f:
        meta = json.load(f)

    vocab_size = meta["vocab_size"]
    hidden_size = meta["hidden_size"]
    chars = meta["chars"]

    # Load model
    state = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)

    # Extract weight matrices as numpy
    Wxh_np = state["Wxh.weight"].T.numpy()   # (vocab, hidden)
    Whh_np = state["Whh.weight"].T.numpy()   # (hidden, hidden)
    Why_np = state["Why.weight"].numpy()      # (vocab, hidden)

    # ── Per-matrix quantization ──────────────────────────────
    #
    # Wxh, Whh: int8 per-matrix scales.
    # WXH_SCALE reconciles scales in the firmware accumulator:
    #   acc = Wxh_q[tok][i] * WXH_SCALE + sum(Whh_q[j][i] * h[j])
    #
    # Why: int16 for maximum output precision (argmax accuracy).
    # Each product is >> WHY_PROD_SHIFT to prevent int32 overflow.
    # Since only argmax matters, the uniform shift is lossless.

    S_wxh = np.abs(Wxh_np).max()
    S_whh = np.abs(Whh_np).max()
    S_why = np.abs(Why_np).max()

    Wxh_q = np.clip(np.round(Wxh_np * 127.0 / S_wxh), -127, 127).astype(np.int8)
    Whh_q = np.clip(np.round(Whh_np * 127.0 / S_whh), -127, 127).astype(np.int8)
    Why_q = np.clip(np.round(Why_np * 32767.0 / S_why), -32767, 32767).astype(np.int16)

    print(f"Wxh: {Wxh_q.shape}  max|float|={S_wxh:.4f}  (int8, own scale)")
    print(f"Whh: {Whh_q.shape}  max|float|={S_whh:.4f}  (int8, own scale)")
    print(f"Why: {Why_q.shape}  max|float|={S_why:.4f}  (int16, own scale)")
    total_bytes = Wxh_q.size + Whh_q.size + Why_q.size * 2
    print(f"Total weight bytes: {total_bytes}")

    # ── Compute WXH_SCALE (reconcile per-matrix scales) ──────
    #
    # In the accumulator: acc = Wxh_q * WXH_SCALE + sum(Whh_q * h)
    # Whh_q * h has implicit scale 127 * 32767 / S_whh
    # Wxh_q * WXH_SCALE should match:
    #   127 * WXH_SCALE / S_wxh = 127 * 32767 / S_whh
    #   WXH_SCALE = S_wxh * 32767 / S_whh
    wxh_scale = round(S_wxh * 32767.0 / S_whh)

    # ── Compute Q_SHIFT via scale matching ──────────────────────
    #
    # Q_SHIFT = round(log2(127 / S_whh))
    # This ensures: acc >> Q_SHIFT ≈ act * 32767 (correct Q15 scale),
    # so tanh_q15 breakpoint (8192) matches training fw_tanh breakpoint (0.25).
    # Also runs corpus diagnostic pass for saturation statistics.

    corpus_path = os.path.join(os.path.dirname(__file__), "corpus.txt")
    char2idx_cal = {c: i for i, c in enumerate(chars)}
    with open(corpus_path, "r") as f:
        corpus_text = f.read()
    corpus_data = np.array([char2idx_cal[c] for c in corpus_text if c in char2idx_cal],
                           dtype=np.int32)

    print("Computing Q_SHIFT (scale-matching)...")
    q_shift = compute_q_shift(float(S_whh), Wxh_q, Whh_q, wxh_scale,
                               corpus_data, hidden_size)

    # ── Compute WHY_PROD_SHIFT (prevent int32 overflow in logits) ──
    #
    # Why is int16, h is int16: max product = 32767*32767 ≈ 1.07B
    # Summed over HIDDEN_SIZE: can overflow int32.
    # >> WHY_PROD_SHIFT per product, argmax is preserved.
    why_ps = max(1, math.ceil(math.log2(hidden_size * 32767.0 * 32767.0 / (2**31 - 1))))

    # Overflow checks
    max_wxh_term = 127 * wxh_scale
    max_whh_sum = hidden_size * 127 * 32767
    max_rnn_acc = max_wxh_term + max_whh_sum
    max_why_logit = hidden_size * ((32767 * 32767) >> why_ps)

    print(f"WXH_SCALE:       {wxh_scale}")
    print(f"Q_SHIFT:         {q_shift}")
    print(f"WHY_PROD_SHIFT:  {why_ps}")
    print(f"Max rnn acc:     {max_rnn_acc:,} (int32 max: 2,147,483,647) {'OK' if max_rnn_acc < 2**31 else 'OVERFLOW!'}")
    print(f"Max why logit:   {max_why_logit:,} {'OK' if max_why_logit < 2**31 else 'OVERFLOW!'}")
    assert max_rnn_acc < 2**31, f"RNN accumulator overflow! {max_rnn_acc}"
    assert max_why_logit < 2**31, f"Why logit overflow! {max_why_logit}"

    # ── Write weights.h ────────────────────────────────────────

    weights_path = os.path.join(FIRMWARE_DIR, "weights.h")
    with open(weights_path, "w") as f:
        f.write("#ifndef WEIGHTS_H\n")
        f.write("#define WEIGHTS_H\n\n")
        f.write("#include \"config.h\"\n")
        f.write("#include <avr/pgmspace.h>\n")
        f.write("#include <stdint.h>\n\n")
        f.write("/*\n")
        f.write(" * RNN weight matrices — Wxh/Whh int8, Why int16.\n")
        f.write(f" * Generated by export.py from {MODEL_FILE}\n")
        f.write(f" * Vocab: {vocab_size}, Hidden: {hidden_size}\n")
        f.write(f" * Scales: Wxh={S_wxh:.4f}  Whh={S_whh:.4f}  Why={S_why:.4f}\n")
        f.write(f" * WXH_SCALE={wxh_scale}  Q_SHIFT={q_shift}  WHY_PROD_SHIFT={why_ps}\n")
        f.write(" */\n\n")
        f.write("/* Firmware constants (computed by export.py) */\n")
        f.write(f"#define WXH_SCALE        {wxh_scale}L\n")
        f.write(f"#define Q_SHIFT          {q_shift}\n")
        f.write(f"#define WHY_PROD_SHIFT   {why_ps}\n\n")

        f.write(format_array_2d("Wxh", Wxh_q, "int8_t",
                f"Input->Hidden  ({vocab_size}x{hidden_size})") + "\n\n")
        f.write(format_array_2d("Whh", Whh_q, "int8_t",
                f"Hidden->Hidden ({hidden_size}x{hidden_size})") + "\n\n")
        f.write(format_array_2d("Why", Why_q, "int16_t",
                f"Hidden->Output ({vocab_size}x{hidden_size})") + "\n\n")

        f.write("#endif /* WEIGHTS_H */\n")

    print(f"Wrote {weights_path}")

    # ── Write vocab.h ──────────────────────────────────────────

    vocab_path = os.path.join(FIRMWARE_DIR, "vocab.h")
    with open(vocab_path, "w") as f:
        f.write("#ifndef VOCAB_H\n")
        f.write("#define VOCAB_H\n\n")
        f.write("#include <avr/pgmspace.h>\n")
        f.write("#include <stdint.h>\n\n")
        f.write("/*\n")
        f.write(f" * Token/char mapping — {vocab_size} tokens.\n")
        f.write(f" * Generated by export.py from {META_FILE}\n")
        f.write(" */\n\n")

        f.write(f"static const char vocab_chars[{vocab_size}] PROGMEM = {{\n")
        for i, ch in enumerate(chars):
            comma = "," if i < len(chars) - 1 else ""
            if ch == '\n':
                f.write(f"    '\\n'{comma}  /* {i:2d} */\n")
            elif ch == '\r':
                f.write(f"    '\\r'{comma}  /* {i:2d} */\n")
            elif ch == '\'':
                f.write(f"    '\\''{comma}  /* {i:2d} */\n")
            elif ch == '\\':
                f.write(f"    '\\\\'{comma}  /* {i:2d} */\n")
            else:
                f.write(f"    '{ch}'{comma}   /* {i:2d} */\n")
        f.write("};\n\n")

        f.write("static inline uint8_t vocab_encode(char ch)\n")
        f.write("{\n")
        f.write(f"    for (uint8_t i = 0; i < {vocab_size}; i++) {{\n")
        f.write("        if (pgm_read_byte(&vocab_chars[i]) == (uint8_t)ch)\n")
        f.write("            return i;\n")
        f.write("    }\n")
        f.write("    return 0;\n")
        f.write("}\n\n")

        f.write("static inline char vocab_decode(uint8_t idx)\n")
        f.write("{\n")
        f.write(f"    if (idx >= {vocab_size}) return '?';\n")
        f.write("    return (char)pgm_read_byte(&vocab_chars[idx]);\n")
        f.write("}\n\n")

        f.write("#endif /* VOCAB_H */\n")

    print(f"Wrote {vocab_path}")

    # ── Update config.h ───────────────────────────────────────

    config_path = os.path.join(FIRMWARE_DIR, "config.h")
    with open(config_path, "r") as f:
        config = f.read()

    import re
    new_config = re.sub(
        r"#define VOCAB_SIZE\s+\d+",
        f"#define VOCAB_SIZE   {vocab_size}",
        config
    )
    new_config = re.sub(
        r"#define HIDDEN_SIZE\s+\d+",
        f"#define HIDDEN_SIZE  {hidden_size}",
        new_config
    )
    if new_config != config:
        with open(config_path, "w") as f:
            f.write(new_config)
        print(f"Updated VOCAB_SIZE={vocab_size}, HIDDEN_SIZE={hidden_size} in config.h")

if __name__ == "__main__":
    export_weights()
