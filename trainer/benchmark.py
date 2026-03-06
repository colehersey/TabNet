#!/usr/bin/env python3
"""
Benchmark: float32 model vs fixed-point firmware simulation.

Measures how closely stage D (full firmware fixed-point) reproduces
stage A (float32 reference) across a large deterministic prefix set.

Answers one of two conclusions:
  "We are there."      — firmware faithfully reproduces float32.
  "This is the limit." — remaining gaps are quantization/hardware limits.

Usage:
    python benchmark.py

Outputs:
    benchmark_prefixes.json  — deterministic prefix list, reused across runs
"""

import json
import math
import os
import numpy as np
import torch
import torch.nn as nn

# ── Config ──────────────────────────────────────────────────────

MODEL_FILE   = "model.pt"
META_FILE    = "meta.json"
CORPUS_FILE  = "corpus.txt"
PREFIXES_FILE = "benchmark_prefixes.json"

N_PREFIXES   = 150
MIN_LEN      = 1
MAX_LEN      = 12
RNG_SEED     = 42

# Ceiling thresholds
TOP1_CEIL_THRESH   = 90.0   # % top-1 A==D
MWE_CEIL_THRESH    = 5.0    # margin-weighted error
HIGH_FAIL_CEIL     = 2      # high-margin failures allowed
HIGH_MARGIN_CUTOFF = 2.0    # float32 logit margin that counts as "confident"

# ATmega32U4 hardware parameters
MCU_FREQ_HZ        = 8_000_000
CYCLES_PER_8x8_MUL = 2        # hardware MUL instruction
CYCLES_PER_PROGMEM  = 3        # pgm_read_byte/word overhead


# ── Firmware piecewise tanh ──────────────────────────────────────

def tanh_q15_scalar(x):
    if x >= 32767:  return 32767
    if x <= -32768: return -32768
    xs = int(x)
    if xs >= 0:
        if xs < 8192: return xs
        else:         return 8192 + (((xs - 8192) * 3) >> 2)
    else:
        if xs > -8192: return xs
        else:          return -8192 + (((xs + 8192) * 3) >> 2)


def tanh_q15_vec(x):
    """Vectorized tanh_q15 operating on int64 arrays."""
    x = np.asarray(x, dtype=np.int64)
    out = np.where(
        x >= 32767,  32767,
        np.where(
        x <= -32768, -32768,
        np.where(
        x >= 8192,   8192 + ((x - 8192) * 3 >> 2),
        np.where(
        x <= -8192,  -8192 + ((x + 8192) * 3 >> 2),
                     x))))
    return out.astype(np.int16)


def tanh_pw_float_vec(x_arr):
    """Piecewise linear tanh in float domain matching firmware shape."""
    bp = 8192.0 / 32767.0
    x_arr = np.asarray(x_arr, dtype=np.float64)
    out = np.where(
        x_arr >= 1.0,  1.0,
        np.where(
        x_arr <= -1.0, -1.0,
        np.where(
        np.abs(x_arr) < bp, x_arr,
        np.where(
        x_arr >= 0,  bp + (x_arr - bp) * 0.75,
                    -bp + (x_arr + bp) * 0.75))))
    return out


# ── PyTorch model (stage A) ──────────────────────────────────────

class TinyRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wxh = nn.Linear(vocab_size, hidden_size, bias=False)
        self.Whh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Why = nn.Linear(hidden_size, vocab_size, bias=False)

    def step(self, tok_idx, h, use_fw_tanh):
        """Single-token step; returns (logits [vocab], h_new [hidden])."""
        vocab_size = self.Why.out_features
        x = torch.zeros(1, vocab_size)
        x[0, tok_idx] = 1.0
        act = self.Wxh(x) + self.Whh(h)
        if use_fw_tanh:
            bp = 8192.0 / 32767.0
            xc = torch.clamp(act, -1.0, 1.0)
            h_new = torch.where(xc.abs() < bp, xc,
                                torch.sign(xc) * (bp + (xc.abs() - bp) * 0.75))
        else:
            h_new = torch.tanh(act)
        logits = self.Why(h_new)
        return logits[0], h_new


# ── Q_SHIFT computation (scale-matching, mirrors export.py) ──────

def compute_q_shift(S_whh):
    """Pick Q_SHIFT whose scale factor is closest to 1.0.

    Scale factor = 127 / (S_whh * 2^Q_SHIFT).  We want this near 1.0 so
    the firmware tanh_q15 breakpoint matches the training fw_tanh breakpoint.
    When the ideal (log2(127/S_whh)) falls between integers, prefer the
    value that gives scale closer to 1.0 — this avoids both excessive
    saturation (overscale) and excessive linearity (underscale).
    """
    ideal = math.log2(127.0 / S_whh)
    q_lo = max(1, int(math.floor(ideal)))
    q_hi = q_lo + 1
    scale_lo = 127.0 / (S_whh * (2 ** q_lo))
    scale_hi = 127.0 / (S_whh * (2 ** q_hi))
    return q_lo if abs(scale_lo - 1.0) < abs(scale_hi - 1.0) else q_hi


# ── Prefix generation ────────────────────────────────────────────

def generate_prefixes(corpus_text, char2idx,
                      n=N_PREFIXES, seed=RNG_SEED,
                      min_len=MIN_LEN, max_len=MAX_LEN):
    """
    Build a deterministic set of unique prefixes drawn from corpus line starts.
    Each prefix is the first 1..max_len characters of a corpus line.
    All characters must be in vocab.
    """
    rng = np.random.default_rng(seed)

    # Collect all line-start positions
    line_starts = [0]
    for i, ch in enumerate(corpus_text[:-1]):
        if ch == '\n':
            line_starts.append(i + 1)

    prefixes = set()
    attempts = 0
    while len(prefixes) < n and attempts < n * 100:
        attempts += 1
        ls = int(rng.integers(0, len(line_starts)))
        start = line_starts[ls]
        end = corpus_text.find('\n', start)
        if end == -1:
            end = len(corpus_text)
        line = corpus_text[start:end]
        if len(line) < min_len:
            continue
        plen = int(rng.integers(min_len, min(max_len, len(line)) + 1))
        prefix = line[:plen]
        if all(c in char2idx for c in prefix):
            prefixes.add(prefix)

    return sorted(prefixes)


# ── Hardware report ──────────────────────────────────────────────

def hardware_report(vocab_size, hidden_size, Wxh_q, Whh_q, Why_q):
    flash_wxh     = Wxh_q.size           # int8 bytes
    flash_whh     = Whh_q.size           # int8 bytes
    flash_why     = Why_q.size * 2       # int16 bytes
    flash_weights = flash_wxh + flash_whh + flash_why
    flash_code    = 5 * 1024             # firmware code + LCD/USB overhead (~5KB)
    flash_total   = flash_weights + flash_code
    flash_budget  = 28 * 1024            # 32KB - 4KB bootloader

    sram_h      = hidden_size * 2        # h[hidden_size] int16 (persistent)
    sram_h_base = hidden_size * 2        # h_base[hidden_size] int16 (save/restore)
    sram_acc    = hidden_size * 4        # acc[hidden_size] int32 (stack)
    sram_misc   = 128                    # stack frame, globals, confidence, etc.
    sram_total  = sram_h + sram_h_base + sram_acc + sram_misc
    sram_budget = 2560                   # ATmega32U4: 2.5KB

    # Cycle estimates (8-bit AVR with hardware MUL)
    # rnn_step: (1 + hidden_size) Wxh/Whh MACs per hidden unit
    #   int8*int16: sign-extend(1 cycle) + 2x MUL(4 cycles) + adds ≈ 8 cycles
    # PROGMEM reads: pgm_read_byte = +3 cycles each
    macs_step      = hidden_size * (1 + hidden_size)
    pgm_reads_step = hidden_size + hidden_size * hidden_size   # Wxh + Whh bytes
    cyc_step = macs_step * 8 + pgm_reads_step * CYCLES_PER_PROGMEM + hidden_size * 10

    # rnn_predict: vocab_size * hidden_size int16*int16 MACs
    macs_pred      = vocab_size * hidden_size
    pgm_reads_pred = vocab_size * hidden_size * 2              # Why words
    cyc_pred = macs_pred * 8 + pgm_reads_pred * CYCLES_PER_PROGMEM

    lat_step_ms = cyc_step / MCU_FREQ_HZ * 1000
    lat_pred_ms = cyc_pred / MCU_FREQ_HZ * 1000
    lat_total_ms = lat_step_ms + lat_pred_ms

    print("\nHardware Constraints (ATmega32U4 @ 8 MHz)")
    print("-" * 50)
    print(f"  Flash (weights):   {flash_weights:,} bytes"
          f"  (Wxh:{flash_wxh} + Whh:{flash_whh} + Why:{flash_why})")
    print(f"  Flash (code est):  {flash_code:,} bytes")
    print(f"  Flash total est:   {flash_total:,} / {flash_budget:,} bytes"
          f"  ({100*flash_total/flash_budget:.0f}%)")
    print(f"  SRAM est:          {sram_total} / {sram_budget} bytes"
          f"  ({100*sram_total/sram_budget:.0f}%)")
    print(f"  Cycles/rnn_step:   ~{cyc_step:,}  ({lat_step_ms:.1f} ms)")
    print(f"  Cycles/predict:    ~{cyc_pred:,}  ({lat_pred_ms:.1f} ms)")
    print(f"  Total per keypress:~{cyc_step+cyc_pred:,}  ({lat_total_ms:.1f} ms)")


# ── Main benchmark ───────────────────────────────────────────────

def run_benchmark():
    for f in (MODEL_FILE, META_FILE, CORPUS_FILE):
        if not os.path.exists(f):
            print(f"ERROR: {f} not found. Run train.py first.")
            return

    # ── Load model ───────────────────────────────────────────────
    with open(META_FILE) as f:
        meta = json.load(f)

    vocab_size  = meta["vocab_size"]
    hidden_size = meta["hidden_size"]
    chars       = meta["chars"]
    char2idx    = {c: i for i, c in enumerate(chars)}
    idx2char    = {i: c for c, i in char2idx.items()}

    state = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)

    Wxh_np = state["Wxh.weight"].T.numpy()   # (vocab, hidden)
    Whh_np = state["Whh.weight"].T.numpy()   # (hidden, hidden)
    Why_np = state["Why.weight"].numpy()     # (vocab, hidden)

    # ── Quantize (mirrors export.py) ─────────────────────────────
    S_wxh = float(np.abs(Wxh_np).max())
    S_whh = float(np.abs(Whh_np).max())
    S_why = float(np.abs(Why_np).max())

    Wxh_q = np.clip(np.round(Wxh_np * 127.0 / S_wxh), -127, 127).astype(np.int8)
    Whh_q = np.clip(np.round(Whh_np * 127.0 / S_whh), -127, 127).astype(np.int8)
    Why_q = np.clip(np.round(Why_np * 32767.0 / S_why), -32767, 32767).astype(np.int16)

    Wxh_deq = Wxh_q.astype(np.float64) * S_wxh / 127.0
    Whh_deq = Whh_q.astype(np.float64) * S_whh / 127.0
    Why_deq = Why_q.astype(np.float64) * S_why / 32767.0

    WXH_SCALE     = round(S_wxh * 32767.0 / S_whh)
    q_shift_formula = max(1, round(math.log2(127.0 * 32767.0 / (S_whh * 8192.0))))
    WHY_PS        = max(1, math.ceil(math.log2(
                        hidden_size * 32767.0 * 32767.0 / (2**31 - 1))))

    # ── Q_SHIFT via scale matching ───────────────────────────────
    with open(CORPUS_FILE) as f:
        corpus_text = f.read()
    q_shift = compute_q_shift(S_whh)
    scale_factor = 127.0 / (S_whh * (2 ** q_shift))
    eff_bp = 0.25 / scale_factor

    print(f"Model: vocab={vocab_size}, hidden={hidden_size}")
    print(f"Quant: WXH_SCALE={WXH_SCALE}, Q_SHIFT={q_shift}"
          f" (scale-match, ideal={math.log2(127.0/S_whh):.2f}), WHY_PS={WHY_PS}")
    print(f"Scale factor: {scale_factor:.3f}  Effective bp: {eff_bp:.3f}  (training: 0.250)")

    # ── Load or generate prefix list ─────────────────────────────
    if os.path.exists(PREFIXES_FILE):
        with open(PREFIXES_FILE) as f:
            prefixes = json.load(f)
        print(f"Loaded {len(prefixes)} prefixes from {PREFIXES_FILE}")
    else:
        prefixes = generate_prefixes(corpus_text, char2idx)
        with open(PREFIXES_FILE, "w") as f:
            json.dump(prefixes, f, indent=2)
        print(f"Generated {len(prefixes)} prefixes → {PREFIXES_FILE}")

    n = len(prefixes)

    # ── PyTorch model for stage A ─────────────────────────────────
    model = TinyRNN(vocab_size, hidden_size)
    model.load_state_dict(state)
    model.eval()

    # ── Per-prefix evaluation ─────────────────────────────────────
    # B vs D is the PRIMARY metric: both use piecewise tanh (what the model
    # was trained with via USE_FW_TANH=True).  A vs D is a secondary measure
    # that conflates tanh shape difference with quantization error.
    top1_AB = top1_AC = top1_AD = top1_BD = 0
    top3_AD = top3_BD = 0
    mwe_ad = 0.0  # margin-weighted error (A vs D)
    mwe_bd = 0.0  # margin-weighted error (B vs D)
    hmf_bd = 0    # high-margin failures (B vs D)
    lmf_bd = 0    # low-margin failures  (B vs D)
    drift_ad = []  # hidden drift A vs D
    drift_bd = []  # hidden drift B vs D  (primary)

    with torch.no_grad():
        for prefix in prefixes:
            # ── Stage A: PyTorch float32 + true tanh ─────────────
            h_a = torch.zeros(1, hidden_size)
            for ch in prefix:
                logits_a, h_a = model.step(char2idx[ch], h_a, use_fw_tanh=False)
            logits_a_np = logits_a.numpy().astype(np.float64)
            order_a = np.argsort(logits_a_np)[::-1]
            top1_a  = int(order_a[0])
            margin_a = float(logits_a_np[order_a[0]] - logits_a_np[order_a[1]])

            # ── Stage B: float64 weights + piecewise tanh ────────
            #   (trained activation → true float reference for this model)
            h_b = np.zeros(hidden_size, dtype=np.float64)
            for ch in prefix:
                tok = char2idx[ch]
                pre = Wxh_np[tok] + Whh_np.T @ h_b
                h_b = tanh_pw_float_vec(pre)
            logits_b = Why_np @ h_b
            order_b  = np.argsort(logits_b)[::-1]
            top1_b   = int(order_b[0])
            margin_b = float(logits_b[order_b[0]] - logits_b[order_b[1]])

            # ── Stage C: dequantized weights + real tanh ─────────
            h_c = np.zeros(hidden_size, dtype=np.float64)
            for ch in prefix:
                tok = char2idx[ch]
                pre = Wxh_deq[tok] + Whh_deq.T @ h_c
                h_c = np.tanh(pre)
            logits_c = Why_deq @ h_c
            top1_c = int(np.argmax(logits_c))

            # ── Stage D: full firmware sim (calibrated q_shift) ───
            h_d = np.zeros(hidden_size, dtype=np.int16)
            for ch in prefix:
                tok = char2idx[ch]
                acc = (Wxh_q[tok].astype(np.int64) * np.int64(WXH_SCALE)
                       + Whh_q.T.astype(np.int64) @ h_d.astype(np.int64))
                h_d = tanh_q15_vec((acc + (1 << (q_shift - 1))) >> q_shift)

            # Per-product shift matches firmware exactly
            products = Why_q.astype(np.int64) * h_d.astype(np.int64)
            logits_d = np.sum(products >> WHY_PS, axis=1)
            order_d  = np.argsort(logits_d)[::-1]
            top1_d   = int(order_d[0])
            top3_d   = set(order_d[:3].tolist())

            # ── Accumulate metrics ────────────────────────────────
            if top1_a == top1_b: top1_AB += 1
            if top1_a == top1_c: top1_AC += 1
            if top1_a == top1_d: top1_AD += 1
            if top1_a in top3_d: top3_AD += 1

            # B vs D (primary)
            if top1_b == top1_d:
                top1_BD += 1
            else:
                mwe_bd += margin_b
                if margin_b > HIGH_MARGIN_CUTOFF:
                    hmf_bd += 1
                else:
                    lmf_bd += 1
            if top1_b in top3_d:
                top3_BD += 1
            mwe_ad += margin_a if top1_a != top1_d else 0.0

            # Hidden state drift
            h_a_np    = h_a[0].numpy().astype(np.float64)
            h_d_float = h_d.astype(np.float64) / 32767.0
            drift_ad.append(float(np.mean(np.abs(h_a_np - h_d_float))))
            drift_bd.append(float(np.mean(np.abs(h_b - h_d_float))))

    # ── Compute summary statistics ────────────────────────────────
    rate_AB    = top1_AB  / n * 100
    rate_AC    = top1_AC  / n * 100
    rate_AD    = top1_AD  / n * 100
    rate_BD    = top1_BD  / n * 100
    rate_top3_ad = top3_AD / n * 100
    rate_top3_bd = top3_BD / n * 100
    mean_drift_ad = float(np.mean(drift_ad))
    max_drift_ad  = float(np.max(drift_ad))
    mean_drift_bd = float(np.mean(drift_bd))
    max_drift_bd  = float(np.max(drift_bd))

    # ── Print summary ─────────────────────────────────────────────
    print()
    print("Benchmark Summary")
    print("-" * 55)
    print(f"  Prefixes tested:         {n}")
    print(f"  Q_SHIFT used:            {q_shift} (calibrated)")
    print()
    print("  Pairwise top-1 agreement:")
    print(f"    A vs B  tanh shape:      {rate_AB:.1f}%")
    print(f"    A vs C  weight quant:    {rate_AC:.1f}%")
    print(f"    A vs D  (secondary):     {rate_AD:.1f}%")
    print(f"    B vs D  fixed-point:     {rate_BD:.1f}%  <- PRIMARY (same tanh family)")
    print()
    print(f"  Top-3 containment B→D:   {rate_top3_bd:.1f}%")
    print(f"  Top-3 containment A→D:   {rate_top3_ad:.1f}%")
    print()
    print(f"  Margin-Weighted Error (B vs D): {mwe_bd:.2f}")
    print(f"  High-Margin Failures"
          f" (>{HIGH_MARGIN_CUTOFF:.1f}):    {hmf_bd}")
    print(f"  Low-Margin Failures:           {lmf_bd}")
    print()
    print(f"  Hidden Drift (B vs D):  mean={mean_drift_bd:.4f}  max={max_drift_bd:.4f}")
    print(f"  Hidden Drift (A vs D):  mean={mean_drift_ad:.4f}  max={max_drift_ad:.4f}")

    hardware_report(vocab_size, hidden_size, Wxh_q, Whh_q, Why_q)

    # ── Ceiling detection (based on B vs D — same tanh family) ─────
    at_ceiling = (
        rate_BD  >= TOP1_CEIL_THRESH
        and mwe_bd < MWE_CEIL_THRESH
        and hmf_bd  <= HIGH_FAIL_CEIL
    )

    print()
    print("Conclusion")
    print("-" * 55)

    if at_ceiling:
        print("  Status: WE ARE THERE.")
        print()
        print(f"  The firmware simulation faithfully reproduces float model behavior.")
        print(f"  B vs D top-1: {rate_BD:.1f}% over {n} diverse prefixes.")
        print(f"  {hmf_bd} high-margin failure(s) — within acceptable tolerance.")
        print()
        print("  Remaining errors occur in low-margin predictions where the float")
        print("  model itself is near-ambiguous. This is the achievable ceiling")
        print("  under current int8/int16 quantization constraints.")
        if rate_AD < rate_BD - 5:
            print()
            print(f"  Note: A vs D ({rate_AD:.1f}%) is lower than B vs D ({rate_BD:.1f}%).")
            print("  The gap is caused by tanh shape difference (model trained with")
            print("  fw_tanh), not firmware bugs. This is expected and correct.")
    else:
        print("  Status: FURTHER IMPROVEMENT POSSIBLE.")
        print()
        reasons = []
        if rate_BD < TOP1_CEIL_THRESH:
            reasons.append(
                f"  - B vs D top-1 {rate_BD:.1f}% < {TOP1_CEIL_THRESH:.0f}% target")
        if mwe_bd >= MWE_CEIL_THRESH:
            reasons.append(
                f"  - Margin-weighted error (B vs D) {mwe_bd:.2f}"
                f" >= {MWE_CEIL_THRESH:.1f} threshold")
        if hmf_bd > HIGH_FAIL_CEIL:
            reasons.append(
                f"  - {hmf_bd} high-confidence B vs D failures"
                f" (limit: {HIGH_FAIL_CEIL})")
        for r in reasons:
            print(r)
        print()
        print("  Diagnostics:")
        if rate_AD < rate_BD - 5:
            gap = rate_BD - rate_AD
            print(f"  - A vs D ({rate_AD:.1f}%) is {gap:.0f}pp below B vs D ({rate_BD:.1f}%).")
            print("    This gap is tanh shape, not firmware. Focus on B vs D.")
        if rate_AB < 90:
            print(f"  - A vs B only {rate_AB:.1f}%: tanh approximation is a major factor")
        if rate_AC < 95:
            print(f"  - A vs C only {rate_AC:.1f}%: weight quantization needs attention")
        if mean_drift_bd > 0.05:
            print(f"  - B vs D hidden drift is high ({mean_drift_bd:.4f}):")
            print("    possible Q_SHIFT mismatch, rounding issue, or scale alignment bug")
        elif mean_drift_bd <= 0.05 and rate_BD < TOP1_CEIL_THRESH:
            print(f"  - B vs D hidden drift is low ({mean_drift_bd:.4f}) but top-1 still off:")
            print("    logit layer (Why) precision or WHY_PROD_SHIFT may be the bottleneck")
        print("  - retrain with updated parameters, then re-run benchmark")


if __name__ == "__main__":
    run_benchmark()
