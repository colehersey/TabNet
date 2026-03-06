#!/usr/bin/env python3
"""
Tiny RNN trainer for the neural sequence generator.

Trains a single-layer RNN (hidden_size=16) on a character-level corpus
of AVR assembly, then quantizes weights to int8 for ATmega32U4 deployment.

Usage:
    python train.py
    python export.py   # generates weights.h and vocab.h
"""

import json
import numpy as np
import torch
import torch.nn as nn

# ── Config ─────────────────────────────────────────────────────

HIDDEN_SIZE = 96
SEQ_LEN = 32
BATCH_SIZE = 16
EPOCHS = 5000
LR = 0.003
CORPUS_FILE = "corpus.txt"
MODEL_FILE = "model.pt"
META_FILE = "meta.json"

# Set True to train with the firmware's piecewise tanh instead of real tanh.
# The model learns weights that work with the Q15 approximation, closing the
# A/D gap without needing to change the quantization math.
USE_FW_TANH = True

# ── Load corpus ────────────────────────────────────────────────

def load_corpus(path):
    with open(path, "r") as f:
        text = f.read()
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    data = [char2idx[c] for c in text]
    return text, chars, char2idx, idx2char, data

# ── Firmware-matching piecewise tanh ───────────────────────────

def fw_tanh(x):
    """
    Differentiable piecewise linear tanh matching the firmware Q15 approximation.
    Breakpoints at ±(8192/32767) ≈ ±0.25, slope 0.75 above, clamp at ±1.
    Gradients: 1.0 in linear region, 0.75 in interpolated, 0.0 at saturation.
    """
    bp = 8192.0 / 32767.0
    x_c = torch.clamp(x, -1.0, 1.0)
    return torch.where(
        x_c.abs() < bp,
        x_c,
        torch.sign(x_c) * (bp + (x_c.abs() - bp) * 0.75)
    )

# ── Model ──────────────────────────────────────────────────────

class TinyRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wxh = nn.Linear(vocab_size, hidden_size, bias=False)
        self.Whh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Why = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x_seq, h=None):
        """
        x_seq: (batch, seq_len) integer token indices
        Returns: logits (batch, seq_len, vocab_size), final h
        """
        batch, seq_len = x_seq.shape
        vocab_size = self.Why.out_features

        if h is None:
            h = torch.zeros(batch, self.hidden_size, device=x_seq.device)

        logits = []
        for t in range(seq_len):
            x_onehot = torch.zeros(batch, vocab_size, device=x_seq.device)
            x_onehot.scatter_(1, x_seq[:, t:t+1], 1.0)

            act = self.Wxh(x_onehot) + self.Whh(h)
            h = fw_tanh(act) if USE_FW_TANH else torch.tanh(act)
            logits.append(self.Why(h))

        return torch.stack(logits, dim=1), h

# ── Training ───────────────────────────────────────────────────

def make_batches(data, seq_len, batch_size):
    """Create input/target pairs for training."""
    n = len(data) - seq_len
    if n < batch_size:
        batch_size = max(1, n)

    indices = np.random.randint(0, n, size=batch_size)
    x = np.array([data[i:i+seq_len] for i in indices])
    y = np.array([data[i+1:i+seq_len+1] for i in indices])
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def train():
    text, chars, char2idx, idx2char, data = load_corpus(CORPUS_FILE)
    vocab_size = len(chars)

    print(f"Corpus: {len(text)} chars, {vocab_size} unique tokens")
    print(f"Vocab: {''.join(repr(c) for c in chars)}")
    print(f"Model: hidden_size={HIDDEN_SIZE}, params={vocab_size*HIDDEN_SIZE*2 + HIDDEN_SIZE**2}")

    model = TinyRNN(vocab_size, HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        x, y = make_batches(data, SEQ_LEN, BATCH_SIZE)
        logits, _ = model(x)

        # Reshape for cross-entropy: (batch*seq, vocab)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1:4d}/{EPOCHS}  loss={loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_FILE)

    # Save metadata
    meta = {
        "vocab_size": vocab_size,
        "hidden_size": HIDDEN_SIZE,
        "chars": chars,
        "char2idx": char2idx,
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {MODEL_FILE} and {META_FILE}")
    print(f"Vocab size: {vocab_size}")

    # ── Diagnostic: prefix completion test (float32) ─────────
    print("\n--- Prefix completion test (float32 model) ---")
    model.eval()
    test_prefixes = [
        ("ls",    "expect ' ' or '\\n'"),
        ("cd ",   "expect dir letter"),
        ("man ",  "expect command letter"),
        ("tar ",  "expect '-'"),
        ("git ",  "expect 's','d','a','l','c','p'"),
        ("rm ",   "expect '-' or file letter"),
        ("rm -",  "expect 'f' or 'r'"),
        ("make",  "expect ' ' or '\\n'"),
        ("grep ", "expect '-'"),
        ("cat ",  "expect file letter"),
    ]
    float32_margins = {}
    float32_top1 = {}
    with torch.no_grad():
        for prefix, expected in test_prefixes:
            h = torch.zeros(1, HIDDEN_SIZE)
            for ch in prefix:
                tok = char2idx[ch]
                x = torch.tensor([[tok]], dtype=torch.long)
                logits, h = model(x, h)
            raw = logits[0, -1]
            top2_vals, top2_idx = torch.topk(raw, 2)
            margin_f = (top2_vals[0] - top2_vals[1]).item()
            float32_margins[prefix] = margin_f
            float32_top1[prefix] = idx2char[top2_idx[0].item()]
            probs = torch.softmax(raw, dim=0)
            top5 = torch.topk(probs, 5)
            preds = []
            for i, p in zip(top5.indices, top5.values):
                c = idx2char[i.item()]
                preds.append(f"'{repr(c)[1:-1]}':{p.item():.2f}")
            print(f"  '{prefix}' -> [{', '.join(preds)}]  margin={margin_f:.2f}  ({expected})")

    # ── Diagnostic: A/B/C/D stage comparison ─────────────────
    # A = float32 + real tanh (baseline, already computed above)
    # B = float32 weights + firmware piecewise tanh
    # C = dequantized int8/16 weights + real tanh (isolates weight quant)
    # D = full firmware sim (int8 weights, Q15 state, piecewise tanh)
    print("\n--- Stage comparison: isolating quantization effects ---")
    print("  A = float32+tanh  B = float32+pwtanh  C = deqWeights+tanh  D = firmwareSim")
    import math

    Wxh_np = model.Wxh.weight.data.T.numpy()   # (vocab, hidden)
    Whh_np = model.Whh.weight.data.T.numpy()   # (hidden, hidden)
    Why_np = model.Why.weight.data.numpy()      # (vocab, hidden)

    S_wxh = np.abs(Wxh_np).max()
    S_whh = np.abs(Whh_np).max()
    S_why = np.abs(Why_np).max()

    Wxh_q = np.clip(np.round(Wxh_np * 127.0 / S_wxh), -127, 127).astype(np.int8)
    Whh_q = np.clip(np.round(Whh_np * 127.0 / S_whh), -127, 127).astype(np.int8)
    Why_q = np.clip(np.round(Why_np * 32767.0 / S_why), -32767, 32767).astype(np.int16)

    # Dequantized weights for stage C
    Wxh_deq = Wxh_q.astype(np.float64) * S_wxh / 127.0
    Whh_deq = Whh_q.astype(np.float64) * S_whh / 127.0
    Why_deq = Why_q.astype(np.float64) * S_why / 32767.0

    WXH_SCALE = round(S_wxh * 32767.0 / S_whh)
    q_shift_formula = max(1, round(math.log2(127.0 * 32767.0 / (S_whh * 8192.0))))
    WHY_PS = max(1, math.ceil(math.log2(HIDDEN_SIZE * 32767.0 * 32767.0 / (2**31 - 1))))

    def tanh_q15(x):
        if x >= 32767:  return 32767
        if x <= -32768: return -32768
        xs = int(x)
        if xs >= 0:
            if xs < 8192: return xs
            else:         return 8192 + (((xs - 8192) * 3) >> 2)
        else:
            if xs > -8192: return xs
            else:          return -8192 + (((xs + 8192) * 3) >> 2)

    # ── Corpus calibration (runs BEFORE stage D so both use the same Q_SHIFT) ──
    # Warmup uses formula q_shift; calibrated value drives stage D and export.py.
    print("\n--- Accumulator calibration (corpus sample) ---")
    CALIB_SEQS, WARMUP, COLLECT = 300, 8, 12
    rng_c = np.random.default_rng(42)
    starts_c = rng_c.integers(0, max(1, len(data) - WARMUP - COLLECT), size=CALIB_SEQS)

    calib_acc = []
    for start in starts_c:
        h_cal = np.zeros(HIDDEN_SIZE, dtype=np.int16)
        for step in range(WARMUP + COLLECT):
            pos = int(start) + step
            if pos >= len(data): break
            tok = data[pos]
            acc_c = np.zeros(HIDDEN_SIZE, dtype=np.int64)
            for i in range(HIDDEN_SIZE):
                acc_c[i] = int(Wxh_q[tok, i]) * WXH_SCALE
                for j in range(HIDDEN_SIZE):
                    acc_c[i] += int(Whh_q[j, i]) * int(h_cal[j])
            if step >= WARMUP:
                calib_acc.extend(int(abs(acc_c[i])) for i in range(HIDDEN_SIZE))
            for i in range(HIDDEN_SIZE):
                h_cal[i] = np.int16(tanh_q15((int(acc_c[i]) + (1 << (q_shift_formula - 1))) >> q_shift_formula))

    calib_arr = np.array(calib_acc, dtype=np.float64)
    p50 = np.percentile(calib_arr, 50)
    p90 = np.percentile(calib_arr, 90)
    p95 = np.percentile(calib_arr, 95)

    # Pick Q_SHIFT whose scale factor is closest to 1.0.
    # scale = 127 / (S_whh * 2^Q_SHIFT); want scale ≈ 1.0 so firmware
    # tanh_q15 breakpoint matches training fw_tanh breakpoint (0.25).
    _ideal = math.log2(127.0 / S_whh)
    _q_lo = max(1, int(math.floor(_ideal)))
    _q_hi = _q_lo + 1
    _s_lo = 127.0 / (S_whh * (2 ** _q_lo))
    _s_hi = 127.0 / (S_whh * (2 ** _q_hi))
    q_shift = _q_lo if abs(_s_lo - 1.0) < abs(_s_hi - 1.0) else _q_hi
    sat_thresh = 32767 * (1 << q_shift)
    pct_sat = float(np.mean(calib_arr > sat_thresh) * 100)
    scale_factor = 127.0 / (S_whh * (2 ** q_shift))
    eff_bp = 0.25 / scale_factor

    print(f"  |acc| percentiles:  p50={p50:,.0f}  p90={p90:,.0f}  p95={p95:,.0f}")
    print(f"  Q_SHIFT={q_shift}  (scale-match, ideal={math.log2(127.0/S_whh):.2f})")
    print(f"  Scale factor: {scale_factor:.3f}  Effective bp: {eff_bp:.3f}  (train: 0.250)")
    print(f"  Saturation: {pct_sat:.1f}%")

    print(f"\n  S_wxh={S_wxh:.4f}, S_whh={S_whh:.4f}, S_why={S_why:.4f}")
    print(f"  WXH_SCALE={WXH_SCALE}, Q_SHIFT={q_shift} (calibrated), WHY_PROD_SHIFT={WHY_PS}")

    def tanh_pw_float(x_arr):
        """Piecewise linear tanh in float domain matching firmware shape."""
        bp = 8192.0 / 32767.0  # ~0.25
        result = np.empty_like(x_arr)
        for i in range(len(x_arr)):
            v = float(x_arr[i])
            if v >= 1.0:    result[i] = 1.0
            elif v <= -1.0: result[i] = -1.0
            elif abs(v) < bp: result[i] = v
            elif v >= 0:    result[i] = bp + (v - bp) * 0.75
            else:           result[i] = -bp + (v + bp) * 0.75
        return result

    def fmt(c):
        if c == '\n': return '\\n'
        if c == ' ':  return 'sp'
        return c

    print(f"\n  {'Prefix':10s} | {'A':4s} | {'B':4s} | {'C':4s} | {'D':4s} | {'fMarg':6s} | {'Wxh/Whh':7s} | Break")
    print("  " + "-" * 68)

    matches_d = 0
    for prefix, expected in test_prefixes:
        top1_a = float32_top1.get(prefix, '?')
        margin_f = float32_margins.get(prefix, 0)

        # Stage B: float weights + piecewise tanh
        h_b = np.zeros(HIDDEN_SIZE, dtype=np.float64)
        for ch in prefix:
            tok = char2idx[ch]
            pre = Wxh_np[tok] + Whh_np.T @ h_b
            h_b = tanh_pw_float(pre)
        logits_b = Why_np @ h_b
        top1_b = idx2char[int(np.argmax(logits_b))]

        # Stage C: dequantized weights + real tanh
        h_c = np.zeros(HIDDEN_SIZE, dtype=np.float64)
        for ch in prefix:
            tok = char2idx[ch]
            pre = Wxh_deq[tok] + Whh_deq.T @ h_c
            h_c = np.tanh(pre)
        logits_c = Why_deq @ h_c
        top1_c = idx2char[int(np.argmax(logits_c))]

        # Stage D: full firmware sim (uses calibrated q_shift, matching export.py)
        h_d = np.zeros(HIDDEN_SIZE, dtype=np.int16)
        last_wxh_max = 0
        last_whh_max = 0
        for ch in prefix:
            tok = char2idx[ch]
            acc = np.zeros(HIDDEN_SIZE, dtype=np.int64)
            for i in range(HIDDEN_SIZE):
                wxh_val = int(Wxh_q[tok, i]) * WXH_SCALE
                whh_sum = 0
                for j in range(HIDDEN_SIZE):
                    whh_sum += int(Whh_q[j, i]) * int(h_d[j])
                acc[i] = wxh_val + whh_sum
            # Track norms for last character only
            last_wxh_max = max(abs(int(Wxh_q[tok, i]) * WXH_SCALE) for i in range(HIDDEN_SIZE))
            last_whh_max = max(abs(sum(int(Whh_q[j, i]) * int(h_d[j]) for j in range(HIDDEN_SIZE))) for i in range(HIDDEN_SIZE))
            for i in range(HIDDEN_SIZE):
                h_d[i] = np.int16(tanh_q15((int(acc[i]) + (1 << (q_shift - 1))) >> q_shift))

        logits_d = np.zeros(vocab_size, dtype=np.int64)
        for i in range(vocab_size):
            for j in range(HIDDEN_SIZE):
                logits_d[i] += (int(Why_q[i, j]) * int(h_d[j])) >> WHY_PS
        top1_d = idx2char[int(np.argmax(logits_d))]

        ratio = f"{last_wxh_max/max(last_whh_max,1):.2f}" if last_whh_max > 0 else "inf"

        # Identify where the break occurs
        if top1_a == top1_d:
            brk = ""
            matches_d += 1
        elif top1_a != top1_b:
            brk = "<- tanh"
        elif top1_a != top1_c:
            brk = "<- weights"
        else:
            brk = "<- fixedpt"

        print(f"  {prefix:10s} | {fmt(top1_a):4s} | {fmt(top1_b):4s} | {fmt(top1_c):4s} | {fmt(top1_d):4s} | {margin_f:6.2f} | {ratio:7s} | {brk}")

    print(f"\n  D matches A: {matches_d}/{len(test_prefixes)}")
    if not USE_FW_TANH:
        print("  (tip: set USE_FW_TANH=True to train with firmware tanh and close the A/D gap)")

    # ── Sample generation ──────────────────────────────────
    print("\n--- Sample generation (greedy, float32) ---")
    with torch.no_grad():
        h = torch.zeros(1, HIDDEN_SIZE)
        seed_char = 'g' if 'g' in char2idx else chars[0]
        tok = char2idx[seed_char]
        output = [seed_char]
        for _ in range(80):
            x = torch.tensor([[tok]], dtype=torch.long)
            logits, h = model(x, h)
            tok = logits[0, -1].argmax().item()
            output.append(idx2char[tok])
        print("".join(output))

if __name__ == "__main__":
    train()
