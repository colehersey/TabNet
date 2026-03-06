#include "config.h"
#include "rnn.h"
#include "weights.h"
#include "vocab.h"
#include <avr/pgmspace.h>
#include <string.h>

/* ── Hidden state (persistent across timesteps) ────────────── */

static int16_t h[HIDDEN_SIZE];
static int16_t h_base[HIDDEN_SIZE];
static uint8_t confidence;
static int32_t raw_margin;

/* ── Piecewise-linear tanh approximation (Q15) ─────────────── */

static int16_t tanh_q15(int32_t x)
{
    /*
     * Input:  x in ~Q15 range (after shift)
     * Output: Q15 result in [-32767, +32767]
     *
     * Regions:
     *   |x| >= 32767  → saturate to ±32767
     *   |x| <  8192   → linear (x itself)
     *   else           → interpolated: 8192 + (x - 8192) * 3/4
     *
     * This gives a reasonable S-curve without division or LUT.
     */
    if (x >= Q15_MAX)  return Q15_MAX;
    if (x <= Q15_MIN)  return Q15_MIN;

    int16_t xs = (int16_t)x;

    if (xs >= 0) {
        if (xs < 8192)
            return xs;
        else
            return 8192 + (int16_t)(((int32_t)(xs - 8192) * 3) >> 2);
    } else {
        if (xs > -8192)
            return xs;
        else
            return -8192 + (int16_t)(((int32_t)(xs + 8192) * 3) >> 2);
    }
}

/* ── Public API ────────────────────────────────────────────── */

void rnn_init(void)
{
    rnn_reset();
}

void rnn_reset(void)
{
    memset(h, 0, sizeof(h));
    memset(h_base, 0, sizeof(h_base));
    confidence = 0;
}

void rnn_step(uint8_t token_idx)
{
    int32_t acc[HIDDEN_SIZE];

    if (token_idx >= VOCAB_SIZE)
        token_idx = 0;

    /*
     * acc[i] = Wxh[tok][i] * WXH_SCALE
     *        + sum_j( Whh[j][i] * h[j] )
     *
     * Wxh and Whh use separate int8 scales for maximum precision.
     * WXH_SCALE reconciles the scales so both contributions are
     * on the same integer scale in the accumulator.
     *
     * No per-product shift needed — int8 * int16 products fit int32
     * even when summed over HIDDEN_SIZE terms.
     * Constants are in weights.h (from export.py).
     */
    for (uint8_t i = 0; i < HIDDEN_SIZE; i++) {
        /* Input contribution: Wxh * WXH_SCALE (reconciles scales) */
        acc[i] = (int32_t)((int8_t)pgm_read_byte(&Wxh[token_idx][i]))
               * (int32_t)WXH_SCALE;

        /* Recurrent contribution (no shift needed — fits int32) */
        for (uint8_t j = 0; j < HIDDEN_SIZE; j++) {
            acc[i] += (int32_t)((int8_t)pgm_read_byte(&Whh[j][i]))
                    * (int32_t)h[j];
        }
    }

    /* Apply tanh and update hidden state */
    for (uint8_t i = 0; i < HIDDEN_SIZE; i++) {
        h[i] = tanh_q15((acc[i] + (1L << (Q_SHIFT - 1))) >> Q_SHIFT);
    }
}

uint8_t rnn_predict(void)
{
    /*
     * logits[i] = sum_j( (Why[i][j] * h[j]) >> WHY_PROD_SHIFT )
     * Why is int16 for maximum output precision.
     * Each product is >> WHY_PROD_SHIFT to prevent int32 overflow.
     * Argmax is preserved since all logits shift equally.
     */
    int32_t best = INT32_MIN;
    int32_t second = INT32_MIN;
    uint8_t best_idx = 0;

    for (uint8_t i = 0; i < VOCAB_SIZE; i++) {
        int32_t logit = 0;
        for (uint8_t j = 0; j < HIDDEN_SIZE; j++) {
            logit += ((int32_t)((int16_t)pgm_read_word(&Why[i][j]))
                    * (int32_t)h[j]) >> WHY_PROD_SHIFT;
        }
        if (logit > best) {
            second = best;
            best = logit;
            best_idx = i;
        } else if (logit > second) {
            second = logit;
        }
    }

    /* Confidence = clamp(margin / CONFIDENCE_SCALE, 0, 8) */
    int32_t margin = best - second;
    if (margin < 0) margin = 0;
    raw_margin = margin;
    margin /= CONFIDENCE_SCALE;
    if (margin > 8) margin = 8;
    confidence = (uint8_t)margin;

    return best_idx;
}

void rnn_save_base(void)
{
    memcpy(h_base, h, sizeof(h));
}

void rnn_restore_base(void)
{
    memcpy(h, h_base, sizeof(h));
}

uint8_t rnn_get_confidence(void)
{
    return confidence;
}

int32_t rnn_get_raw_margin(void)
{
    return raw_margin;
}
