#include "config.h"
#include "glcd.h"
#include "display.h"
#include "leds.h"
#include "serial.h"
#include "controls.h"
#include "rnn.h"
#include "vocab.h"

#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <string.h>
#include <stdlib.h>

/* ── State machine ────────────────────────────────────────── */

#define STATE_BOOT   0
#define STATE_ACTIVE 1

static uint8_t state;

/* ── Context buffer (rolling display) ─────────────────────── */

#define CTX_BUF_SIZE (CONTEXT_MAX + 1)  /* +1 for safety */
static char    ctx_buf[CTX_BUF_SIZE];
static uint8_t ctx_len;

/* ── Prediction buffer ────────────────────────────────────── */

#define PRED_BUF_SIZE (PREDICT_MAX + 1)
static char    pred_buf[PRED_BUF_SIZE];
static uint8_t pred_len;

/* ── Auto-deny: clear prediction before processing input ──── */

static void auto_deny(void)
{
    if (pred_len > 0) {
        pred_len = 0;
        rnn_restore_base();
        display_clear_predict();
        leds_all_off();
    }
}

/* ── Append a character to the context buffer ─────────────── */

static void ctx_append(char ch)
{
    if (ctx_len >= CONTEXT_MAX) {
        /* Shift left to make room */
        memmove(ctx_buf, ctx_buf + 1, CONTEXT_MAX - 1);
        ctx_buf[CONTEXT_MAX - 1] = ch;
    } else {
        ctx_buf[ctx_len++] = ch;
    }
}

/* ── Handle a typed character ─────────────────────────────── */

static void handle_char(char ch)
{
    auto_deny();

    /* Feed the character to the RNN */
    uint8_t tok = vocab_encode(ch);
    rnn_step(tok);
    rnn_save_base();

    /* Update context display */
    ctx_append(ch);
    display_context_line(ctx_buf, ctx_len);

    /* Echo to serial */
    serial_write(ch);
}

/* ── Button: PREDICT ──────────────────────────────────────── */

static void do_predict(void)
{
    if (pred_len >= PREDICT_MAX)
        return;

    uint8_t tok = rnn_predict();
    uint8_t conf = rnn_get_confidence();
    int32_t margin = rnn_get_raw_margin();

    /* Advance working hidden state (not base) */
    rnn_step(tok);

    char ch = vocab_decode(tok);
    pred_buf[pred_len++] = ch;

    display_predict_line(pred_buf, pred_len);
    leds_set_bar(conf);

    /* Debug: print token, confidence, raw margin */
    char num[12];
    serial_print("[P tok=");
    ltoa(tok, num, 10);
    serial_print(num);
    serial_print(" ch='");
    serial_write(ch);
    serial_print("' conf=");
    ltoa(conf, num, 10);
    serial_print(num);
    serial_print(" margin=");
    ltoa(margin, num, 10);
    serial_print(num);
    serial_print("]\r\n");
}

/* ── Button: ACCEPT ───────────────────────────────────────── */

static void do_accept(void)
{
    if (pred_len == 0)
        return;

    /* Move prediction into context */
    for (uint8_t i = 0; i < pred_len; i++) {
        ctx_append(pred_buf[i]);
        serial_write(pred_buf[i]);
    }

    rnn_save_base();
    display_context_line(ctx_buf, ctx_len);

    pred_len = 0;
    display_clear_predict();
    leds_all_off();
}

/* ── Button: DENY ─────────────────────────────────────────── */

static void do_deny(void)
{
    if (pred_len == 0)
        return;

    pred_len = 0;
    rnn_restore_base();
    display_clear_predict();
    leds_all_off();
}

/* ── Button: RESET ────────────────────────────────────────── */

static void do_reset(void)
{
    ctx_len = 0;
    pred_len = 0;
    rnn_reset();
    leds_all_off();
    display_clear();
    display_context_line(ctx_buf, 0);
}

/* ── Main ──────────────────────────────────────────────────── */

int main(void)
{
    /* Init all subsystems */
    display_init();
    leds_init();
    serial_init();
    controls_init();
    rnn_init();

    sei();

    /* Boot state */
    state = STATE_BOOT;
    ctx_len = 0;
    pred_len = 0;
    leds_all_off();

    display_boot_screen();
    serial_print("Neural Seq Gen ready.\r\n");

    for (;;) {
        /* ── Serial input ────────────────────────────────── */
        while (serial_available()) {
            char ch = serial_read();

            if (state == STATE_BOOT) {
                /* First char transitions to active mode */
                state = STATE_ACTIVE;
                display_clear();
            }

            if (ch == '\r' || ch == '\n') {
                /* Ignore CR/LF */
            } else if (ch >= ' ' && ch <= '~') {
                handle_char(ch);
            }
        }

        /* ── Button input (active mode only) ─────────────── */
        if (state == STATE_ACTIVE) {
            uint8_t sw = controls_poll();

            switch (sw) {
            case SW_ID_PREDICT: do_predict(); break;
            case SW_ID_ACCEPT:  do_accept();  break;
            case SW_ID_DENY:    do_deny();    break;
            case SW_ID_RESET:   do_reset();   break;
            default: break;
            }
        }

        serial_task();
        _delay_ms(5);
    }

    return 0;
}
