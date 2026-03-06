#ifndef CONFIG_H
#define CONFIG_H

#include <avr/io.h>

/* ── Clock ─────────────────────────────────────────────────── */
/* F_CPU defined by Makefile via -DF_CPU (8 MHz)               */

/* ── Model dimensions ──────────────────────────────────────── */
#define HIDDEN_SIZE  96
#define VOCAB_SIZE   41      /* set by trainer export          */

/* ── Fixed-point config ────────────────────────────────────── */
/* Q_SHIFT is now generated in weights.h by export.py         */
#define Q15_MAX      32767
#define Q15_MIN     (-32768)

/* ── SPI GLCD (ST7565, 128x32) ─────────────────────────────── */
#define GLCD_WIDTH   128
#define GLCD_HEIGHT  32
#define GLCD_PAGES   (GLCD_HEIGHT / 8)   /* 4 pages            */

#define GLCD_DDR     DDRB
#define GLCD_PORT    PORTB
#define GLCD_CS      PB0     /* SS_n   */
#define GLCD_SCLK    PB1     /* SCLK   */
#define GLCD_MOSI    PB2     /* MOSI   */

#define GLCD_DC_DDR  DDRF
#define GLCD_DC_PORT PORTF
#define GLCD_DC_PIN  PF1     /* A0 / D/C  */

#define GLCD_RST_DDR  DDRF
#define GLCD_RST_PORT PORTF
#define GLCD_RST_PIN  PF0    /* active-low reset */

#define GLCD_BL_DDR  DDRC
#define GLCD_BL_PORT PORTC
#define GLCD_BL_PIN  PC7     /* backlight PWM    */

/* ── Hidden-state LEDs ─────────────────────────────────────── */
/*  LED 0-4 → PB3-PB7   LED 5-7 → PF4-PF6                    */
#define LED_PORTB_MASK  0xF8  /* PB3..PB7 = bits 3-7           */
#define LED_PORTF_MASK  0x70  /* PF4..PF6 = bits 4-6           */

/* ── 4-Switch Controls (active-low, pull-up) ──────────────── */
#define SW_RESET     PD7
#define SW_PREDICT   PD6
#define SW_ACCEPT    PD5
#define SW_DENY      PD4
#define SW_MASK      ((1<<SW_RESET)|(1<<SW_PREDICT)|(1<<SW_ACCEPT)|(1<<SW_DENY))
#define SW_COUNT     4
#define DEBOUNCE_MS  20

/* ── Prediction parameters ────────────────────────────────── */
#define CONTEXT_MAX      19   /* visible chars after "C:" prefix */
#define PREDICT_MAX       8
#define CONFIDENCE_SCALE 256  /* logit margin divisor            */

/* ── Display layout (page indices) ─────────────────────────── */
/* COM reverse (0xC8): page 0 = top, page 3 = bottom           */
#define DISPLAY_CONTEXT_PAGE 0   /* "C:" context line           */
#define DISPLAY_PREDICT_PAGE 2   /* "P:" prediction line        */

/* ── Font ──────────────────────────────────────────────────── */
#define FONT_WIDTH   5
#define FONT_HEIGHT  7
#define FONT_STRIDE  6       /* 5 pixels + 1 gap               */
#define CHARS_PER_LINE (GLCD_WIDTH / FONT_STRIDE)  /* 21       */

#endif /* CONFIG_H */
