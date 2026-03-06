#include "config.h"
#include "display.h"
#include "glcd.h"
#include "font5x7.h"
#include <avr/pgmspace.h>
#include <string.h>

/* ── Font rendering ────────────────────────────────────────── */

static void render_glyph(uint8_t col, uint8_t page, char ch)
{
    if (ch < FONT_FIRST_CHAR || ch > FONT_LAST_CHAR)
        ch = ' ';

    uint16_t offset = (uint16_t)(ch - FONT_FIRST_CHAR) * FONT_WIDTH;

    glcd_set_page(page);
    glcd_set_column(col);

    for (uint8_t i = 0; i < FONT_WIDTH; i++) {
        glcd_write_data(pgm_read_byte(&font5x7[offset + i]));
    }
    glcd_write_data(0x00);  /* 1-pixel gap between chars */
}

static void clear_page(uint8_t page)
{
    glcd_set_page(page);
    glcd_set_column(0);
    for (uint8_t i = 0; i < GLCD_WIDTH; i++)
        glcd_write_data(0x00);
}

/* ── Public API ────────────────────────────────────────────── */

void display_init(void)
{
    glcd_init();
    display_clear();
}

void display_clear(void)
{
    glcd_clear();
}

void display_char_at(uint8_t col, uint8_t page, char ch)
{
    glcd_begin();
    render_glyph(col * FONT_STRIDE, page, ch);
    glcd_end();
}

void display_string_at(uint8_t col, uint8_t page, const char *str)
{
    glcd_begin();
    for (uint8_t i = 0; str[i] && (col + i) < CHARS_PER_LINE; i++) {
        render_glyph((col + i) * FONT_STRIDE, page, str[i]);
    }
    glcd_end();
}

void display_boot_screen(void)
{
    /* Center "ready" on page 1 (middle of 4-page display) */
    glcd_begin();
    clear_page(1);
    const char *msg = "press any key";
    uint8_t len = 13;
    uint8_t start_col = (CHARS_PER_LINE - len) / 2;
    for (uint8_t i = 0; i < len; i++) {
        render_glyph((start_col + i) * FONT_STRIDE, 1, msg[i]);
    }
    glcd_end();
}

void display_context_line(const char *buf, uint8_t len)
{
    glcd_begin();
    clear_page(DISPLAY_CONTEXT_PAGE);

    /* "C:" prefix */
    render_glyph(0 * FONT_STRIDE, DISPLAY_CONTEXT_PAGE, 'C');
    render_glyph(1 * FONT_STRIDE, DISPLAY_CONTEXT_PAGE, ':');

    /* Show last CONTEXT_MAX chars (or fewer) */
    uint8_t start = 0;
    if (len > CONTEXT_MAX)
        start = len - CONTEXT_MAX;

    for (uint8_t i = start; i < len; i++) {
        render_glyph((2 + i - start) * FONT_STRIDE, DISPLAY_CONTEXT_PAGE, buf[i]);
    }
    glcd_end();
}

void display_predict_line(const char *buf, uint8_t len)
{
    glcd_begin();
    clear_page(DISPLAY_PREDICT_PAGE);

    /* "P:" prefix */
    render_glyph(0 * FONT_STRIDE, DISPLAY_PREDICT_PAGE, 'P');
    render_glyph(1 * FONT_STRIDE, DISPLAY_PREDICT_PAGE, ':');

    for (uint8_t i = 0; i < len && i < PREDICT_MAX; i++) {
        render_glyph((2 + i) * FONT_STRIDE, DISPLAY_PREDICT_PAGE, buf[i]);
    }
    glcd_end();
}

void display_clear_predict(void)
{
    glcd_begin();
    clear_page(DISPLAY_PREDICT_PAGE);
    glcd_end();
}
