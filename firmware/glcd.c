#include "config.h"
#include "glcd.h"
#include <avr/io.h>
#include <util/delay.h>

/* ── Low-level SPI ─────────────────────────────────────────── */

static void spi_init(void)
{
    /* PB0=SS, PB1=SCLK, PB2=MOSI as outputs */
    GLCD_DDR |= (1 << GLCD_CS) | (1 << GLCD_SCLK) | (1 << GLCD_MOSI);

    /* Ensure SS deactivated (active low) */
    GLCD_PORT |= (1 << GLCD_CS);

    /* D/C, RESET, backlight as outputs */
    GLCD_DC_DDR  |= (1 << GLCD_DC_PIN);
    GLCD_RST_DDR |= (1 << GLCD_RST_PIN);
    GLCD_BL_DDR  |= (1 << GLCD_BL_PIN);

    /* SPI master, mode 3, MSB first, fck/2 — matched to LCDDriver.asm */
    SPCR = (1 << SPE) | (1 << MSTR) | (1 << CPOL) | (1 << CPHA);
    SPSR = (1 << SPI2X);
}

/* ── Command / data helpers ────────────────────────────────── */

static inline void cs_low(void)  { GLCD_PORT &= ~(1 << GLCD_CS); }
static inline void cs_high(void) { GLCD_PORT |=  (1 << GLCD_CS); }

static void glcd_cmd(uint8_t cmd)
{
    /* Match LCDDriver.asm: start TX then set D/C during transfer */
    SPDR = cmd;
    GLCD_DC_PORT &= ~(1 << GLCD_DC_PIN);  /* D/C low = command */
    while (!(SPSR & (1 << SPIF)))
        ;
}

static void glcd_data(uint8_t data)
{
    SPDR = data;
    GLCD_DC_PORT |= (1 << GLCD_DC_PIN);   /* D/C high = data */
    while (!(SPSR & (1 << SPIF)))
        ;
}

/* ── Public API ────────────────────────────────────────────── */

void glcd_init(void)
{
    spi_init();

    /* Hardware reset */
    GLCD_RST_PORT &= ~(1 << GLCD_RST_PIN);
    _delay_ms(1);
    GLCD_RST_PORT |= (1 << GLCD_RST_PIN);

    /*
     * ST7565 init — based on LCDDriver.asm with COM reverse.
     *
     * LCDDriver.asm uses COM normal (0xC0) where MSB = top of page.
     * Our font5x7 has LSB = top row, so we use COM reverse (0xC8)
     * which flips the bit order: LSB = top of page.
     * No ADC reverse — columns go left to right normally.
     * Page 0 = top of display, page 3 = bottom.
     */
    cs_low();
    glcd_cmd(0xA2);        /* LCD bias: 1/6               */
    glcd_cmd(0xC8);        /* COM reverse: LSB=top, page 0=top */
    glcd_cmd(0x81);        /* Electronic volume (contrast) */
    glcd_cmd(0x0F);        /* contrast value: 15           */
    glcd_cmd(0x22);        /* Resistor ratio               */
    glcd_cmd(0x2F);        /* Voltage booster + reg + follower ON */
    glcd_cmd(0xAF);        /* Display ON                   */
    cs_high();

    glcd_clear();
    glcd_backlight_on();
}

void glcd_set_page(uint8_t page)
{
    glcd_cmd(0xB0 | (page & 0x0F));
}

void glcd_set_column(uint8_t col)
{
    glcd_cmd(0x10 | ((col >> 4) & 0x0F));  /* upper nibble */
    glcd_cmd(0x00 | (col & 0x0F));         /* lower nibble */
}

void glcd_write_data(uint8_t data)
{
    glcd_data(data);
}

void glcd_write_data_buf(const uint8_t *buf, uint8_t len)
{
    for (uint8_t i = 0; i < len; i++) {
        glcd_data(buf[i]);
    }
}

void glcd_clear(void)
{
    cs_low();
    for (uint8_t page = 0; page < GLCD_PAGES; page++) {
        glcd_set_page(page);
        glcd_set_column(0);
        for (uint8_t col = 0; col < GLCD_WIDTH; col++) {
            glcd_data(0x00);
        }
    }
    cs_high();
}

void glcd_backlight_on(void)
{
    GLCD_BL_PORT |= (1 << GLCD_BL_PIN);
}

void glcd_backlight_off(void)
{
    GLCD_BL_PORT &= ~(1 << GLCD_BL_PIN);
}

void glcd_begin(void)
{
    cs_low();
}

void glcd_end(void)
{
    cs_high();
}
