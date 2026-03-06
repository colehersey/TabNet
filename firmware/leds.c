#include "config.h"
#include "leds.h"
#include <avr/io.h>
#include <util/delay.h>

/*
 * Confidence bar: 8 LEDs filled cumulatively.
 *
 * Fill order (position 0..7):
 *   PB3, PB4, PB5, PB6, PB7, PF4, PF5, PF6
 *
 * leds_set_bar(level) lights LEDs 0..level-1.
 */

/* Pin mapping: positions 0-4 are PB3-PB7, positions 5-7 are PF4-PF6 */
static const uint8_t portb_bit[5] = { PB3, PB4, PB5, PB6, PB7 };
static const uint8_t portf_bit[3] = { PF4, PF5, PF6 };

void leds_init(void)
{
    DDRB |= LED_PORTB_MASK;
    PORTB &= ~LED_PORTB_MASK;

    DDRF |= LED_PORTF_MASK;
    PORTF &= ~LED_PORTF_MASK;
}

void leds_set_bar(uint8_t level)
{
    if (level > 8) level = 8;

    uint8_t pb = PORTB & ~LED_PORTB_MASK;
    uint8_t pf = PORTF & ~LED_PORTF_MASK;

    /* Positions 0-4 → PORTB */
    for (uint8_t i = 0; i < 5 && i < level; i++) {
        pb |= (1 << portb_bit[i]);
    }

    /* Positions 5-7 → PORTF */
    for (uint8_t i = 0; i < 3 && (i + 5) < level; i++) {
        pf |= (1 << portf_bit[i]);
    }

    PORTB = pb;
    PORTF = pf;
}

void leds_all_off(void)
{
    PORTB &= ~LED_PORTB_MASK;
    PORTF &= ~LED_PORTF_MASK;
}

void leds_test(void)
{
    for (uint8_t i = 0; i < 5; i++) {
        PORTB |= (1 << (PB3 + i));
        _delay_ms(100);
        PORTB &= ~(1 << (PB3 + i));
    }
    for (uint8_t i = 0; i < 3; i++) {
        PORTF |= (1 << (PF4 + i));
        _delay_ms(100);
        PORTF &= ~(1 << (PF4 + i));
    }
}
