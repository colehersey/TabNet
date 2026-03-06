#include "config.h"
#include "controls.h"
#include <avr/io.h>

/*
 * 4 independent switches on PD4-PD7, active-low with pull-ups.
 * Each switch has its own debounce counter.
 * Returns one SW_ID per press (falling edge), no repeat while held.
 */

static const uint8_t sw_pins[SW_COUNT] = {
    SW_PREDICT, SW_ACCEPT, SW_DENY, SW_RESET
};
static const uint8_t sw_ids[SW_COUNT] = {
    SW_ID_PREDICT, SW_ID_ACCEPT, SW_ID_DENY, SW_ID_RESET
};

static uint8_t sw_last[SW_COUNT];
static uint8_t sw_debounce[SW_COUNT];

void controls_init(void)
{
    /* PD4-PD7 as inputs with pull-ups */
    DDRD  &= ~SW_MASK;
    PORTD |=  SW_MASK;

    for (uint8_t i = 0; i < SW_COUNT; i++) {
        sw_last[i] = 1;      /* idle high (pull-up) */
        sw_debounce[i] = 0;
    }
}

uint8_t controls_poll(void)
{
    uint8_t port = PIND;

    for (uint8_t i = 0; i < SW_COUNT; i++) {
        uint8_t raw = (port & (1 << sw_pins[i])) ? 1 : 0;

        if (raw != sw_last[i]) {
            sw_debounce[i]++;
            if (sw_debounce[i] >= (DEBOUNCE_MS / 5)) {
                sw_debounce[i] = 0;
                sw_last[i] = raw;
                if (raw == 0) {
                    return sw_ids[i];  /* falling edge = pressed */
                }
            }
        } else {
            sw_debounce[i] = 0;
        }
    }

    return SW_ID_NONE;
}
