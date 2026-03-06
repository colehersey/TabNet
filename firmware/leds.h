#ifndef LEDS_H
#define LEDS_H

#include <stdint.h>

void leds_init(void);
void leds_set_bar(uint8_t level);
void leds_all_off(void);
void leds_test(void);

#endif /* LEDS_H */
