#ifndef CONTROLS_H
#define CONTROLS_H

#include <stdint.h>

/* Switch ID return values */
#define SW_ID_NONE    0
#define SW_ID_PREDICT 1
#define SW_ID_ACCEPT  2
#define SW_ID_DENY    3
#define SW_ID_RESET   4

void    controls_init(void);
uint8_t controls_poll(void);

#endif /* CONTROLS_H */
