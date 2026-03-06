#ifndef SERIAL_H
#define SERIAL_H

#include <stdint.h>

void    serial_init(void);
void    serial_task(void);
uint8_t serial_available(void);
char    serial_read(void);
void    serial_write(char ch);
void    serial_print(const char *str);

#endif /* SERIAL_H */
