#ifndef GLCD_H
#define GLCD_H

#include <stdint.h>

void    glcd_init(void);
void    glcd_clear(void);
void    glcd_set_page(uint8_t page);
void    glcd_set_column(uint8_t col);
void    glcd_write_data(uint8_t data);
void    glcd_write_data_buf(const uint8_t *buf, uint8_t len);
void    glcd_backlight_on(void);
void    glcd_backlight_off(void);
void    glcd_begin(void);
void    glcd_end(void);

#endif /* GLCD_H */
