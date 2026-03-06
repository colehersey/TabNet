#ifndef DISPLAY_H
#define DISPLAY_H

#include <stdint.h>

void display_init(void);
void display_clear(void);
void display_char_at(uint8_t col, uint8_t page, char ch);
void display_string_at(uint8_t col, uint8_t page, const char *str);

void display_boot_screen(void);
void display_context_line(const char *buf, uint8_t len);
void display_predict_line(const char *buf, uint8_t len);
void display_clear_predict(void);

#endif /* DISPLAY_H */
