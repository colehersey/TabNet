#ifndef RNN_H
#define RNN_H

#include <stdint.h>

void    rnn_init(void);
void    rnn_reset(void);
void    rnn_step(uint8_t token_idx);
uint8_t rnn_predict(void);

void    rnn_save_base(void);
void    rnn_restore_base(void);
uint8_t rnn_get_confidence(void);
int32_t rnn_get_raw_margin(void);

#endif /* RNN_H */
