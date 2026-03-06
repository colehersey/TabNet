#include "config.h"
#include "serial.h"
#include <avr/io.h>
#include <avr/interrupt.h>

/*
 * Minimal USB CDC serial for ATmega32U4.
 *
 * This uses the hardware USB peripheral directly with a stripped-down
 * CDC-ACM implementation. For a full-featured version, integrate LUFA.
 *
 * For now: uses UART1 (PD2=RX, PD3=TX) as a fallback serial interface
 * at 9600 baud so firmware can be tested before USB CDC is integrated.
 *
 * TODO: Replace with LUFA VirtualSerial CDC for true USB serial.
 */

#define BAUD 9600
/* Use U2X (double speed) for better baud accuracy across clock speeds */
#define UBRR_VAL ((F_CPU / 8 / BAUD) - 1)

#define RX_BUF_SIZE 32

static volatile char    rx_buf[RX_BUF_SIZE];
static volatile uint8_t rx_head;
static volatile uint8_t rx_tail;

void serial_init(void)
{
    /* UART1 baud rate */
    UBRR1H = (uint8_t)(UBRR_VAL >> 8);
    UBRR1L = (uint8_t)(UBRR_VAL);

    /* Enable double speed mode */
    UCSR1A = (1 << U2X1);

    /* Enable RX, TX, and RX interrupt */
    UCSR1B = (1 << RXEN1) | (1 << TXEN1) | (1 << RXCIE1);

    /* 8N1 frame format */
    UCSR1C = (1 << UCSZ11) | (1 << UCSZ10);

    rx_head = 0;
    rx_tail = 0;
}

ISR(USART1_RX_vect)
{
    char ch = UDR1;
    uint8_t next = (rx_head + 1) % RX_BUF_SIZE;
    if (next != rx_tail) {
        rx_buf[rx_head] = ch;
        rx_head = next;
    }
}

void serial_task(void)
{
    /* No-op for UART mode. Required for LUFA USB polling. */
}

uint8_t serial_available(void)
{
    return rx_head != rx_tail;
}

char serial_read(void)
{
    while (!serial_available())
        ;
    char ch = rx_buf[rx_tail];
    rx_tail = (rx_tail + 1) % RX_BUF_SIZE;
    return ch;
}

void serial_write(char ch)
{
    while (!(UCSR1A & (1 << UDRE1)))
        ;
    UDR1 = ch;
}

void serial_print(const char *str)
{
    while (*str) {
        serial_write(*str++);
    }
}
