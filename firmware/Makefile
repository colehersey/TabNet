# Neural Sequence Generator — ATmega32U4 Firmware
# Toolchain: avr-gcc + avrdude

MCU        = atmega32u4
F_CPU      = 8000000UL
PROGRAMMER = avr109
PORT       = /dev/ttyS4

# LUFA paths (adjust to your LUFA installation)
LUFA_PATH  = ./lufa/LUFA

CC         = avr-gcc
OBJCOPY    = avr-objcopy
SIZE       = avr-size
AVRDUDE    = avrdude

TARGET     = neuralseq

# Source files
SRC = main.c glcd.c display.c leds.c controls.c rnn.c serial.c \
      $(wildcard lufa_usb/*.c)

# LUFA includes (when LUFA is set up)
LUFA_INC   = -I$(LUFA_PATH)/..

CFLAGS  = -mmcu=$(MCU) -DF_CPU=$(F_CPU) -Os -std=c11 \
          -Wall -Werror -Wextra -Wno-unused-parameter \
          -funsigned-char -funsigned-bitfields -fpack-struct -fshort-enums \
          -ffunction-sections -fdata-sections \
          -I. $(LUFA_INC)

LDFLAGS = -mmcu=$(MCU) -Wl,--gc-sections

OBJ = $(SRC:.c=.o)

# ── Build targets ──────────────────────────────────────────────

all: $(TARGET).hex size

$(TARGET).elf: $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

$(TARGET).hex: $(TARGET).elf
	$(OBJCOPY) -O ihex -R .eeprom $< $@

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

size: $(TARGET).elf
	$(SIZE) --format=avr --mcu=$(MCU) $<

# ── Flash ──────────────────────────────────────────────────────

flash: $(TARGET).hex
	$(AVRDUDE) -c $(PROGRAMMER) -p $(MCU) -P $(PORT) -U flash:w:$<:i

# ── Clean ──────────────────────────────────────────────────────

clean:
	rm -f $(OBJ) $(TARGET).elf $(TARGET).hex

.PHONY: all flash clean size
