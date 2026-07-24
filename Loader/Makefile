# Compiler & Flags
CC      := gcc
CFLAGS  := -Wall -Wextra -O3 -std=c11
INCLUDES:= -Iinc/loader

# Target Executable
TARGET  := loader

# Source and Object Files
SRCS    := src/loader/load_safetensors.c
OBJS    := $(SRCS:.c=.o)

# Default Rule
all: $(TARGET)

# Compile Executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

# Compile Object Files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean Up Generated Files
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean