# Makefile for CLAHE test application

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

# Source files
SRCS = main.c test_clahe.c clahe.c
OBJS = $(SRCS:.c=.o)

# Target executable
TARGET = clahe_test

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET) clahe_test_result.jpg rgb2lab.o
	@echo "Clean complete"

# Run the application
run: $(TARGET)
	./$(TARGET)

# Build and run
test: all run

# Phony targets
.PHONY: all clean run test

# Dependencies
main.o: main.c stb_image.h stb_image_write.h
test_clahe.o: test_clahe.c clahe.h
clahe.o: clahe.c clahe.h
