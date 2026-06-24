# Makefile to build the SDL library

CPPFLAGS = -I./include
CFLAGS  = -g -O2
AR	= ar
RANLIB	= ranlib

TARGET  = libSDL2.a
TESTTARGET = libSDL2_test.a

SOURCES = \
	src/*.c \
	src/atomic/*.c \
	src/audio/*.c \
	src/audio/dummy/*.c \
	src/cpuinfo/*.c \
	src/events/*.c \
	src/file/*.c \
	src/haptic/*.c \
	src/haptic/dummy/*.c \
	src/hidapi/*.c \
	src/joystick/*.c \
	src/joystick/dummy/*.c \
	src/loadso/dummy/*.c \
	src/power/*.c \
	src/filesystem/dummy/*.c \
	src/locale/*.c \
	src/locale/dummy/*.c \
	src/misc/*.c \
	src/misc/dummy/*.c \
	src/render/*.c \
	src/render/software/*.c \
	src/sensor/*.c \
	src/sensor/dummy/*.c \
	src/stdlib/*.c \
	src/libm/*.c \
	src/thread/*.c \
	src/thread/generic/*.c \
	src/timer/*.c \
	src/timer/dummy/*.c \
	src/video/*.c \
	src/video/yuv2rgb/*.c \
	src/video/dummy/*.c \

TSOURCES = src/test/*.c

OBJECTS = $(shell echo $(SOURCES) | sed -e 's,\.c,\.o,g')
TOBJECTS= $(shell echo $(TSOURCES) | sed -e 's,\.c,\.o,g')

all: $(TARGET) $(TESTTARGET)

$(TARGET): $(OBJECTS)
	$(AR) crv $@ $^
	$(RANLIB) $@

$(TESTTARGET): $(TOBJECTS)
	$(AR) crv $@ $^
	$(RANLIB) $@

clean:
	rm -f $(TARGET) $(TESTTARGET) $(OBJECTS) $(TOBJECTS)
