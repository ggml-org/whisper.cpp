#
# Universal iconv implementation for OS/2.
#
# OpenWatcom makefile to build a library that uses kiconv.dll / iconv2.dll /
# iconv.dll or OS/2 Uni*() API.
#
# Andrey Vasilkin, 2016.
#

LIBFILE = geniconv.lib

all: $(LIBFILE) test.exe .symbolic

CFLAGS = -I$(%WATCOM)/h/os2 -I$(%WATCOM)/h -I. -bt=os2 -q -d0 -w2 -DGENICONV_STANDALONE=1

SRCS = geniconv.c os2cp.c os2iconv.c
SRCS+= sys2utf8.c

OBJS = $(SRCS:.c=.obj)

LIBS = libuls.lib libconv.lib $(LIBFILE)

test.exe: $(LIBFILE) test.obj
  wlink op quiet system os2v2 file test.obj lib {$(LIBS)} name $*

$(LIBFILE): $(OBJS)
  @if exist $@ rm $@
  @for %f in ($(OBJS)) do wlib -q -b $* +%f

.c.obj:
    wcc386 $(CFLAGS) -fo=$^@ $<

clean: .SYMBOLIC
  @if exist *.obj rm *.obj
  @if exist *.err rm *.err
  @if exist $(LIBFILE) rm $(LIBFILE)
  @if exist test.exe rm test.exe
