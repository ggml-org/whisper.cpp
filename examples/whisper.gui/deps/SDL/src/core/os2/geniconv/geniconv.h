/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

/*
  Universal iconv implementation for OS/2.

  Andrey Vasilkin, 2016.
*/

#ifndef GENICONV_H
#define GENICONV_H

#include "iconv.h"

#ifdef iconv_open
#undef iconv_open
#endif
#define iconv_open libiconv_open

#ifdef iconv
#undef iconv
#endif
#define iconv libiconv

#ifdef iconv_close
#undef iconv_close
#endif
#define iconv_close libiconv_close

#define iconv_clean libiconv_clean

/* Non-standard function for iconv to unload the used dynamic library */
void libiconv_clean(void);

iconv_t libiconv_open (const char *tocode, const char *fromcode);
int     libiconv_close(iconv_t cd);
size_t  libiconv      (iconv_t cd, char **inbuf, size_t *inbytesleft,
                       char **outbuf, size_t *outbytesleft);

/* System codepage <-> UTF-8
 *
 * StrUTF8()
 * Converts string from system cp to UTF-8 (to_utf8 is not 0) or from UTF-8 to
 * the system cp (to_utf8 is 0). Converted ASCIIZ string will be placed at the
 * buffer dst, up to c_dst - 1 (for sys->utf8) or 2 (for utf8->sys) bytes.
 * Returns the number of bytes written into dst, not counting the terminating
 * 0 byte(s) or -1 on error.
 */
int StrUTF8(int to_utf8, char *dst, int c_dst, char *src, int c_src);

/* StrUTF8New()
 * Converts string from system cp to UTF-8 (to_utf8 is not 0) or from UTF-8
 * to the system cp (to_utf8 is 0). Memory for the new string is obtained by
 * using libc malloc().
 * Returns converted string, terminating two bytes 0 is appended to the result.
 * Returns null on error.
 */
char *StrUTF8New(int to_utf8, char *str, int c_str);

/* StrUTF8Free()
 * Deallocates the memory block allocated by StrUTF8New() (just libc free()).
 */
void StrUTF8Free(char *str);

#endif /* GENICONV_H */

/* vi: set ts=4 sw=4 expandtab: */
