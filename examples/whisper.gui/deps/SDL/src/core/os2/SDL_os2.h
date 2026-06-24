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
#ifndef SDL_os2_h_
#define SDL_os2_h_

#include "SDL_log.h"
#include "SDL_stdinc.h"

#ifdef OS2DEBUG
#if (OS2DEBUG-0 >= 2)
# define debug_os2(s,...) SDL_LogDebug(SDL_LOG_CATEGORY_APPLICATION,    \
                                 __func__ "(): " ##s,      ##__VA_ARGS__)
#else
# define debug_os2(s,...) printf(__func__ "(): " ##s "\n", ##__VA_ARGS__)
#endif

#else /* no debug */

# define debug_os2(s,...) do {} while (0)

#endif /* OS2DEBUG */

#if defined(HAVE_ICONV) && defined(HAVE_ICONV_H)
#define OS2_SysToUTF8(S) SDL_iconv_string("UTF-8", "", (char *)(S), SDL_strlen(S)+1)
#define OS2_UTF8ToSys(S) SDL_iconv_string("", "UTF-8", (char *)(S), SDL_strlen(S)+1)
#define libiconv_clean() do {} while(0)
#else
/* StrUTF8New() - geniconv/sys2utf8.c */
#include "geniconv/geniconv.h"
#define OS2_SysToUTF8(S) StrUTF8New(1,         (S), SDL_strlen((S)) + 1)
#define OS2_UTF8ToSys(S) StrUTF8New(0, (char *)(S), SDL_strlen((S)) + 1)
#endif

/* SDL_OS2Quit() will be called from SDL_QuitSubSystem() */
void SDL_OS2Quit(void);

#endif /* SDL_os2_h_ */

/* vi: set ts=4 sw=4 expandtab: */
