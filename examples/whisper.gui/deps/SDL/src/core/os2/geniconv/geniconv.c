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

#define INCL_DOSMODULEMGR   /* Module Manager */
#define INCL_DOSERRORS      /* Error values   */
#include <os2.h>

#include "geniconv.h"

/*#define DEBUG*/
#ifdef DEBUG
# include <stdio.h>
# define iconv_debug(s,...) printf(__func__"(): "##s"\n" ,##__VA_ARGS__)
#else
# define iconv_debug(s,...) do {} while (0)
#endif

/* Exports from os2iconv.c */
extern iconv_t _System os2_iconv_open  (const char* tocode, const char* fromcode);
extern size_t  _System os2_iconv       (iconv_t cd,
                                        char **inbuf,  size_t *inbytesleft,
                                        char **outbuf, size_t *outbytesleft);
extern int     _System os2_iconv_close (iconv_t cd);

/* Functions pointers */
typedef iconv_t (_System *FNICONV_OPEN)(const char*, const char*);
typedef size_t  (_System *FNICONV)     (iconv_t, char **, size_t *, char **, size_t *);
typedef int     (_System *FNICONV_CLOSE)(iconv_t);

static HMODULE         hmIconv = NULLHANDLE;
static FNICONV_OPEN    fn_iconv_open  = os2_iconv_open;
static FNICONV         fn_iconv       = os2_iconv;
static FNICONV_CLOSE   fn_iconv_close = os2_iconv_close;

static int geniconv_init = 0;


static BOOL _loadDLL(const char *dllname,
                     const char *sym_iconvopen,
                     const char *sym_iconv,
                     const char *sym_iconvclose)
{
    ULONG rc;
    char  error[256];

    rc = DosLoadModule(error, sizeof(error), dllname, &hmIconv);
    if (rc != NO_ERROR) {
        iconv_debug("DLL %s not loaded: %s", dllname, error);
        return FALSE;
    }

    rc = DosQueryProcAddr(hmIconv, 0, sym_iconvopen, (PFN *)&fn_iconv_open);
    if (rc != NO_ERROR) {
        iconv_debug("Error: cannot find entry %s in %s", sym_iconvopen, dllname);
        goto fail;
    }

    rc = DosQueryProcAddr(hmIconv, 0, sym_iconv, (PFN *)&fn_iconv);
    if (rc != NO_ERROR) {
        iconv_debug("Error: cannot find entry %s in %s", sym_iconv, dllname);
        goto fail;
    }

    rc = DosQueryProcAddr(hmIconv, 0, sym_iconvclose, (PFN *)&fn_iconv_close);
    if (rc != NO_ERROR) {
        iconv_debug("Error: cannot find entry %s in %s", sym_iconvclose, dllname);
        goto fail;
    }

    iconv_debug("DLL %s used", dllname);
    return TRUE;

    fail:
    DosFreeModule(hmIconv);
    hmIconv = NULLHANDLE;
    return FALSE;
}

static void _init(void)
{
    if (geniconv_init) {
        return; /* Already initialized */
    }

    geniconv_init = 1;

    /* Try to load kiconv.dll, iconv2.dll or iconv.dll */
    if (!_loadDLL("KICONV", "_libiconv_open", "_libiconv", "_libiconv_close") &&
        !_loadDLL("ICONV2", "_libiconv_open", "_libiconv", "_libiconv_close") &&
        !_loadDLL("ICONV",  "_iconv_open",    "_iconv",    "_iconv_close") ) {
        /* No DLL was loaded - use OS/2 conversion objects API */
        iconv_debug("Uni*() API used");
        fn_iconv_open  = os2_iconv_open;
        fn_iconv       = os2_iconv;
        fn_iconv_close = os2_iconv_close;
    }
}


/* Public routines.
 * ----------------
 */

/* function to unload the used iconv dynamic library */
void libiconv_clean(void)
{
    geniconv_init = 0;

    /* reset the function pointers. */
    fn_iconv_open  = os2_iconv_open;
    fn_iconv       = os2_iconv;
    fn_iconv_close = os2_iconv_close;

    if (hmIconv != NULLHANDLE) {
        DosFreeModule(hmIconv);
        hmIconv = NULLHANDLE;
    }
}

iconv_t libiconv_open(const char* tocode, const char* fromcode)
{
    _init();
    return fn_iconv_open(tocode, fromcode);
}

size_t libiconv(iconv_t cd, char* * inbuf, size_t *inbytesleft,
                char* * outbuf, size_t *outbytesleft)
{
    return fn_iconv(cd, inbuf, inbytesleft, outbuf, outbytesleft);
}

int libiconv_close(iconv_t cd)
{
    return fn_iconv_close(cd);
}

/* vi: set ts=4 sw=4 expandtab: */
