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
   Implementation iconv via OS/2 conversion objects API.

   Andrey Vasilkin.
*/

#define ICONV_THREAD_SAFE 1

#include "geniconv.h"
#define _ULS_CALLCONV_
#define CALLCONV _System
#include <uconv.h>
#ifdef ICONV_THREAD_SAFE
#define INCL_DOSSEMAPHORES
#define INCL_DOSERRORS
#include <os2.h>
#endif

#include "os2cp.h"

#ifndef GENICONV_STANDALONE
#include "../../../SDL_internal.h"
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(min)
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif
#define SDL_min   min
#define SDL_strcasecmp stricmp
#define SDL_snprintf _snprintf
#define SDL_malloc malloc
#define SDL_free free
#define SDL_memcpy memcpy
#endif

#define MAX_CP_NAME_LEN 64

typedef struct iuconv_obj {
  UconvObject	uo_tocode;
  UconvObject	uo_fromcode;
  int		buf_len;
  UniChar	*buf;
#ifdef ICONV_THREAD_SAFE
  HMTX		hMtx;
#endif
} iuconv_obj;


static int _createUconvObj(const char *code, UconvObject *uobj)
{
    UniChar uc_code[MAX_CP_NAME_LEN];
    int i;
    const unsigned char *ch =
         (const unsigned char *)code;

    if (!code)
        uc_code[0] = 0;
    else {
        for (i = 0; i < MAX_CP_NAME_LEN; i++) {
            uc_code[i] = (unsigned short)*ch;
            if (! (*ch))
                break;
            ch++;
        }
    }

    return UniCreateUconvObject(uc_code, uobj);
}

static int uconv_open(const char *code, UconvObject *uobj)
{
    int rc;

    if (!SDL_strcasecmp(code, "UTF-16")) {
        *uobj = NULL;
        return ULS_SUCCESS;
    }

    rc = _createUconvObj(code, uobj);
    if (rc != ULS_SUCCESS) {
        unsigned long cp = os2cpFromName((char *)code);
        char cp_name[16];
        if (cp != 0 && SDL_snprintf(cp_name, sizeof(cp_name), "IBM-%u", cp) > 0) {
            rc = _createUconvObj(cp_name, uobj);
        }
    }

    return rc;
}


iconv_t _System os2_iconv_open(const char* tocode, const char* fromcode)
{
    UconvObject uo_tocode;
    UconvObject uo_fromcode;
    int rc;
    iuconv_obj *iuobj;

    if (!tocode) {
        tocode = "";
    }
    if (!fromcode) {
        fromcode = "";
    }

    if (SDL_strcasecmp(tocode, fromcode) != 0) {
        rc = uconv_open(fromcode, &uo_fromcode);
        if (rc != ULS_SUCCESS) {
            errno = EINVAL;
            return (iconv_t)(-1);
        }
        rc = uconv_open(tocode, &uo_tocode);
        if (rc != ULS_SUCCESS) {
            UniFreeUconvObject(uo_fromcode);
            errno = EINVAL;
            return (iconv_t)(-1);
        }
    } else {
        uo_tocode = NULL;
        uo_fromcode = NULL;
    }

    iuobj = (iuconv_obj *) SDL_malloc(sizeof(iuconv_obj));
    iuobj->uo_tocode = uo_tocode;
    iuobj->uo_fromcode = uo_fromcode;
    iuobj->buf_len = 0;
    iuobj->buf = NULL;
#ifdef ICONV_THREAD_SAFE
    DosCreateMutexSem(NULL, &iuobj->hMtx, 0, FALSE);
#endif

    return iuobj;
}

size_t _System os2_iconv(iconv_t cd,
                         char **inbuf,  size_t *inbytesleft ,
                         char **outbuf, size_t *outbytesleft)
{
    UconvObject uo_tocode = ((iuconv_obj *)(cd))->uo_tocode;
    UconvObject uo_fromcode = ((iuconv_obj *)(cd))->uo_fromcode;
    size_t nonIdenticalConv = 0;
    UniChar *uc_buf;
    size_t uc_buf_len;
    UniChar **uc_str;
    size_t *uc_str_len;
    int rc;
    size_t ret = (size_t)(-1);

    if (!uo_tocode && !uo_fromcode) {
        uc_buf_len = SDL_min(*inbytesleft, *outbytesleft);
        SDL_memcpy(*outbuf, *inbuf, uc_buf_len);
        *inbytesleft -= uc_buf_len;
        *outbytesleft -= uc_buf_len;
        outbuf += uc_buf_len;
        inbuf += uc_buf_len;
        return uc_buf_len;
    }

#ifdef ICONV_THREAD_SAFE
    DosRequestMutexSem(((iuconv_obj *)(cd))->hMtx, SEM_INDEFINITE_WAIT);
#endif

    if (uo_tocode && uo_fromcode && (((iuconv_obj *)cd)->buf_len >> 1) < *inbytesleft) {
        if (((iuconv_obj *)cd)->buf != NULL) {
            SDL_free(((iuconv_obj *)cd)->buf);
        }
        ((iuconv_obj *)cd)->buf_len = *inbytesleft << 1;
        ((iuconv_obj *)cd)->buf = (UniChar *) SDL_malloc(((iuconv_obj *)cd)->buf_len);
    }

    if (uo_fromcode) {
        if (uo_tocode) {
            uc_buf = ((iuconv_obj *)cd)->buf;
            uc_buf_len = ((iuconv_obj *)cd)->buf_len;
            uc_str = &uc_buf;
        } else {
            uc_str = (UniChar **)outbuf;
            uc_buf_len = *outbytesleft;
        }
        uc_buf_len = uc_buf_len >> 1;
        uc_str_len = &uc_buf_len;
        rc = UniUconvToUcs(uo_fromcode, (void **)inbuf, inbytesleft,
                           uc_str, uc_str_len, &nonIdenticalConv);
        uc_buf_len = uc_buf_len << 1;
        if (!uo_tocode) {
            *outbytesleft = uc_buf_len;
        }

        if (rc != ULS_SUCCESS) {
            errno = EILSEQ;
            goto done;
        } else if (*inbytesleft && !*uc_str_len) {
            errno = E2BIG;
            goto done;
        }

        if (!uo_tocode) {
            return nonIdenticalConv;
        }

        uc_buf = ((iuconv_obj *)cd)->buf;
        uc_buf_len = ((iuconv_obj *)cd)->buf_len - uc_buf_len;
        uc_str = &uc_buf;
        uc_str_len = &uc_buf_len;
    } else {
        uc_str = (UniChar **)inbuf;
        uc_str_len = inbytesleft;
    }

    *uc_str_len = *uc_str_len>>1;
    rc = UniUconvFromUcs(uo_tocode, uc_str, uc_str_len, (void **)outbuf,
                         outbytesleft, &nonIdenticalConv);
    if (rc != ULS_SUCCESS) {
        switch (rc) {
        case ULS_BUFFERFULL:
            errno = E2BIG;
            break;
        case ULS_ILLEGALSEQUENCE:
            errno = EILSEQ;
            break;
        case ULS_INVALID:
            errno = EINVAL;
            break;
        }
        goto done;
    } else if (*uc_str_len && !*outbytesleft) {
        errno = E2BIG;
        goto done;
    }

    ret = nonIdenticalConv;

done:

#ifdef ICONV_THREAD_SAFE
    DosReleaseMutexSem(((iuconv_obj *)cd)->hMtx);
#endif
    return ret;
}

int _System os2_iconv_close(iconv_t cd)
{
    if (!cd) return 0;

#ifdef ICONV_THREAD_SAFE
    DosCloseMutexSem(((iuconv_obj *)cd)->hMtx);
#endif
    if (((iuconv_obj *)cd)->uo_tocode != NULL) {
        UniFreeUconvObject(((iuconv_obj *)cd)->uo_tocode);
    }
    if (((iuconv_obj *)cd)->uo_fromcode != NULL) {
        UniFreeUconvObject(((iuconv_obj *)cd)->uo_fromcode);
    }

    if (((iuconv_obj *)cd)->buf != NULL) {
        SDL_free(((iuconv_obj *)cd)->buf);
    }
    SDL_free(cd);

    return 0;
}

/* vi: set ts=4 sw=4 expandtab: */
