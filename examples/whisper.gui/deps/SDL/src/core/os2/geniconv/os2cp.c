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

#define INCL_DOSNLS
#define INCL_DOSERRORS
#include <os2.h>

#include "os2cp.h"

#ifndef GENICONV_STANDALONE
#include "../../../SDL_internal.h"
#else
#include <string.h>
#include <ctype.h>
#define SDL_isspace isspace
#define SDL_strchr strchr
#define SDL_memcpy memcpy
#define SDL_strupr strupr
#define SDL_strcmp strcmp
#endif

typedef struct _CP2NAME {
  ULONG ulCode;
  PSZ  pszName;
} CP2NAME;

typedef struct _NAME2CP {
  PSZ  pszName;
  ULONG ulCode;
} NAME2CP;

static CP2NAME aCP2Name[] = {
  {367, "ANSI_X3.4-1968"},
  {813, "ECMA-118"},
  {819, "CP819"},
  {850, "850"},
  {862, "862"},
  {866, "866"},
  {874, "ISO-IR-166"},
  {878, "KOI8-R"},
  {896, "JISX0201-1976"},
  {901, "ISO-8859-13"},
  {912, "ISO-8859-2"},
  {913, "ISO-8859-3"},
  {914, "ISO-8859-4"},
  {915, "CYRILLIC"},
  {920, "ISO-8859-9"},
  {923, "ISO-8859-15"},
  {943, "MS_KANJI"},
  {954, "EUC-JP"},
  {964, "EUC-TW"},
  {970, "EUC-KR"},
  {1051, "HP-ROMAN8"},
  {1089, "ARABIC"},
  {1129, "VISCII"},
  {1168, "KOI8-U"},
  {1200, "ISO-10646-UCS-2"},
  {1202, "UTF-16LE"},
  {1204, "UCS-2BE"},
  {1208, "UTF-8"},
  {1232, "UTF-32BE"},
  {1234, "UTF-32LE"},
  {1236, "ISO-10646-UCS-4"},
  {1250, "CP1250"},
  {1251, "CP1251"},
  {1252, "CP1252"},
  {1253, "CP1253"},
  {1254, "CP1254"},
  {1255, "CP1255"},
  {1256, "CP1256"},
  {1257, "CP1257"},
  {1275, "MAC"},
  {1383, "CN-GB"},
  {1386, "GBK"},
  {1392, "GB18030"},
  {62210, "HEBREW"}
};

static NAME2CP aName2CP[] = {
  {"850", 850},
  {"862", 862},
  {"866", 866},
  {"ANSI_X3.4-1968", 367},
  {"ANSI_X3.4-1986", 367},
  {"ARABIC", 1089},
  {"ASCII", 367},
  {"ASMO-708", 1089},
  {"CN-GB", 1383},
  {"CP1250", 1250},
  {"CP1251", 1251},
  {"CP1252", 1252},
  {"CP1253", 1253},
  {"CP1254", 1254},
  {"CP1255", 1255},
  {"CP1256", 1256},
  {"CP1257", 1257},
  {"CP367", 367},
  {"CP819", 819},
  {"CP850", 850},
  {"CP862", 862},
  {"CP866", 866},
  {"CP936", 1386},
  {"CSASCII", 367},
  {"CSEUCKR", 970},
  {"CSEUCPKDFMTJAPANESE", 954},
  {"CSEUCTW", 964},
  {"CSGB2312", 1383},
  {"CSHALFWIDTHKATAKANA", 896},
  {"CSHPROMAN8", 1051},
  {"CSIBM866", 866},
  {"CSISOLATIN1", 819},
  {"CSISOLATIN2", 912},
  {"CSISOLATIN3", 913},
  {"CSISOLATIN4", 914},
  {"CSISOLATIN5", 920},
  {"CSISOLATINARABIC", 1089},
  {"CSISOLATINCYRILLIC", 915},
  {"CSISOLATINGREEK", 813},
  {"CSISOLATINHEBREW", 62210},
  {"CSKOI8R", 878},
  {"CSKSC56011987", 970},
  {"CSMACINTOSH", 1275},
  {"CSPC850MULTILINGUAL", 850},
  {"CSPC862LATINHEBREW", 862},
  {"CSSHIFTJIS", 943},
  {"CSUCS4", 1236},
  {"CSUNICODE", 1200},
  {"CSUNICODE11", 1204},
  {"CSVISCII", 1129},
  {"CYRILLIC", 915},
  {"ECMA-114", 1089},
  {"ECMA-118", 813},
  {"ELOT_928", 813},
  {"EUC-CN", 1383},
  {"EUC-JP", 954},
  {"EUC-KR", 970},
  {"EUC-TW", 964},
  {"EUCCN", 1383},
  {"EUCJP", 954},
  {"EUCKR", 970},
  {"EUCTW", 964},
  {"EXTENDED_UNIX_CODE_PACKED_FORMAT_FOR_JAPANESE", 954},
  {"GB18030", 1392},
  {"GB2312", 1383},
  {"GBK", 1386},
  {"GREEK", 813},
  {"GREEK8", 813},
  {"HEBREW", 62210},
  {"HP-ROMAN8", 1051},
  {"IBM367", 367},
  {"IBM819", 819},
  {"IBM850", 850},
  {"IBM862", 862},
  {"IBM866", 866},
  {"ISO-10646-UCS-2", 1200},
  {"ISO-10646-UCS-4", 1236},
  {"ISO-8859-1", 819},
  {"ISO-8859-13", 901},
  {"ISO-8859-15", 923},
  {"ISO-8859-2", 912},
  {"ISO-8859-3", 913},
  {"ISO-8859-4", 914},
  {"ISO-8859-5", 915},
  {"ISO-8859-6", 1089},
  {"ISO-8859-7", 813},
  {"ISO-8859-8", 62210},
  {"ISO-8859-9", 920},
  {"ISO-IR-100", 819},
  {"ISO-IR-101", 912},
  {"ISO-IR-109", 913},
  {"ISO-IR-110", 914},
  {"ISO-IR-126", 813},
  {"ISO-IR-127", 1089},
  {"ISO-IR-138", 62210},
  {"ISO-IR-144", 915},
  {"ISO-IR-148", 920},
  {"ISO-IR-149", 970},
  {"ISO-IR-166", 874},
  {"ISO-IR-179", 901},
  {"ISO-IR-203", 923},
  {"ISO-IR-6", 367},
  {"ISO646-US", 367},
  {"ISO8859-1", 819},
  {"ISO8859-13", 901},
  {"ISO8859-15", 923},
  {"ISO8859-2", 912},
  {"ISO8859-3", 913},
  {"ISO8859-4", 914},
  {"ISO8859-5", 915},
  {"ISO8859-6", 1089},
  {"ISO8859-7", 813},
  {"ISO8859-8", 62210},
  {"ISO8859-9", 920},
  {"ISO_646.IRV:1991", 367},
  {"ISO_8859-1", 819},
  {"ISO_8859-13", 901},
  {"ISO_8859-15", 923},
  {"ISO_8859-15:1998", 923},
  {"ISO_8859-1:1987", 819},
  {"ISO_8859-2", 912},
  {"ISO_8859-2:1987", 912},
  {"ISO_8859-3", 913},
  {"ISO_8859-3:1988", 913},
  {"ISO_8859-4", 914},
  {"ISO_8859-4:1988", 914},
  {"ISO_8859-5", 915},
  {"ISO_8859-5:1988", 915},
  {"ISO_8859-6", 1089},
  {"ISO_8859-6:1987", 1089},
  {"ISO_8859-7", 813},
  {"ISO_8859-7:1987", 813},
  {"ISO_8859-7:2003", 813},
  {"ISO_8859-8", 62210},
  {"ISO_8859-8:1988", 62210},
  {"ISO_8859-9", 920},
  {"ISO_8859-9:1989", 920},
  {"JISX0201-1976", 896},
  {"JIS_X0201", 896},
  {"KOI8-R", 878},
  {"KOI8-U", 1168},
  {"KOREAN", 970},
  {"KSC_5601", 970},
  {"KS_C_5601-1987", 970},
  {"KS_C_5601-1989", 970},
  {"L1", 819},
  {"L2", 912},
  {"L3", 913},
  {"L4", 914},
  {"L5", 920},
  {"L7", 901},
  {"LATIN-9", 923},
  {"LATIN1", 819},
  {"LATIN2", 912},
  {"LATIN3", 913},
  {"LATIN4", 914},
  {"LATIN5", 920},
  {"LATIN7", 901},
  {"MAC", 1275},
  {"MACINTOSH", 1275},
  {"MACROMAN", 1275},
  {"MS-ANSI", 1252},
  {"MS-ARAB", 1256},
  {"MS-CYRL", 1251},
  {"MS-EE", 1250},
  {"MS-GREEK", 1253},
  {"MS-HEBR", 1255},
  {"MS-TURK", 1254},
  {"MS936", 1386},
  {"MS_KANJI", 943},
  {"R8", 1051},
  {"ROMAN8", 1051},
  {"SHIFT-JIS", 943},
  {"SHIFT_JIS", 943},
  {"SJIS", 943},
  {"TIS-620", 874},
  {"TIS620", 874},
  {"TIS620-0", 874},
  {"TIS620.2529-1", 874},
  {"TIS620.2533-0", 874},
  {"TIS620.2533-1", 874},
  {"UCS-2", 1200},
  {"UCS-2BE", 1204},
  {"UCS-4", 1236},
  {"UNICODE-1-1", 1204},
  {"UNICODEBIG", 1204},
  {"US", 367},
  {"US-ASCII", 367},
  {"UTF-16", 1204},
  {"UTF-16BE", 1200},
  {"UTF-16LE", 1202},
  {"UTF-32", 1236},
  {"UTF-32BE", 1232},
  {"UTF-32LE", 1234},
  {"UTF-8", 1208},
  {"VISCII", 1129},
  {"VISCII1.1-1", 1129},
  {"WINBALTRIM", 1257},
  {"WINDOWS-1250", 1250},
  {"WINDOWS-1251", 1251},
  {"WINDOWS-1252", 1252},
  {"WINDOWS-1253", 1253},
  {"WINDOWS-1254", 1254},
  {"WINDOWS-1255", 1255},
  {"WINDOWS-1256", 1256},
  {"WINDOWS-1257", 1257},
  {"WINDOWS-936", 1386},
  {"X0201", 896}
};

char *os2cpToName(unsigned long cp)
{
    ULONG ulLo = 0;
    ULONG ulHi = (sizeof(aCP2Name) / sizeof(struct _CP2NAME)) - 1;
    ULONG ulNext;
    LONG  lFound = -1;

    if (cp == SYSTEM_CP) {
        ULONG aulCP[3];
        ULONG cCP;
        if (DosQueryCp(sizeof(aulCP), aulCP, &cCP) != NO_ERROR) {
            return NULL;
        }
        cp = aulCP[0];
    }

    if (aCP2Name[0].ulCode > cp || aCP2Name[ulHi].ulCode < cp) {
        return NULL;
    }
    if (aCP2Name[0].ulCode == cp) {
        return aCP2Name[0].pszName;
    }
    if (aCP2Name[ulHi].ulCode == cp) {
        return aCP2Name[ulHi].pszName;
    }

    while ((ulHi - ulLo) > 1) {
        ulNext = (ulLo + ulHi) / 2;

        if (aCP2Name[ulNext].ulCode < cp) {
            ulLo = ulNext;
        } else if (aCP2Name[ulNext].ulCode > cp) {
            ulHi = ulNext;
        } else {
            lFound = ulNext;
            break;
        }
    }

    return (lFound == -1)? NULL : aCP2Name[lFound].pszName;
}

unsigned long os2cpFromName(char *cp)
{
    ULONG ulLo = 0;
    ULONG ulHi = (sizeof(aName2CP) / sizeof(struct _NAME2CP)) - 1;
    ULONG ulNext;
    LONG  lFound = -1;
    LONG  lCmp;
    PCHAR pcEnd;
    CHAR  acBuf[64];

    if (!cp) {
        ULONG aulCP[3];
        ULONG cCP;
        return (DosQueryCp(sizeof(aulCP), aulCP, &cCP) != NO_ERROR)? 0 : aulCP[0];
    }

    while (SDL_isspace((unsigned char) *cp)) {
        cp++;
    }

    pcEnd = SDL_strchr(cp, ' ');
    if (!pcEnd) {
        pcEnd = SDL_strchr(cp, '\0');
    }
    ulNext = pcEnd - cp;
    if (ulNext >= sizeof(acBuf)) {
        return 0;
    }

    SDL_memcpy(acBuf, cp, ulNext);
    acBuf[ulNext] = '\0';
    SDL_strupr(acBuf);

    lCmp = SDL_strcmp(aName2CP[0].pszName, acBuf);
    if (lCmp > 0) {
        return 0;
    }
    if (lCmp == 0) {
        return aName2CP[0].ulCode;
    }

    lCmp = SDL_strcmp(aName2CP[ulHi].pszName, acBuf);
    if (lCmp < 0) {
        return 0;
    }
    if (lCmp == 0) {
        return aName2CP[ulHi].ulCode;
    }

    while ((ulHi - ulLo) > 1) {
        ulNext = (ulLo + ulHi) / 2;

        lCmp = SDL_strcmp(aName2CP[ulNext].pszName, acBuf);
        if (lCmp < 0) {
            ulLo = ulNext;
        } else if (lCmp > 0) {
            ulHi = ulNext;
        } else {
            lFound = ulNext;
            break;
        }
    }

    return (lFound == -1)? 0 : aName2CP[lFound].ulCode;
}

/* vi: set ts=4 sw=4 expandtab: */
