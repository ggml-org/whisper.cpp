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

 Used by the test execution component.
 Original source code contributed by A. Schiffler for GSOC project.

*/

#include "SDL_config.h"

#include "SDL_test.h"

int SDLTest_Crc32Init(SDLTest_Crc32Context *crcContext)
{
    int i, j;
    CrcUint32 c;

    /* Sanity check context pointer */
    if (!crcContext) {
        return -1;
    }

    /*
     * Build auxiliary table for parallel byte-at-a-time CRC-32
     */
#ifdef ORIGINAL_METHOD
    for (i = 0; i < 256; ++i) {
        for (c = i << 24, j = 8; j > 0; --j) {
            c = c & 0x80000000 ? (c << 1) ^ CRC32_POLY : (c << 1);
        }
        crcContext->crc32_table[i] = c;
    }
#else
    for (i = 0; i < 256; i++) {
        c = i;
        for (j = 8; j > 0; j--) {
            if (c & 1) {
                c = (c >> 1) ^ CRC32_POLY;
            } else {
                c >>= 1;
            }
        }
        crcContext->crc32_table[i] = c;
    }
#endif

    return 0;
}

/* Complete CRC32 calculation on a memory block */
int SDLTest_Crc32Calc(SDLTest_Crc32Context *crcContext, CrcUint8 *inBuf, CrcUint32 inLen, CrcUint32 *crc32)
{
    if (SDLTest_Crc32CalcStart(crcContext, crc32)) {
        return -1;
    }

    if (SDLTest_Crc32CalcBuffer(crcContext, inBuf, inLen, crc32)) {
        return -1;
    }

    if (SDLTest_Crc32CalcEnd(crcContext, crc32)) {
        return -1;
    }

    return 0;
}

/* Start crc calculation */

int SDLTest_Crc32CalcStart(SDLTest_Crc32Context *crcContext, CrcUint32 *crc32)
{
    /* Sanity check pointers */
    if (!crcContext) {
        *crc32 = 0;
        return -1;
    }

    /*
     * Preload shift register, per CRC-32 spec
     */
    *crc32 = 0xffffffff;

    return 0;
}

/* Finish crc calculation */

int SDLTest_Crc32CalcEnd(SDLTest_Crc32Context *crcContext, CrcUint32 *crc32)
{
    /* Sanity check pointers */
    if (!crcContext) {
        *crc32 = 0;
        return -1;
    }

    /*
     * Return complement, per CRC-32 spec
     */
    *crc32 = (~(*crc32));

    return 0;
}

/* Include memory block in crc */

int SDLTest_Crc32CalcBuffer(SDLTest_Crc32Context *crcContext, CrcUint8 *inBuf, CrcUint32 inLen, CrcUint32 *crc32)
{
    CrcUint8 *p;
    register CrcUint32 crc;

    if (!crcContext) {
        *crc32 = 0;
        return -1;
    }

    if (!inBuf) {
        return -1;
    }

    /*
     * Calculate CRC from data
     */
    crc = *crc32;
    for (p = inBuf; inLen > 0; ++p, --inLen) {
#ifdef ORIGINAL_METHOD
        crc = (crc << 8) ^ crcContext->crc32_table[(crc >> 24) ^ *p];
#else
        crc = ((crc >> 8) & 0x00FFFFFF) ^ crcContext->crc32_table[(crc ^ *p) & 0xFF];
#endif
    }
    *crc32 = crc;

    return 0;
}

int SDLTest_Crc32Done(SDLTest_Crc32Context *crcContext)
{
    if (!crcContext) {
        return -1;
    }

    return 0;
}

/* vi: set ts=4 sw=4 expandtab: */
