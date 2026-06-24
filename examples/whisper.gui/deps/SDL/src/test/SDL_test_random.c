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

 A portable "32-bit Multiply with carry" random number generator.

 Used by the fuzzer component.
 Original source code contributed by A. Schiffler for GSOC project.

*/

#include "SDL_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "SDL_test.h"

/* Initialize random number generator with two integer variables */

void SDLTest_RandomInit(SDLTest_RandomContext *rndContext, unsigned int xi, unsigned int ci)
{
    if (!rndContext) {
        return;
    }

    /*
     * Choose a value for 'a' from this list
     * 1791398085 1929682203 1683268614 1965537969 1675393560
     * 1967773755 1517746329 1447497129 1655692410 1606218150
     * 2051013963 1075433238 1557985959 1781943330 1893513180
     * 1631296680 2131995753 2083801278 1873196400 1554115554
     */
    rndContext->a = 1655692410;
    rndContext->x = 30903;
    rndContext->c = 0;
    if (xi != 0) {
        rndContext->x = xi;
    }
    rndContext->c = ci;
    rndContext->ah = rndContext->a >> 16;
    rndContext->al = rndContext->a & 65535;
}

/* Initialize random number generator from system time */

void SDLTest_RandomInitTime(SDLTest_RandomContext *rndContext)
{
    int a, b;

    if (!rndContext) {
        return;
    }

    srand((unsigned int)time(NULL));
    a = rand();
    srand((unsigned int)SDL_GetPerformanceCounter());
    b = rand();
    SDLTest_RandomInit(rndContext, a, b);
}

/* Returns random numbers */

unsigned int SDLTest_Random(SDLTest_RandomContext *rndContext)
{
    unsigned int xh, xl;

    if (!rndContext) {
        return -1;
    }

    xh = rndContext->x >> 16;
    xl = rndContext->x & 65535;
    rndContext->x = rndContext->x * rndContext->a + rndContext->c;
    rndContext->c =
        xh * rndContext->ah + ((xh * rndContext->al) >> 16) +
        ((xl * rndContext->ah) >> 16);
    if (xl * rndContext->al >= (~rndContext->c + 1)) {
        rndContext->c++;
    }
    return rndContext->x;
}

/* vi: set ts=4 sw=4 expandtab: */
