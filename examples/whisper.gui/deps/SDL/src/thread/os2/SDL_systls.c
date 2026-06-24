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

#include "../../SDL_internal.h"

#ifdef SDL_THREAD_OS2

#include "../../core/os2/SDL_os2.h"

#include "SDL_thread.h"
#include "../SDL_thread_c.h"

#define INCL_DOSPROCESS
#define INCL_DOSERRORS
#include <os2.h>

SDL_TLSData **ppSDLTLSData = NULL;

void SDL_SYS_InitTLSData(void)
{
    ULONG ulRC;

    if (!ppSDLTLSData) {
        /* Allocate the thread local memory (1 DWORD) */
        ulRC = DosAllocThreadLocalMemory(1, (PULONG *)&ppSDLTLSData);
        if (ulRC != NO_ERROR) {
            debug_os2("DosAllocThreadLocalMemory() failed, rc = %u", ulRC);
        }
    }
}

SDL_TLSData *SDL_SYS_GetTLSData(void)
{
    return ppSDLTLSData ? *ppSDLTLSData : NULL;
}

int SDL_SYS_SetTLSData(SDL_TLSData *data)
{
    if (!ppSDLTLSData)
        return -1;

    *ppSDLTLSData = data;
    return 0;
}

void SDL_SYS_QuitTLSData(void)
{
    ULONG ulRC;

    if (ppSDLTLSData) {
        /* Free the thread local memory */
        ulRC = DosFreeThreadLocalMemory((PULONG)ppSDLTLSData);
        if (ulRC != NO_ERROR) {
            debug_os2("DosFreeThreadLocalMemory() failed, rc = %u", ulRC);
        } else {
            ppSDLTLSData = NULL;
        }
    }
}

#endif /* SDL_THREAD_OS2 */

/* vi: set ts=4 sw=4 expandtab: */
