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

#ifndef SDL_DirectFB_dyn_h_
#define SDL_DirectFB_dyn_h_

#define DFB_SYMS \
    DFB_SYM(DFBResult, DirectFBError, (const char *msg, DFBResult result), (msg, result), return) \
    DFB_SYM(DFBResult, DirectFBErrorFatal, (const char *msg, DFBResult result), (msg, result), return) \
    DFB_SYM(const char *, DirectFBErrorString, (DFBResult result), (result), return) \
    DFB_SYM(const char *, DirectFBUsageString, ( void ), (), return) \
    DFB_SYM(DFBResult, DirectFBInit, (int *argc, char *(*argv[]) ), (argc, argv), return) \
    DFB_SYM(DFBResult, DirectFBSetOption, (const char *name, const char *value), (name, value), return) \
    DFB_SYM(DFBResult, DirectFBCreate, (IDirectFB **interface), (interface), return) \
    DFB_SYM(const char *, DirectFBCheckVersion, (unsigned int required_major, unsigned int required_minor, unsigned int required_micro), \
                (required_major, required_minor, required_micro), return)

int SDL_DirectFB_LoadLibrary(void);
void SDL_DirectFB_UnLoadLibrary(void);

#endif /* SDL_DirectFB_dyn_h_ */

/* vi: set ts=4 sw=4 expandtab: */
