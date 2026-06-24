/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>
  Copyright (C) 2022 Collabora Ltd.

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

#include "SDL_sandbox.h"

#include <unistd.h>

SDL_Sandbox SDL_DetectSandbox(void)
{
    if (access("/.flatpak-info", F_OK) == 0) {
        return SDL_SANDBOX_FLATPAK;
    }

    /* For Snap, we check multiple variables because they might be set for
     * unrelated reasons. This is the same thing WebKitGTK does. */
    if (SDL_getenv("SNAP") != NULL && SDL_getenv("SNAP_NAME") != NULL && SDL_getenv("SNAP_REVISION") != NULL) {
        return SDL_SANDBOX_SNAP;
    }

    if (access("/run/host/container-manager", F_OK) == 0) {
        return SDL_SANDBOX_UNKNOWN_CONTAINER;
    }

    return SDL_SANDBOX_NONE;
}

/* vi: set ts=4 sw=4 expandtab: */
