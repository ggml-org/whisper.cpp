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

#ifdef SDL_VIDEO_DRIVER_NACL

/* Include the SDL main definition header */
#include "SDL_main.h"

#include "ppapi_simple/ps_main.h"
#include "ppapi_simple/ps_event.h"
#include "ppapi_simple/ps_interface.h"
#include "nacl_io/nacl_io.h"
#include "sys/mount.h"

extern void NACL_SetScreenResolution(int width, int height, Uint32 format);

int nacl_main(int argc, char *argv[])
{
    int status;
    PSEvent* ps_event;
    PP_Resource event;
    struct PP_Rect rect;
    int ready = 0;
    const PPB_View *ppb_view = PSInterfaceView();

    /* This is started in a worker thread by ppapi_simple! */

    /* Wait for the first PSE_INSTANCE_DIDCHANGEVIEW event before starting the app */

    PSEventSetFilter(PSE_INSTANCE_DIDCHANGEVIEW);
    while (!ready) {
        /* Process all waiting events without blocking */
        while (!ready && (ps_event = PSEventWaitAcquire()) != NULL) {
            event = ps_event->as_resource;
            switch(ps_event->type) {
                /* From DidChangeView, contains a view resource */
                case PSE_INSTANCE_DIDCHANGEVIEW:
                    ppb_view->GetRect(event, &rect);
                    NACL_SetScreenResolution(rect.size.width, rect.size.height, 0);
                    ready = 1;
                    break;
                default:
                    break;
            }
            PSEventRelease(ps_event);
        }
    }

    /* Do a default httpfs mount on /,
     * apps can override this by unmounting /
     * and remounting with the desired configuration
     */
    nacl_io_init_ppapi(PSGetInstanceId(), PSGetInterface);

    umount("/");
    mount(
        "",  /* source */
        "/",  /* target */
        "httpfs",  /* filesystemtype */
        0,  /* mountflags */
        "");  /* data specific to the html5fs type */

    /* Everything is ready, start the user main function */
    SDL_SetMainReady();
    status = SDL_main(argc, argv);

    return 0;
}

/* ppapi_simple will start nacl_main in a worker thread */
PPAPI_SIMPLE_REGISTER_MAIN(nacl_main);

#endif /* SDL_VIDEO_DRIVER_NACL */
