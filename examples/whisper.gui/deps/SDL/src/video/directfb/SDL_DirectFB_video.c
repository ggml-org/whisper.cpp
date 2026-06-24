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

#ifdef SDL_VIDEO_DRIVER_DIRECTFB

/*
 * #include "SDL_DirectFB_keyboard.h"
 */
#include "SDL_DirectFB_modes.h"
#include "SDL_DirectFB_opengl.h"
#include "SDL_DirectFB_vulkan.h"
#include "SDL_DirectFB_window.h"
#include "SDL_DirectFB_WM.h"


/* DirectFB video driver implementation.
*/

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <directfb.h>
#include <directfb_version.h>
#include <directfb_strings.h>

#include "SDL_video.h"
#include "SDL_mouse.h"
#include "../SDL_sysvideo.h"
#include "../SDL_pixels_c.h"
#include "../../events/SDL_events_c.h"
#include "SDL_DirectFB_video.h"
#include "SDL_DirectFB_events.h"
#include "SDL_DirectFB_render.h"
#include "SDL_DirectFB_mouse.h"
#include "SDL_DirectFB_shape.h"


#include "SDL_DirectFB_dyn.h"

/* Initialization/Query functions */
static int DirectFB_VideoInit(_THIS);
static void DirectFB_VideoQuit(_THIS);

static SDL_VideoDevice *DirectFB_CreateDevice(void);

VideoBootStrap DirectFB_bootstrap = {
    "directfb", "DirectFB",
    DirectFB_CreateDevice,
    NULL /* no ShowMessageBox implementation */
};

static const DirectFBSurfaceDrawingFlagsNames(drawing_flags);
static const DirectFBSurfaceBlittingFlagsNames(blitting_flags);
static const DirectFBAccelerationMaskNames(acceleration_mask);

/* DirectFB driver bootstrap functions */

static void DirectFB_DeleteDevice(SDL_VideoDevice * device)
{
    SDL_DirectFB_UnLoadLibrary();
    SDL_DFB_FREE(device->driverdata);
    SDL_DFB_FREE(device);
}

static SDL_VideoDevice *DirectFB_CreateDevice(void)
{
    SDL_VideoDevice *device;

    if (!SDL_DirectFB_LoadLibrary()) {
        return NULL;
    }

    /* Initialize all variables that we clean on shutdown */
    SDL_DFB_ALLOC_CLEAR(device, sizeof(SDL_VideoDevice));

    /* Set the function pointers */
    device->VideoInit = DirectFB_VideoInit;
    device->VideoQuit = DirectFB_VideoQuit;
    device->GetDisplayModes = DirectFB_GetDisplayModes;
    device->SetDisplayMode = DirectFB_SetDisplayMode;
    device->PumpEvents = DirectFB_PumpEventsWindow;
    device->CreateSDLWindow = DirectFB_CreateWindow;
    device->CreateSDLWindowFrom = DirectFB_CreateWindowFrom;
    device->SetWindowTitle = DirectFB_SetWindowTitle;
    device->SetWindowIcon = DirectFB_SetWindowIcon;
    device->SetWindowPosition = DirectFB_SetWindowPosition;
    device->SetWindowSize = DirectFB_SetWindowSize;
    device->SetWindowOpacity = DirectFB_SetWindowOpacity;
    device->ShowWindow = DirectFB_ShowWindow;
    device->HideWindow = DirectFB_HideWindow;
    device->RaiseWindow = DirectFB_RaiseWindow;
    device->MaximizeWindow = DirectFB_MaximizeWindow;
    device->MinimizeWindow = DirectFB_MinimizeWindow;
    device->RestoreWindow = DirectFB_RestoreWindow;
    device->SetWindowMouseGrab = DirectFB_SetWindowMouseGrab;
    device->SetWindowKeyboardGrab = DirectFB_SetWindowKeyboardGrab;
    device->DestroyWindow = DirectFB_DestroyWindow;
    device->GetWindowWMInfo = DirectFB_GetWindowWMInfo;

    /* !!! FIXME: implement SetWindowBordered */

#ifdef SDL_DIRECTFB_OPENGL
    device->GL_LoadLibrary = DirectFB_GL_LoadLibrary;
    device->GL_GetProcAddress = DirectFB_GL_GetProcAddress;
    device->GL_MakeCurrent = DirectFB_GL_MakeCurrent;

    device->GL_CreateContext = DirectFB_GL_CreateContext;
    device->GL_SetSwapInterval = DirectFB_GL_SetSwapInterval;
    device->GL_GetSwapInterval = DirectFB_GL_GetSwapInterval;
    device->GL_SwapWindow = DirectFB_GL_SwapWindow;
    device->GL_DeleteContext = DirectFB_GL_DeleteContext;

#endif

    /* Shaped window support */
    device->shape_driver.CreateShaper = DirectFB_CreateShaper;
    device->shape_driver.SetWindowShape = DirectFB_SetWindowShape;
    device->shape_driver.ResizeWindowShape = DirectFB_ResizeWindowShape;

#ifdef SDL_VIDEO_VULKAN
    device->Vulkan_LoadLibrary = DirectFB_Vulkan_LoadLibrary;
    device->Vulkan_UnloadLibrary = DirectFB_Vulkan_UnloadLibrary;
    device->Vulkan_GetInstanceExtensions = DirectFB_Vulkan_GetInstanceExtensions;
    device->Vulkan_CreateSurface = DirectFB_Vulkan_CreateSurface;
#endif

    device->free = DirectFB_DeleteDevice;

    return device;
  error:
    if (device)
        SDL_free(device);
    return (0);
}

static void DirectFB_DeviceInformation(IDirectFB * dfb)
{
    DFBGraphicsDeviceDescription desc;
    int n;

    dfb->GetDeviceDescription(dfb, &desc);

    SDL_DFB_LOG( "DirectFB Device Information");
    SDL_DFB_LOG( "===========================");
    SDL_DFB_LOG( "Name:           %s", desc.name);
    SDL_DFB_LOG( "Vendor:         %s", desc.vendor);
    SDL_DFB_LOG( "Driver Name:    %s", desc.driver.name);
    SDL_DFB_LOG( "Driver Vendor:  %s", desc.driver.vendor);
    SDL_DFB_LOG( "Driver Version: %d.%d", desc.driver.major,
            desc.driver.minor);

    SDL_DFB_LOG( "Video memory:   %d", desc.video_memory);

    SDL_DFB_LOG( "Blitting flags:");
    for (n = 0; blitting_flags[n].flag; n++) {
        if (desc.blitting_flags & blitting_flags[n].flag)
            SDL_DFB_LOG( "    %s", blitting_flags[n].name);
    }

    SDL_DFB_LOG( "Drawing flags:");
    for (n = 0; drawing_flags[n].flag; n++) {
        if (desc.drawing_flags & drawing_flags[n].flag)
            SDL_DFB_LOG( "    %s", drawing_flags[n].name);
    }


    SDL_DFB_LOG( "Acceleration flags:");
    for (n = 0; acceleration_mask[n].mask; n++) {
        if (desc.acceleration_mask & acceleration_mask[n].mask)
            SDL_DFB_LOG( "    %s", acceleration_mask[n].name);
    }


}

static int readBoolEnv(const char *env_name, int def_val)
{
    char *stemp;

    stemp = SDL_getenv(env_name);
    if (stemp)
        return SDL_atoi(stemp);
    else
        return def_val;
}

static int DirectFB_VideoInit(_THIS)
{
    IDirectFB *dfb = NULL;
    DFB_DeviceData *devdata = NULL;
    DFBResult ret;

    SDL_DFB_ALLOC_CLEAR(devdata, sizeof(*devdata));

    SDL_DFB_CHECKERR(DirectFBInit(NULL, NULL));

    /* avoid switching to the framebuffer when we
     * are running X11 */
    ret = readBoolEnv(DFBENV_USE_X11_CHECK , 1);
    if (ret) {
        if (SDL_getenv("DISPLAY"))
            DirectFBSetOption("system", "x11");
        else
            DirectFBSetOption("disable-module", "x11input");
    }

    devdata->use_linux_input = readBoolEnv(DFBENV_USE_LINUX_INPUT, 1);       /* default: on */

    if (!devdata->use_linux_input)
    {
        SDL_DFB_LOG("Disabling linux input\n");
        DirectFBSetOption("disable-module", "linux_input");
    }

    SDL_DFB_CHECKERR(DirectFBCreate(&dfb));

    DirectFB_DeviceInformation(dfb);

    devdata->use_yuv_underlays = readBoolEnv(DFBENV_USE_YUV_UNDERLAY, 0);     /* default: off */
    devdata->use_yuv_direct = readBoolEnv(DFBENV_USE_YUV_DIRECT, 0);      /* default is off! */

    /* Create global Eventbuffer for axis events */
    if (devdata->use_linux_input) {
        SDL_DFB_CHECKERR(dfb->CreateInputEventBuffer(dfb, DICAPS_ALL,
                                                     DFB_TRUE,
                                                     &devdata->events));
    } else {
        SDL_DFB_CHECKERR(dfb->CreateInputEventBuffer(dfb, DICAPS_AXES
                                                     /* DICAPS_ALL */ ,
                                                     DFB_TRUE,
                                                     &devdata->events));
    }

    /* simple window manager support */
    devdata->has_own_wm = readBoolEnv(DFBENV_USE_WM, 0);

    devdata->initialized = 1;

    devdata->dfb = dfb;
    devdata->firstwin = NULL;

    _this->driverdata = devdata;

    DirectFB_InitModes(_this);

#ifdef SDL_DIRECTFB_OPENGL
    DirectFB_GL_Initialize(_this);
#endif

    DirectFB_InitMouse(_this);
    DirectFB_InitKeyboard(_this);

    return 0;


  error:
    SDL_DFB_FREE(devdata);
    SDL_DFB_RELEASE(dfb);
    return -1;
}

static void DirectFB_VideoQuit(_THIS)
{
    DFB_DeviceData *devdata = (DFB_DeviceData *) _this->driverdata;

    DirectFB_QuitModes(_this);
    DirectFB_QuitKeyboard(_this);
    DirectFB_QuitMouse(_this);

    devdata->events->Reset(devdata->events);
    SDL_DFB_RELEASE(devdata->events);
    SDL_DFB_RELEASE(devdata->dfb);

#ifdef SDL_DIRECTFB_OPENGL
    DirectFB_GL_Shutdown(_this);
#endif

    devdata->initialized = 0;
}

/* DirectFB driver general support functions */

static const struct {
    DFBSurfacePixelFormat dfb;
    Uint32 sdl;
} pixelformat_tab[] =
{
    { DSPF_RGB32, SDL_PIXELFORMAT_RGB888 },             /* 24 bit RGB (4 byte, nothing@24, red 8@16, green 8@8, blue 8@0) */
    { DSPF_ARGB, SDL_PIXELFORMAT_ARGB8888 },            /* 32 bit ARGB (4 byte, alpha 8@24, red 8@16, green 8@8, blue 8@0) */
    { DSPF_RGB16, SDL_PIXELFORMAT_RGB565 },             /* 16 bit RGB (2 byte, red 5@11, green 6@5, blue 5@0) */
    { DSPF_RGB332, SDL_PIXELFORMAT_RGB332 },            /* 8 bit RGB (1 byte, red 3@5, green 3@2, blue 2@0) */
    { DSPF_ARGB4444, SDL_PIXELFORMAT_ARGB4444 },        /* 16 bit ARGB (2 byte, alpha 4@12, red 4@8, green 4@4, blue 4@0) */
    { DSPF_ARGB1555, SDL_PIXELFORMAT_ARGB1555 },        /* 16 bit ARGB (2 byte, alpha 1@15, red 5@10, green 5@5, blue 5@0) */
    { DSPF_RGB24, SDL_PIXELFORMAT_RGB24 },              /* 24 bit RGB (3 byte, red 8@16, green 8@8, blue 8@0) */
    { DSPF_RGB444, SDL_PIXELFORMAT_RGB444 },            /* 16 bit RGB (2 byte, nothing @12, red 4@8, green 4@4, blue 4@0) */
    { DSPF_YV12, SDL_PIXELFORMAT_YV12 },                /* 12 bit YUV (8 bit Y plane followed by 8 bit quarter size V/U planes) */
    { DSPF_I420,SDL_PIXELFORMAT_IYUV },                 /* 12 bit YUV (8 bit Y plane followed by 8 bit quarter size U/V planes) */
    { DSPF_YUY2, SDL_PIXELFORMAT_YUY2 },                /* 16 bit YUV (4 byte/ 2 pixel, macropixel contains CbYCrY [31:0]) */
    { DSPF_UYVY, SDL_PIXELFORMAT_UYVY },                /* 16 bit YUV (4 byte/ 2 pixel, macropixel contains YCbYCr [31:0]) */
    { DSPF_RGB555, SDL_PIXELFORMAT_RGB555 },            /* 16 bit RGB (2 byte, nothing @15, red 5@10, green 5@5, blue 5@0) */
#if (DFB_VERSION_ATLEAST(1,5,0))
    { DSPF_ABGR, SDL_PIXELFORMAT_ABGR8888 },            /* 32 bit ABGR (4  byte, alpha 8@24, blue 8@16, green 8@8, red 8@0) */
#endif
#if (ENABLE_LUT8)
    { DSPF_LUT8, SDL_PIXELFORMAT_INDEX8 },              /* 8 bit LUT (8 bit color and alpha lookup from palette) */
#endif

#if (DFB_VERSION_ATLEAST(1,2,0))
    { DSPF_BGR555, SDL_PIXELFORMAT_BGR555 },            /* 16 bit BGR (2 byte, nothing @15, blue 5@10, green 5@5, red 5@0) */
#else
    { DSPF_UNKNOWN, SDL_PIXELFORMAT_BGR555 },
#endif

    /* Pfff ... nonmatching formats follow */

    { DSPF_ALUT44, SDL_PIXELFORMAT_UNKNOWN },           /* 8 bit ALUT (1 byte, alpha 4@4, color lookup 4@0) */
    { DSPF_A8, SDL_PIXELFORMAT_UNKNOWN },               /*  8 bit alpha (1 byte, alpha 8@0), e.g. anti-aliased glyphs */
    { DSPF_AiRGB, SDL_PIXELFORMAT_UNKNOWN },            /*  32 bit ARGB (4 byte, inv. alpha 8@24, red 8@16, green 8@8, blue 8@0) */
    { DSPF_A1, SDL_PIXELFORMAT_UNKNOWN },               /*  1 bit alpha (1 byte/ 8 pixel, most significant bit used first) */
    { DSPF_NV12, SDL_PIXELFORMAT_UNKNOWN },             /*  12 bit YUV (8 bit Y plane followed by one 16 bit quarter size CbCr [15:0] plane) */
    { DSPF_NV16, SDL_PIXELFORMAT_UNKNOWN },             /*  16 bit YUV (8 bit Y plane followed by one 16 bit half width CbCr [15:0] plane) */
    { DSPF_ARGB2554, SDL_PIXELFORMAT_UNKNOWN },         /*  16 bit ARGB (2 byte, alpha 2@14, red 5@9, green 5@4, blue 4@0) */
    { DSPF_NV21, SDL_PIXELFORMAT_UNKNOWN },             /*  12 bit YUV (8 bit Y plane followed by one 16 bit quarter size CrCb [15:0] plane) */
    { DSPF_AYUV, SDL_PIXELFORMAT_UNKNOWN },             /*  32 bit AYUV (4 byte, alpha 8@24, Y 8@16, Cb 8@8, Cr 8@0) */
    { DSPF_A4, SDL_PIXELFORMAT_UNKNOWN },               /*  4 bit alpha (1 byte/ 2 pixel, more significant nibble used first) */
    { DSPF_ARGB1666, SDL_PIXELFORMAT_UNKNOWN },         /*  1 bit alpha (3 byte/ alpha 1@18, red 6@16, green 6@6, blue 6@0) */
    { DSPF_ARGB6666, SDL_PIXELFORMAT_UNKNOWN },         /*  6 bit alpha (3 byte/ alpha 6@18, red 6@16, green 6@6, blue 6@0) */
    { DSPF_RGB18, SDL_PIXELFORMAT_UNKNOWN },            /*  6 bit RGB (3 byte/ red 6@16, green 6@6, blue 6@0) */
    { DSPF_LUT2, SDL_PIXELFORMAT_UNKNOWN },             /*  2 bit LUT (1 byte/ 4 pixel, 2 bit color and alpha lookup from palette) */

#if (DFB_VERSION_ATLEAST(1,3,0))
    { DSPF_RGBA4444, SDL_PIXELFORMAT_UNKNOWN },         /* 16 bit RGBA (2 byte, red 4@12, green 4@8, blue 4@4, alpha 4@0) */
#endif

#if (DFB_VERSION_ATLEAST(1,4,3))
    { DSPF_RGBA5551, SDL_PIXELFORMAT_UNKNOWN },         /*  16 bit RGBA (2 byte, red 5@11, green 5@6, blue 5@1, alpha 1@0) */
    { DSPF_YUV444P, SDL_PIXELFORMAT_UNKNOWN },          /*  24 bit full YUV planar (8 bit Y plane followed by an 8 bit Cb and an 8 bit Cr plane) */
    { DSPF_ARGB8565, SDL_PIXELFORMAT_UNKNOWN },         /*  24 bit ARGB (3 byte, alpha 8@16, red 5@11, green 6@5, blue 5@0) */
    { DSPF_AVYU, SDL_PIXELFORMAT_UNKNOWN },             /*  32 bit AVYU 4:4:4 (4 byte, alpha 8@24, Cr 8@16, Y 8@8, Cb 8@0) */
    { DSPF_VYU, SDL_PIXELFORMAT_UNKNOWN },              /*  24 bit VYU 4:4:4 (3 byte, Cr 8@16, Y 8@8, Cb 8@0)  */
#endif

    { DSPF_UNKNOWN, SDL_PIXELFORMAT_INDEX1LSB },
    { DSPF_UNKNOWN, SDL_PIXELFORMAT_INDEX1MSB },
    { DSPF_UNKNOWN, SDL_PIXELFORMAT_INDEX4LSB },
    { DSPF_UNKNOWN, SDL_PIXELFORMAT_INDEX4MSB },
    { DSPF_UNKNOWN, SDL_PIXELFORMAT_BGR24 },
    { DSPF_UNKNOWN, SDL_PIXELFORMAT_BGR888 },
    { DSPF_UNKNOWN, SDL_PIXELFORMAT_RGBA8888 },
    { DSPF_UNKNOWN, SDL_PIXELFORMAT_BGRA8888 },
    { DSPF_UNKNOWN, SDL_PIXELFORMAT_ARGB2101010 },
    { DSPF_UNKNOWN, SDL_PIXELFORMAT_ABGR4444 },
    { DSPF_UNKNOWN, SDL_PIXELFORMAT_ABGR1555 },
    { DSPF_UNKNOWN, SDL_PIXELFORMAT_BGR565 },
    { DSPF_UNKNOWN, SDL_PIXELFORMAT_YVYU },                        /**< Packed mode: Y0+V0+Y1+U0 (1 pla */
};

Uint32 DirectFB_DFBToSDLPixelFormat(DFBSurfacePixelFormat pixelformat)
{
    int i;

    for (i=0; pixelformat_tab[i].dfb != DSPF_UNKNOWN; i++)
        if (pixelformat_tab[i].dfb == pixelformat)
        {
            return pixelformat_tab[i].sdl;
        }
    return SDL_PIXELFORMAT_UNKNOWN;
}

DFBSurfacePixelFormat DirectFB_SDLToDFBPixelFormat(Uint32 format)
{
    int i;

    for (i=0; pixelformat_tab[i].dfb != DSPF_UNKNOWN; i++)
        if (pixelformat_tab[i].sdl == format)
        {
            return pixelformat_tab[i].dfb;
        }
    return DSPF_UNKNOWN;
}

void DirectFB_SetSupportedPixelFormats(SDL_RendererInfo* ri)
{
    int i, j;

    for (i=0, j=0; pixelformat_tab[i].dfb != DSPF_UNKNOWN; i++)
        if (pixelformat_tab[i].sdl != SDL_PIXELFORMAT_UNKNOWN)
            ri->texture_formats[j++] = pixelformat_tab[i].sdl;
    ri->num_texture_formats = j;
}

#endif /* SDL_VIDEO_DRIVER_DIRECTFB */
