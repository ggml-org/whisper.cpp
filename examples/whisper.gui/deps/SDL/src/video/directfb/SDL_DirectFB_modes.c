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

#include "SDL_DirectFB_video.h"
#include "SDL_DirectFB_modes.h"

#define DFB_MAX_MODES 200

struct screen_callback_t
{
    int numscreens;
    DFBScreenID screenid[DFB_MAX_SCREENS];
    DFBDisplayLayerID gralayer[DFB_MAX_SCREENS];
    DFBDisplayLayerID vidlayer[DFB_MAX_SCREENS];
    int aux;                    /* auxiliary integer for callbacks */
};

struct modes_callback_t
{
    int nummodes;
    SDL_DisplayMode *modelist;
};

static DFBEnumerationResult EnumModesCallback(int width, int height, int bpp, void *data)
{
    struct modes_callback_t *modedata = (struct modes_callback_t *) data;
    SDL_DisplayMode mode;

    mode.w = width;
    mode.h = height;
    mode.refresh_rate = 0;
    mode.driverdata = NULL;
    mode.format = SDL_PIXELFORMAT_UNKNOWN;

    if (modedata->nummodes < DFB_MAX_MODES) {
        modedata->modelist[modedata->nummodes++] = mode;
    }

    return DFENUM_OK;
}

static DFBEnumerationResult EnumScreensCallback(DFBScreenID screen_id, DFBScreenDescription desc, void *callbackdata)
{
    struct screen_callback_t *devdata = (struct screen_callback_t *) callbackdata;

    devdata->screenid[devdata->numscreens++] = screen_id;
    return DFENUM_OK;
}

static DFBEnumerationResult EnumLayersCallback(DFBDisplayLayerID layer_id, DFBDisplayLayerDescription desc, void *callbackdata)
{
    struct screen_callback_t *devdata = (struct screen_callback_t *) callbackdata;

    if (desc.caps & DLCAPS_SURFACE) {
        if ((desc.type & DLTF_GRAPHICS) && (desc.type & DLTF_VIDEO)) {
            if (devdata->vidlayer[devdata->aux] == -1)
                devdata->vidlayer[devdata->aux] = layer_id;
        } else if (desc.type & DLTF_GRAPHICS) {
            if (devdata->gralayer[devdata->aux] == -1)
                devdata->gralayer[devdata->aux] = layer_id;
        }
    }
    return DFENUM_OK;
}

static void CheckSetDisplayMode(_THIS, SDL_VideoDisplay * display, DFB_DisplayData * data, SDL_DisplayMode * mode)
{
    SDL_DFB_DEVICEDATA(_this);
    DFBDisplayLayerConfig config;
    DFBDisplayLayerConfigFlags failed;

    SDL_DFB_CHECKERR(data->layer->SetCooperativeLevel(data->layer,
                                                      DLSCL_ADMINISTRATIVE));
    config.width = mode->w;
    config.height = mode->h;
    config.pixelformat = DirectFB_SDLToDFBPixelFormat(mode->format);
    config.flags = DLCONF_WIDTH | DLCONF_HEIGHT | DLCONF_PIXELFORMAT;
    if (devdata->use_yuv_underlays) {
        config.flags |= DLCONF_OPTIONS;
        config.options = DLOP_ALPHACHANNEL;
    }
    failed = 0;
    data->layer->TestConfiguration(data->layer, &config, &failed);
    SDL_DFB_CHECKERR(data->layer->SetCooperativeLevel(data->layer,
                                                      DLSCL_SHARED));
    if (failed == 0)
    {
        SDL_AddDisplayMode(display, mode);
        SDL_DFB_LOG("Mode %d x %d Added\n", mode->w, mode->h);
    }
    else
        SDL_DFB_ERR("Mode %d x %d not available: %x\n", mode->w,
                      mode->h, failed);

    return;
  error:
    return;
}


void DirectFB_SetContext(_THIS, SDL_Window *window)
{
#if (DFB_VERSION_ATLEAST(1,0,0))
    /* FIXME: does not work on 1.0/1.2 with radeon driver
     *        the approach did work with the matrox driver
     *        This has simply no effect.
     */

    SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);
    DFB_DisplayData *dispdata = (DFB_DisplayData *) display->driverdata;

    /* FIXME: should we handle the error */
    if (dispdata->vidIDinuse)
        SDL_DFB_CHECK(dispdata->vidlayer->SwitchContext(dispdata->vidlayer,
                                                           DFB_TRUE));
#endif
}

void DirectFB_InitModes(_THIS)
{
    SDL_DFB_DEVICEDATA(_this);
    IDirectFBDisplayLayer *layer = NULL;
    SDL_VideoDisplay display;
    DFB_DisplayData *dispdata = NULL;
    SDL_DisplayMode mode;
    DFBGraphicsDeviceDescription caps;
    DFBDisplayLayerConfig dlc;
    struct screen_callback_t *screencbdata;

    int tcw[DFB_MAX_SCREENS];
    int tch[DFB_MAX_SCREENS];
    int i;
    DFBResult ret;

    SDL_DFB_ALLOC_CLEAR(screencbdata, sizeof(*screencbdata));

    screencbdata->numscreens = 0;

    for (i = 0; i < DFB_MAX_SCREENS; i++) {
        screencbdata->gralayer[i] = -1;
        screencbdata->vidlayer[i] = -1;
    }

    SDL_DFB_CHECKERR(devdata->dfb->EnumScreens(devdata->dfb, &EnumScreensCallback,
                                               screencbdata));

    for (i = 0; i < screencbdata->numscreens; i++) {
        IDirectFBScreen *screen;

        SDL_DFB_CHECKERR(devdata->dfb->GetScreen(devdata->dfb,
                                                 screencbdata->screenid
                                                 [i], &screen));

        screencbdata->aux = i;
        SDL_DFB_CHECKERR(screen->EnumDisplayLayers(screen, &EnumLayersCallback,
                                                   screencbdata));
        screen->GetSize(screen, &tcw[i], &tch[i]);

        screen->Release(screen);
    }

    /* Query card capabilities */

    devdata->dfb->GetDeviceDescription(devdata->dfb, &caps);

    for (i = 0; i < screencbdata->numscreens; i++) {
        SDL_DFB_CHECKERR(devdata->dfb->GetDisplayLayer(devdata->dfb,
                                                       screencbdata->gralayer
                                                       [i], &layer));

        SDL_DFB_CHECKERR(layer->SetCooperativeLevel(layer,
                                                    DLSCL_ADMINISTRATIVE));
        layer->EnableCursor(layer, 1);
        SDL_DFB_CHECKERR(layer->SetCursorOpacity(layer, 0xC0));

        if (devdata->use_yuv_underlays) {
            dlc.flags = DLCONF_PIXELFORMAT | DLCONF_OPTIONS;
            dlc.pixelformat = DSPF_ARGB;
            dlc.options = DLOP_ALPHACHANNEL;

            ret = layer->SetConfiguration(layer, &dlc);
            if (ret != DFB_OK) {
                /* try AiRGB if the previous failed */
                dlc.pixelformat = DSPF_AiRGB;
                SDL_DFB_CHECKERR(layer->SetConfiguration(layer, &dlc));
            }
        }

        /* Query layer configuration to determine the current mode and pixelformat */
        dlc.flags = DLCONF_ALL;
        SDL_DFB_CHECKERR(layer->GetConfiguration(layer, &dlc));

        mode.format = DirectFB_DFBToSDLPixelFormat(dlc.pixelformat);

        if (mode.format == SDL_PIXELFORMAT_UNKNOWN) {
            SDL_DFB_ERR("Unknown dfb pixelformat %x !\n", dlc.pixelformat);
            goto error;
        }

        mode.w = dlc.width;
        mode.h = dlc.height;
        mode.refresh_rate = 0;
        mode.driverdata = NULL;

        SDL_DFB_ALLOC_CLEAR(dispdata, sizeof(*dispdata));

        dispdata->layer = layer;
        dispdata->pixelformat = dlc.pixelformat;
        dispdata->cw = tcw[i];
        dispdata->ch = tch[i];

        /* YUV - Video layer */

        dispdata->vidID = screencbdata->vidlayer[i];
        dispdata->vidIDinuse = 0;

        SDL_zero(display);

        display.desktop_mode = mode;
        display.current_mode = mode;
        display.driverdata = dispdata;

#if (DFB_VERSION_ATLEAST(1,2,0))
        dlc.flags =
            DLCONF_WIDTH | DLCONF_HEIGHT | DLCONF_PIXELFORMAT |
            DLCONF_OPTIONS;
        ret = layer->SetConfiguration(layer, &dlc);
#endif

        SDL_DFB_CHECKERR(layer->SetCooperativeLevel(layer, DLSCL_SHARED));

        SDL_AddVideoDisplay(&display, SDL_FALSE);
    }
    SDL_DFB_FREE(screencbdata);
    return;
  error:
    /* FIXME: Cleanup not complete, Free existing displays */
    SDL_DFB_FREE(dispdata);
    SDL_DFB_RELEASE(layer);
    return;
}

void DirectFB_GetDisplayModes(_THIS, SDL_VideoDisplay * display)
{
    SDL_DFB_DEVICEDATA(_this);
    DFB_DisplayData *dispdata = (DFB_DisplayData *) display->driverdata;
    SDL_DisplayMode mode;
    struct modes_callback_t data;
    int i;

    data.nummodes = 0;
    /* Enumerate the available fullscreen modes */
    SDL_DFB_CALLOC(data.modelist, DFB_MAX_MODES, sizeof(SDL_DisplayMode));
    SDL_DFB_CHECKERR(devdata->dfb->EnumVideoModes(devdata->dfb,
                                                  EnumModesCallback, &data));

    for (i = 0; i < data.nummodes; ++i) {
        mode = data.modelist[i];

        mode.format = SDL_PIXELFORMAT_ARGB8888;
        CheckSetDisplayMode(_this, display, dispdata, &mode);
        mode.format = SDL_PIXELFORMAT_RGB888;
        CheckSetDisplayMode(_this, display, dispdata, &mode);
        mode.format = SDL_PIXELFORMAT_RGB24;
        CheckSetDisplayMode(_this, display, dispdata, &mode);
        mode.format = SDL_PIXELFORMAT_RGB565;
        CheckSetDisplayMode(_this, display, dispdata, &mode);
        mode.format = SDL_PIXELFORMAT_INDEX8;
        CheckSetDisplayMode(_this, display, dispdata, &mode);
    }

    SDL_DFB_FREE(data.modelist);
error:
    return;
}

int DirectFB_SetDisplayMode(_THIS, SDL_VideoDisplay * display, SDL_DisplayMode * mode)
{
    /*
     * FIXME: video mode switch is currently broken for 1.2.0
     *
     */

    SDL_DFB_DEVICEDATA(_this);
    DFB_DisplayData *data = (DFB_DisplayData *) display->driverdata;
    DFBDisplayLayerConfig config, rconfig;
    DFBDisplayLayerConfigFlags fail = 0;

    SDL_DFB_CHECKERR(data->layer->SetCooperativeLevel(data->layer,
                                                      DLSCL_ADMINISTRATIVE));

    SDL_DFB_CHECKERR(data->layer->GetConfiguration(data->layer, &config));
    config.flags = DLCONF_WIDTH | DLCONF_HEIGHT;
    if (mode->format != SDL_PIXELFORMAT_UNKNOWN) {
        config.flags |= DLCONF_PIXELFORMAT;
        config.pixelformat = DirectFB_SDLToDFBPixelFormat(mode->format);
        data->pixelformat = config.pixelformat;
    }
    config.width = mode->w;
    config.height = mode->h;

    if (devdata->use_yuv_underlays) {
        config.flags |= DLCONF_OPTIONS;
        config.options = DLOP_ALPHACHANNEL;
    }

    data->layer->TestConfiguration(data->layer, &config, &fail);

    if (fail &
        (DLCONF_WIDTH | DLCONF_HEIGHT | DLCONF_PIXELFORMAT |
         DLCONF_OPTIONS)) {
        SDL_DFB_ERR("Error setting mode %dx%d-%x\n", mode->w, mode->h,
                    mode->format);
        return -1;
    }

    config.flags &= ~fail;
    SDL_DFB_CHECKERR(data->layer->SetConfiguration(data->layer, &config));
#if (DFB_VERSION_ATLEAST(1,2,0))
    /* Need to call this twice ! */
    SDL_DFB_CHECKERR(data->layer->SetConfiguration(data->layer, &config));
#endif

    /* Double check */
    SDL_DFB_CHECKERR(data->layer->GetConfiguration(data->layer, &rconfig));
    SDL_DFB_CHECKERR(data->
                     layer->SetCooperativeLevel(data->layer, DLSCL_SHARED));

    if ((config.width != rconfig.width) || (config.height != rconfig.height)
        || ((mode->format != SDL_PIXELFORMAT_UNKNOWN)
            && (config.pixelformat != rconfig.pixelformat))) {
        SDL_DFB_ERR("Error setting mode %dx%d-%x\n", mode->w, mode->h,
                    mode->format);
        return -1;
    }

    data->pixelformat = rconfig.pixelformat;
    data->cw = config.width;
    data->ch = config.height;
    display->current_mode = *mode;

    return 0;
  error:
    return -1;
}

void DirectFB_QuitModes(_THIS)
{
    SDL_DisplayMode tmode;
    int i;

    for (i = 0; i < _this->num_displays; ++i) {
        SDL_VideoDisplay *display = &_this->displays[i];
        DFB_DisplayData *dispdata = (DFB_DisplayData *) display->driverdata;

        SDL_GetDesktopDisplayMode(i, &tmode);
        tmode.format = SDL_PIXELFORMAT_UNKNOWN;
        DirectFB_SetDisplayMode(_this, display, &tmode);

        SDL_GetDesktopDisplayMode(i, &tmode);
        DirectFB_SetDisplayMode(_this, display, &tmode);

        if (dispdata->layer) {
            SDL_DFB_CHECK(dispdata->
                          layer->SetCooperativeLevel(dispdata->layer,
                                                     DLSCL_ADMINISTRATIVE));
            SDL_DFB_CHECK(dispdata->
                          layer->SetCursorOpacity(dispdata->layer, 0x00));
            SDL_DFB_CHECK(dispdata->
                          layer->SetCooperativeLevel(dispdata->layer,
                                                     DLSCL_SHARED));
        }

        SDL_DFB_RELEASE(dispdata->layer);
        SDL_DFB_RELEASE(dispdata->vidlayer);

    }
}

#endif /* SDL_VIDEO_DRIVER_DIRECTFB */

/* vi: set ts=4 sw=4 expandtab: */
