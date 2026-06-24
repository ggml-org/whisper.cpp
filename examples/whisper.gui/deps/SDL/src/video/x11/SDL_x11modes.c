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

#ifdef SDL_VIDEO_DRIVER_X11

#include "SDL_hints.h"
#include "SDL_x11video.h"
#include "SDL_timer.h"
#include "edid.h"

/* #define X11MODES_DEBUG */

/* I'm becoming more and more convinced that the application should never
 * use XRandR, and it's the window manager's responsibility to track and
 * manage display modes for fullscreen windows.  Right now XRandR is completely
 * broken with respect to window manager behavior on every window manager that
 * I can find.  For example, on Unity 3D if you show a fullscreen window while
 * the resolution is changing (within ~250 ms) your window will retain the
 * fullscreen state hint but be decorated and windowed.
 *
 * However, many people swear by it, so let them swear at it. :)
 */
/* #define XRANDR_DISABLED_BY_DEFAULT */

static int get_visualinfo(Display *display, int screen, XVisualInfo *vinfo)
{
    const char *visual_id = SDL_getenv("SDL_VIDEO_X11_VISUALID");
    int depth;

    /* Look for an exact visual, if requested */
    if (visual_id) {
        XVisualInfo *vi, template;
        int nvis;

        SDL_zero(template);
        template.visualid = SDL_strtol(visual_id, NULL, 0);
        vi = X11_XGetVisualInfo(display, VisualIDMask, &template, &nvis);
        if (vi) {
            *vinfo = *vi;
            X11_XFree(vi);
            return 0;
        }
    }

    depth = DefaultDepth(display, screen);
    if ((X11_UseDirectColorVisuals() &&
         X11_XMatchVisualInfo(display, screen, depth, DirectColor, vinfo)) ||
        X11_XMatchVisualInfo(display, screen, depth, TrueColor, vinfo) ||
        X11_XMatchVisualInfo(display, screen, depth, PseudoColor, vinfo) ||
        X11_XMatchVisualInfo(display, screen, depth, StaticColor, vinfo)) {
        return 0;
    }
    return -1;
}

int X11_GetVisualInfoFromVisual(Display *display, Visual *visual, XVisualInfo *vinfo)
{
    XVisualInfo *vi;
    int nvis;

    vinfo->visualid = X11_XVisualIDFromVisual(visual);
    vi = X11_XGetVisualInfo(display, VisualIDMask, vinfo, &nvis);
    if (vi) {
        *vinfo = *vi;
        X11_XFree(vi);
        return 0;
    }
    return -1;
}

Uint32 X11_GetPixelFormatFromVisualInfo(Display *display, XVisualInfo *vinfo)
{
    if (vinfo->class == DirectColor || vinfo->class == TrueColor) {
        int bpp;
        Uint32 Rmask, Gmask, Bmask, Amask;

        Rmask = vinfo->visual->red_mask;
        Gmask = vinfo->visual->green_mask;
        Bmask = vinfo->visual->blue_mask;
        if (vinfo->depth == 32) {
            Amask = (0xFFFFFFFF & ~(Rmask | Gmask | Bmask));
        } else {
            Amask = 0;
        }

        bpp = vinfo->depth;
        if (bpp == 24) {
            int i, n;
            XPixmapFormatValues *p = X11_XListPixmapFormats(display, &n);
            if (p) {
                for (i = 0; i < n; ++i) {
                    if (p[i].depth == 24) {
                        bpp = p[i].bits_per_pixel;
                        break;
                    }
                }
                X11_XFree(p);
            }
        }

        return SDL_MasksToPixelFormatEnum(bpp, Rmask, Gmask, Bmask, Amask);
    }

    if (vinfo->class == PseudoColor || vinfo->class == StaticColor) {
        switch (vinfo->depth) {
        case 8:
            return SDL_PIXELFORMAT_INDEX8;
        case 4:
            if (BitmapBitOrder(display) == LSBFirst) {
                return SDL_PIXELFORMAT_INDEX4LSB;
            } else {
                return SDL_PIXELFORMAT_INDEX4MSB;
            }
            /* break; -Wunreachable-code-break */
        case 1:
            if (BitmapBitOrder(display) == LSBFirst) {
                return SDL_PIXELFORMAT_INDEX1LSB;
            } else {
                return SDL_PIXELFORMAT_INDEX1MSB;
            }
            /* break; -Wunreachable-code-break */
        }
    }

    return SDL_PIXELFORMAT_UNKNOWN;
}

#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
static SDL_bool CheckXRandR(Display *display, int *major, int *minor)
{
    /* Default the extension not available */
    *major = *minor = 0;

    /* Allow environment override */
#ifdef XRANDR_DISABLED_BY_DEFAULT
    if (!SDL_GetHintBoolean(SDL_HINT_VIDEO_X11_XRANDR, SDL_FALSE)) {
#ifdef X11MODES_DEBUG
        printf("XRandR disabled by default due to window manager issues\n");
#endif
        return SDL_FALSE;
    }
#else
    if (!SDL_GetHintBoolean(SDL_HINT_VIDEO_X11_XRANDR, SDL_TRUE)) {
#ifdef X11MODES_DEBUG
        printf("XRandR disabled due to hint\n");
#endif
        return SDL_FALSE;
    }
#endif /* XRANDR_ENABLED_BY_DEFAULT */

    if (!SDL_X11_HAVE_XRANDR) {
#ifdef X11MODES_DEBUG
        printf("XRandR support not available\n");
#endif
        return SDL_FALSE;
    }

    /* Query the extension version */
    *major = 1;
    *minor = 3; /* we want 1.3 */
    if (!X11_XRRQueryVersion(display, major, minor)) {
#ifdef X11MODES_DEBUG
        printf("XRandR not active on the display\n");
#endif
        *major = *minor = 0;
        return SDL_FALSE;
    }
#ifdef X11MODES_DEBUG
    printf("XRandR available at version %d.%d!\n", *major, *minor);
#endif
    return SDL_TRUE;
}

#define XRANDR_ROTATION_LEFT  (1 << 1)
#define XRANDR_ROTATION_RIGHT (1 << 3)

static int CalculateXRandRRefreshRate(const XRRModeInfo *info)
{
    return (info->hTotal && info->vTotal) ? SDL_round(((double)info->dotClock / (double)(info->hTotal * info->vTotal))) : 0;
}

static SDL_bool SetXRandRModeInfo(Display *display, XRRScreenResources *res, RRCrtc crtc,
                                  RRMode modeID, SDL_DisplayMode *mode)
{
    int i;
    for (i = 0; i < res->nmode; ++i) {
        const XRRModeInfo *info = &res->modes[i];
        if (info->id == modeID) {
            XRRCrtcInfo *crtcinfo;
            Rotation rotation = 0;

            crtcinfo = X11_XRRGetCrtcInfo(display, res, crtc);
            if (crtcinfo) {
                rotation = crtcinfo->rotation;
                X11_XRRFreeCrtcInfo(crtcinfo);
            }

            if (rotation & (XRANDR_ROTATION_LEFT | XRANDR_ROTATION_RIGHT)) {
                mode->w = info->height;
                mode->h = info->width;
            } else {
                mode->w = info->width;
                mode->h = info->height;
            }
            mode->refresh_rate = CalculateXRandRRefreshRate(info);
            ((SDL_DisplayModeData *)mode->driverdata)->xrandr_mode = modeID;
#ifdef X11MODES_DEBUG
            printf("XRandR mode %d: %dx%d@%dHz\n", (int)modeID, mode->w, mode->h, mode->refresh_rate);
#endif
            return SDL_TRUE;
        }
    }
    return SDL_FALSE;
}

static void SetXRandRDisplayName(Display *dpy, Atom EDID, char *name, const size_t namelen, RROutput output, const unsigned long widthmm, const unsigned long heightmm)
{
    /* See if we can get the EDID data for the real monitor name */
    int inches;
    int nprop;
    Atom *props = X11_XRRListOutputProperties(dpy, output, &nprop);
    int i;

    for (i = 0; i < nprop; ++i) {
        unsigned char *prop;
        int actual_format;
        unsigned long nitems, bytes_after;
        Atom actual_type;

        if (props[i] == EDID) {
            if (X11_XRRGetOutputProperty(dpy, output, props[i], 0, 100, False,
                                         False, AnyPropertyType, &actual_type,
                                         &actual_format, &nitems, &bytes_after,
                                         &prop) == Success) {
                MonitorInfo *info = decode_edid(prop);
                if (info) {
#ifdef X11MODES_DEBUG
                    printf("Found EDID data for %s\n", name);
                    dump_monitor_info(info);
#endif
                    SDL_strlcpy(name, info->dsc_product_name, namelen);
                    SDL_free(info);
                }
                X11_XFree(prop);
            }
            break;
        }
    }

    if (props) {
        X11_XFree(props);
    }

    inches = (int)((SDL_sqrtf(widthmm * widthmm + heightmm * heightmm) / 25.4f) + 0.5f);
    if (*name && inches) {
        const size_t len = SDL_strlen(name);
        (void)SDL_snprintf(&name[len], namelen - len, " %d\"", inches);
    }

#ifdef X11MODES_DEBUG
    printf("Display name: %s\n", name);
#endif
}

static int X11_AddXRandRDisplay(_THIS, Display *dpy, int screen, RROutput outputid, XRRScreenResources *res, SDL_bool send_event)
{
    Atom EDID = X11_XInternAtom(dpy, "EDID", False);
    XRROutputInfo *output_info;
    int display_x, display_y;
    unsigned long display_mm_width, display_mm_height;
    SDL_DisplayData *displaydata;
    char display_name[128];
    SDL_DisplayMode mode;
    SDL_DisplayModeData *modedata;
    SDL_VideoDisplay display;
    RRMode modeID;
    RRCrtc output_crtc;
    XRRCrtcInfo *crtc;
    XVisualInfo vinfo;
    Uint32 pixelformat;
    XPixmapFormatValues *pixmapformats;
    int scanline_pad;
    int i, n;

    if (get_visualinfo(dpy, screen, &vinfo) < 0) {
        return 0; /* uh, skip this screen? */
    }

    pixelformat = X11_GetPixelFormatFromVisualInfo(dpy, &vinfo);
    if (SDL_ISPIXELFORMAT_INDEXED(pixelformat)) {
        return 0; /* Palettized video modes are no longer supported, ignore this one. */
    }

    scanline_pad = SDL_BYTESPERPIXEL(pixelformat) * 8;
    pixmapformats = X11_XListPixmapFormats(dpy, &n);
    if (pixmapformats) {
        for (i = 0; i < n; i++) {
            if (pixmapformats[i].depth == vinfo.depth) {
                scanline_pad = pixmapformats[i].scanline_pad;
                break;
            }
        }
        X11_XFree(pixmapformats);
    }

    output_info = X11_XRRGetOutputInfo(dpy, res, outputid);
    if (!output_info || !output_info->crtc || output_info->connection == RR_Disconnected) {
        X11_XRRFreeOutputInfo(output_info);
        return 0; /* ignore this one. */
    }

    SDL_strlcpy(display_name, output_info->name, sizeof(display_name));
    display_mm_width = output_info->mm_width;
    display_mm_height = output_info->mm_height;
    output_crtc = output_info->crtc;
    X11_XRRFreeOutputInfo(output_info);

    crtc = X11_XRRGetCrtcInfo(dpy, res, output_crtc);
    if (!crtc) {
        return 0; /* oh well, ignore it. */
    }

    SDL_zero(mode);
    modeID = crtc->mode;
    mode.w = crtc->width;
    mode.h = crtc->height;
    mode.format = pixelformat;

    display_x = crtc->x;
    display_y = crtc->y;

    X11_XRRFreeCrtcInfo(crtc);

    displaydata = (SDL_DisplayData *)SDL_calloc(1, sizeof(*displaydata));
    if (!displaydata) {
        return SDL_OutOfMemory();
    }

    modedata = (SDL_DisplayModeData *)SDL_calloc(1, sizeof(SDL_DisplayModeData));
    if (!modedata) {
        SDL_free(displaydata);
        return SDL_OutOfMemory();
    }

    modedata->xrandr_mode = modeID;
    mode.driverdata = modedata;

    displaydata->screen = screen;
    displaydata->visual = vinfo.visual;
    displaydata->depth = vinfo.depth;
    displaydata->hdpi = display_mm_width ? (((float)mode.w) * 25.4f / display_mm_width) : 0.0f;
    displaydata->vdpi = display_mm_height ? (((float)mode.h) * 25.4f / display_mm_height) : 0.0f;
    displaydata->ddpi = SDL_ComputeDiagonalDPI(mode.w, mode.h, ((float)display_mm_width) / 25.4f, ((float)display_mm_height) / 25.4f);
    displaydata->scanline_pad = scanline_pad;
    displaydata->x = display_x;
    displaydata->y = display_y;
    displaydata->use_xrandr = SDL_TRUE;
    displaydata->xrandr_output = outputid;

    SetXRandRModeInfo(dpy, res, output_crtc, modeID, &mode);
    SetXRandRDisplayName(dpy, EDID, display_name, sizeof(display_name), outputid, display_mm_width, display_mm_height);

    SDL_zero(display);
    if (*display_name) {
        display.name = display_name;
    }
    display.desktop_mode = mode;
    display.current_mode = mode;
    display.driverdata = displaydata;
    return SDL_AddVideoDisplay(&display, send_event);
}

static void X11_HandleXRandROutputChange(_THIS, const XRROutputChangeNotifyEvent *ev)
{
    const int num_displays = SDL_GetNumVideoDisplays();
    SDL_VideoDisplay *display = NULL;
    int displayidx = -1;
    int i;

#if 0
    printf("XRROutputChangeNotifyEvent! [output=%u, crtc=%u, mode=%u, rotation=%u, connection=%u]", (unsigned int) ev->output, (unsigned int) ev->crtc, (unsigned int) ev->mode, (unsigned int) ev->rotation, (unsigned int) ev->connection);
#endif

    for (i = 0; i < num_displays; i++) {
        SDL_VideoDisplay *thisdisplay = SDL_GetDisplay(i);
        const SDL_DisplayData *displaydata = (const SDL_DisplayData *)thisdisplay->driverdata;
        if (displaydata->xrandr_output == ev->output) {
            display = thisdisplay;
            displayidx = i;
            break;
        }
    }

    SDL_assert((displayidx == -1) == (display == NULL));

    if (ev->connection == RR_Disconnected) { /* output is going away */
        if (display) {
            SDL_DelVideoDisplay(displayidx);
        }
    } else if (ev->connection == RR_Connected) { /* output is coming online */
        if (display) {
            /* !!! FIXME: update rotation or current mode of existing display? */
        } else {
            Display *dpy = ev->display;
            const int screen = DefaultScreen(dpy);
            XVisualInfo vinfo;
            if (get_visualinfo(dpy, screen, &vinfo) == 0) {
                XRRScreenResources *res = X11_XRRGetScreenResourcesCurrent(dpy, RootWindow(dpy, screen));
                if (!res || res->noutput == 0) {
                    if (res) {
                        X11_XRRFreeScreenResources(res);
                    }
                    res = X11_XRRGetScreenResources(dpy, RootWindow(dpy, screen));
                }

                if (res) {
                    X11_AddXRandRDisplay(_this, dpy, screen, ev->output, res, SDL_TRUE);
                    X11_XRRFreeScreenResources(res);
                }
            }
        }
    }
}

void X11_HandleXRandREvent(_THIS, const XEvent *xevent)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    SDL_assert(xevent->type == (videodata->xrandr_event_base + RRNotify));

    switch (((const XRRNotifyEvent *)xevent)->subtype) {
    case RRNotify_OutputChange:
        X11_HandleXRandROutputChange(_this, (const XRROutputChangeNotifyEvent *)xevent);
        break;
    default:
        break;
    }
}

static int X11_InitModes_XRandR(_THIS)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->driverdata;
    Display *dpy = data->display;
    const int screencount = ScreenCount(dpy);
    const int default_screen = DefaultScreen(dpy);
    RROutput primary = X11_XRRGetOutputPrimary(dpy, RootWindow(dpy, default_screen));
    XRRScreenResources *res = NULL;
    int xrandr_error_base = 0;
    int looking_for_primary;
    int output;
    int screen;

    if (!X11_XRRQueryExtension(dpy, &data->xrandr_event_base, &xrandr_error_base)) {
        return SDL_SetError("XRRQueryExtension failed");
    }

    for (looking_for_primary = 1; looking_for_primary >= 0; looking_for_primary--) {
        for (screen = 0; screen < screencount; screen++) {

            /* we want the primary output first, and then skipped later. */
            if (looking_for_primary && (screen != default_screen)) {
                continue;
            }

            res = X11_XRRGetScreenResourcesCurrent(dpy, RootWindow(dpy, screen));
            if (!res || res->noutput == 0) {
                if (res) {
                    X11_XRRFreeScreenResources(res);
                }

                res = X11_XRRGetScreenResources(dpy, RootWindow(dpy, screen));
                if (!res) {
                    continue;
                }
            }

            for (output = 0; output < res->noutput; output++) {
                /* The primary output _should_ always be sorted first, but just in case... */
                if ((looking_for_primary && (res->outputs[output] != primary)) ||
                    (!looking_for_primary && (screen == default_screen) && (res->outputs[output] == primary))) {
                    continue;
                }
                if (X11_AddXRandRDisplay(_this, dpy, screen, res->outputs[output], res, SDL_FALSE) == -1) {
                    break;
                }
            }

            X11_XRRFreeScreenResources(res);

            /* This will generate events for displays that come and go at runtime. */
            X11_XRRSelectInput(dpy, RootWindow(dpy, screen), RROutputChangeNotifyMask);
        }
    }

    if (_this->num_displays == 0) {
        return SDL_SetError("No available displays");
    }

    return 0;
}
#endif /* SDL_VIDEO_DRIVER_X11_XRANDR */

static int GetXftDPI(Display *dpy)
{
    char *xdefault_resource;
    int xft_dpi, err;

    xdefault_resource = X11_XGetDefault(dpy, "Xft", "dpi");

    if (!xdefault_resource) {
        return 0;
    }

    /*
     * It's possible for SDL_atoi to call SDL_strtol, if it fails due to a
     * overflow or an underflow, it will return LONG_MAX or LONG_MIN and set
     * errno to ERANGE. So we need to check for this so we dont get crazy dpi
     * values
     */
    xft_dpi = SDL_atoi(xdefault_resource);
    err = errno;

    return err == ERANGE ? 0 : xft_dpi;
}

/* This is used if there's no better functionality--like XRandR--to use.
   It won't attempt to supply different display modes at all, but it can
   enumerate the current displays and their current sizes. */
static int X11_InitModes_StdXlib(_THIS)
{
    /* !!! FIXME: a lot of copy/paste from X11_InitModes_XRandR in this function. */
    SDL_VideoData *data = (SDL_VideoData *)_this->driverdata;
    Display *dpy = data->display;
    const int default_screen = DefaultScreen(dpy);
    Screen *screen = ScreenOfDisplay(dpy, default_screen);
    int display_mm_width, display_mm_height, xft_dpi, scanline_pad, n, i;
    SDL_DisplayModeData *modedata;
    SDL_DisplayData *displaydata;
    SDL_DisplayMode mode;
    XPixmapFormatValues *pixmapformats;
    Uint32 pixelformat;
    XVisualInfo vinfo;
    SDL_VideoDisplay display;

    /* note that generally even if you have a multiple physical monitors, ScreenCount(dpy) still only reports ONE screen. */

    if (get_visualinfo(dpy, default_screen, &vinfo) < 0) {
        return SDL_SetError("Failed to find an X11 visual for the primary display");
    }

    pixelformat = X11_GetPixelFormatFromVisualInfo(dpy, &vinfo);
    if (SDL_ISPIXELFORMAT_INDEXED(pixelformat)) {
        return SDL_SetError("Palettized video modes are no longer supported");
    }

    SDL_zero(mode);
    mode.w = WidthOfScreen(screen);
    mode.h = HeightOfScreen(screen);
    mode.format = pixelformat;
    mode.refresh_rate = 0; /* don't know it, sorry. */

    displaydata = (SDL_DisplayData *)SDL_calloc(1, sizeof(*displaydata));
    if (!displaydata) {
        return SDL_OutOfMemory();
    }

    modedata = (SDL_DisplayModeData *)SDL_calloc(1, sizeof(SDL_DisplayModeData));
    if (!modedata) {
        SDL_free(displaydata);
        return SDL_OutOfMemory();
    }
    mode.driverdata = modedata;

    display_mm_width = WidthMMOfScreen(screen);
    display_mm_height = HeightMMOfScreen(screen);

    displaydata->screen = default_screen;
    displaydata->visual = vinfo.visual;
    displaydata->depth = vinfo.depth;
    displaydata->hdpi = display_mm_width ? (((float)mode.w) * 25.4f / display_mm_width) : 0.0f;
    displaydata->vdpi = display_mm_height ? (((float)mode.h) * 25.4f / display_mm_height) : 0.0f;
    displaydata->ddpi = SDL_ComputeDiagonalDPI(mode.w, mode.h, ((float)display_mm_width) / 25.4f, ((float)display_mm_height) / 25.4f);

    xft_dpi = GetXftDPI(dpy);
    if (xft_dpi > 0) {
        displaydata->hdpi = (float)xft_dpi;
        displaydata->vdpi = (float)xft_dpi;
    }

    scanline_pad = SDL_BYTESPERPIXEL(pixelformat) * 8;
    pixmapformats = X11_XListPixmapFormats(dpy, &n);
    if (pixmapformats) {
        for (i = 0; i < n; ++i) {
            if (pixmapformats[i].depth == vinfo.depth) {
                scanline_pad = pixmapformats[i].scanline_pad;
                break;
            }
        }
        X11_XFree(pixmapformats);
    }

    displaydata->scanline_pad = scanline_pad;
    displaydata->x = 0;
    displaydata->y = 0;
    displaydata->use_xrandr = SDL_FALSE;

    SDL_zero(display);
    display.name = (char *)"Generic X11 Display"; /* this is just copied and thrown away, it's safe to cast to char* here. */
    display.desktop_mode = mode;
    display.current_mode = mode;
    display.driverdata = displaydata;
    SDL_AddVideoDisplay(&display, SDL_TRUE);

    return 0;
}

int X11_InitModes(_THIS)
{
    /* XRandR is the One True Modern Way to do this on X11. If this
       fails, we just won't report any display modes except the current
       desktop size. */
#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
    {
        SDL_VideoData *data = (SDL_VideoData *)_this->driverdata;
        int xrandr_major, xrandr_minor;
        /* require at least XRandR v1.3 */
        if (CheckXRandR(data->display, &xrandr_major, &xrandr_minor) &&
            (xrandr_major >= 2 || (xrandr_major == 1 && xrandr_minor >= 3)) &&
            X11_InitModes_XRandR(_this) == 0) {
            return 0;
        }
    }
#endif /* SDL_VIDEO_DRIVER_X11_XRANDR */

    /* still here? Just set up an extremely basic display. */
    return X11_InitModes_StdXlib(_this);
}

void X11_GetDisplayModes(_THIS, SDL_VideoDisplay *sdl_display)
{
    SDL_DisplayData *data = (SDL_DisplayData *)sdl_display->driverdata;
    SDL_DisplayMode mode;

    /* Unfortunately X11 requires the window to be created with the correct
     * visual and depth ahead of time, but the SDL API allows you to create
     * a window before setting the fullscreen display mode.  This means that
     * we have to use the same format for all windows and all display modes.
     * (or support recreating the window with a new visual behind the scenes)
     */
    mode.format = sdl_display->current_mode.format;
    mode.driverdata = NULL;

#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
    if (data->use_xrandr) {
        Display *display = ((SDL_VideoData *)_this->driverdata)->display;
        XRRScreenResources *res;

        res = X11_XRRGetScreenResources(display, RootWindow(display, data->screen));
        if (res) {
            SDL_DisplayModeData *modedata;
            XRROutputInfo *output_info;
            int i;

            output_info = X11_XRRGetOutputInfo(display, res, data->xrandr_output);
            if (output_info && output_info->connection != RR_Disconnected) {
                for (i = 0; i < output_info->nmode; ++i) {
                    modedata = (SDL_DisplayModeData *)SDL_calloc(1, sizeof(SDL_DisplayModeData));
                    if (!modedata) {
                        continue;
                    }
                    mode.driverdata = modedata;

                    if (!SetXRandRModeInfo(display, res, output_info->crtc, output_info->modes[i], &mode) ||
                        !SDL_AddDisplayMode(sdl_display, &mode)) {
                        SDL_free(modedata);
                    }
                }
            }
            X11_XRRFreeOutputInfo(output_info);
            X11_XRRFreeScreenResources(res);
        }
        return;
    }
#endif /* SDL_VIDEO_DRIVER_X11_XRANDR */

    if (!data->use_xrandr) {
        SDL_DisplayModeData *modedata;
        /* Add the desktop mode */
        mode = sdl_display->desktop_mode;
        modedata = (SDL_DisplayModeData *)SDL_calloc(1, sizeof(SDL_DisplayModeData));
        if (modedata) {
            *modedata = *(SDL_DisplayModeData *)sdl_display->desktop_mode.driverdata;
        }
        mode.driverdata = modedata;
        if (!SDL_AddDisplayMode(sdl_display, &mode)) {
            SDL_free(modedata);
        }
    }
}

#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
/* This catches an error from XRRSetScreenSize, as a workaround for now. */
/* !!! FIXME: remove this later when we have a better solution. */
static int (*PreXRRSetScreenSizeErrorHandler)(Display *, XErrorEvent *) = NULL;
static int SDL_XRRSetScreenSizeErrHandler(Display *d, XErrorEvent *e)
{
    /* BadMatch: https://github.com/libsdl-org/SDL/issues/4561 */
    /* BadValue: https://github.com/libsdl-org/SDL/issues/4840 */
    if ((e->error_code == BadMatch) || (e->error_code == BadValue)) {
        return 0;
    }

    return PreXRRSetScreenSizeErrorHandler(d, e);
}
#endif

int X11_SetDisplayMode(_THIS, SDL_VideoDisplay *sdl_display, SDL_DisplayMode *mode)
{
    SDL_VideoData *viddata = (SDL_VideoData *)_this->driverdata;
    SDL_DisplayData *data = (SDL_DisplayData *)sdl_display->driverdata;

    viddata->last_mode_change_deadline = SDL_GetTicks() + (PENDING_FOCUS_TIME * 2);

#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
    if (data->use_xrandr) {
        Display *display = viddata->display;
        SDL_DisplayModeData *modedata = (SDL_DisplayModeData *)mode->driverdata;
        int mm_width, mm_height;
        XRRScreenResources *res;
        XRROutputInfo *output_info;
        XRRCrtcInfo *crtc;
        Status status;

        res = X11_XRRGetScreenResources(display, RootWindow(display, data->screen));
        if (!res) {
            return SDL_SetError("Couldn't get XRandR screen resources");
        }

        output_info = X11_XRRGetOutputInfo(display, res, data->xrandr_output);
        if (!output_info || output_info->connection == RR_Disconnected) {
            X11_XRRFreeScreenResources(res);
            return SDL_SetError("Couldn't get XRandR output info");
        }

        crtc = X11_XRRGetCrtcInfo(display, res, output_info->crtc);
        if (!crtc) {
            X11_XRRFreeOutputInfo(output_info);
            X11_XRRFreeScreenResources(res);
            return SDL_SetError("Couldn't get XRandR crtc info");
        }

        if (crtc->mode == modedata->xrandr_mode) {
#ifdef X11MODES_DEBUG
            printf("already in desired mode 0x%lx (%ux%u), nothing to do\n",
                   crtc->mode, crtc->width, crtc->height);
#endif
            status = Success;
            goto freeInfo;
        }

        X11_XGrabServer(display);
        status = X11_XRRSetCrtcConfig(display, res, output_info->crtc, CurrentTime,
                                      0, 0, None, crtc->rotation, NULL, 0);
        if (status != Success) {
            goto ungrabServer;
        }

        mm_width = mode->w * DisplayWidthMM(display, data->screen) / DisplayWidth(display, data->screen);
        mm_height = mode->h * DisplayHeightMM(display, data->screen) / DisplayHeight(display, data->screen);

        /* !!! FIXME: this can get into a problem scenario when a window is
           bigger than a physical monitor in a configuration where one screen
           spans multiple physical monitors. A detailed reproduction case is
           discussed at https://github.com/libsdl-org/SDL/issues/4561 ...
           for now we cheat and just catch the X11 error and carry on, which
           is likely to cause subtle issues but is better than outright
           crashing */
        X11_XSync(display, False);
        PreXRRSetScreenSizeErrorHandler = X11_XSetErrorHandler(SDL_XRRSetScreenSizeErrHandler);
        X11_XRRSetScreenSize(display, RootWindow(display, data->screen), mode->w, mode->h, mm_width, mm_height);
        X11_XSync(display, False);
        X11_XSetErrorHandler(PreXRRSetScreenSizeErrorHandler);

        status = X11_XRRSetCrtcConfig(display, res, output_info->crtc, CurrentTime,
                                      crtc->x, crtc->y, modedata->xrandr_mode, crtc->rotation,
                                      &data->xrandr_output, 1);

    ungrabServer:
        X11_XUngrabServer(display);
    freeInfo:
        X11_XRRFreeCrtcInfo(crtc);
        X11_XRRFreeOutputInfo(output_info);
        X11_XRRFreeScreenResources(res);

        if (status != Success) {
            return SDL_SetError("X11_XRRSetCrtcConfig failed");
        }
    }
#else
    (void)data;
#endif /* SDL_VIDEO_DRIVER_X11_XRANDR */

    return 0;
}

void X11_QuitModes(_THIS)
{
}

int X11_GetDisplayBounds(_THIS, SDL_VideoDisplay *sdl_display, SDL_Rect *rect)
{
    SDL_DisplayData *data = (SDL_DisplayData *)sdl_display->driverdata;

    rect->x = data->x;
    rect->y = data->y;
    rect->w = sdl_display->current_mode.w;
    rect->h = sdl_display->current_mode.h;

    return 0;
}

int X11_GetDisplayDPI(_THIS, SDL_VideoDisplay *sdl_display, float *ddpi, float *hdpi, float *vdpi)
{
    SDL_DisplayData *data = (SDL_DisplayData *)sdl_display->driverdata;

    if (ddpi) {
        *ddpi = data->ddpi;
    }
    if (hdpi) {
        *hdpi = data->hdpi;
    }
    if (vdpi) {
        *vdpi = data->vdpi;
    }

    return data->ddpi != 0.0f ? 0 : SDL_SetError("Couldn't get DPI");
}

int X11_GetDisplayUsableBounds(_THIS, SDL_VideoDisplay *sdl_display, SDL_Rect *rect)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->driverdata;
    Display *display = data->display;
    Atom _NET_WORKAREA;
    int status, real_format;
    int retval = -1;
    Atom real_type;
    unsigned long items_read = 0, items_left = 0;
    unsigned char *propdata = NULL;

    if (X11_GetDisplayBounds(_this, sdl_display, rect) < 0) {
        return -1;
    }

    _NET_WORKAREA = X11_XInternAtom(display, "_NET_WORKAREA", False);
    status = X11_XGetWindowProperty(display, DefaultRootWindow(display),
                                    _NET_WORKAREA, 0L, 4L, False, XA_CARDINAL,
                                    &real_type, &real_format, &items_read,
                                    &items_left, &propdata);
    if ((status == Success) && (items_read >= 4)) {
        const long *p = (long *)propdata;
        const SDL_Rect usable = { (int)p[0], (int)p[1], (int)p[2], (int)p[3] };
        retval = 0;
        if (!SDL_IntersectRect(rect, &usable, rect)) {
            SDL_zerop(rect);
        }
    }

    if (propdata) {
        X11_XFree(propdata);
    }

    return retval;
}

#endif /* SDL_VIDEO_DRIVER_X11 */

/* vi: set ts=4 sw=4 expandtab: */
