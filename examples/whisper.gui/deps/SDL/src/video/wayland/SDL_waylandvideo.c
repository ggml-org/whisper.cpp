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

#ifdef SDL_VIDEO_DRIVER_WAYLAND

#include "SDL_video.h"
#include "SDL_mouse.h"
#include "SDL_stdinc.h"
#include "../../events/SDL_events_c.h"

#include "SDL_waylandvideo.h"
#include "SDL_waylandevents_c.h"
#include "SDL_waylandwindow.h"
#include "SDL_waylandopengles.h"
#include "SDL_waylandmouse.h"
#include "SDL_waylandkeyboard.h"
#include "SDL_waylandtouch.h"
#include "SDL_waylandclipboard.h"
#include "SDL_waylandvulkan.h"
#include "SDL_waylandmessagebox.h"
#include "SDL_hints.h"

#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <xkbcommon/xkbcommon.h>

#include <wayland-util.h>

#include "xdg-shell-client-protocol.h"
#include "xdg-decoration-unstable-v1-client-protocol.h"
#include "keyboard-shortcuts-inhibit-unstable-v1-client-protocol.h"
#include "idle-inhibit-unstable-v1-client-protocol.h"
#include "xdg-activation-v1-client-protocol.h"
#include "text-input-unstable-v3-client-protocol.h"
#include "tablet-unstable-v2-client-protocol.h"
#include "xdg-output-unstable-v1-client-protocol.h"
#include "viewporter-client-protocol.h"
#include "primary-selection-unstable-v1-client-protocol.h"
#include "fractional-scale-v1-client-protocol.h"

#ifdef HAVE_LIBDECOR_H
#include <libdecor.h>
#endif

#define WAYLANDVID_DRIVER_NAME "wayland"

static void display_handle_done(void *data, struct wl_output *output);

/* Initialization/Query functions */
static int Wayland_VideoInit(_THIS);

static int Wayland_GetDisplayBounds(_THIS, SDL_VideoDisplay *display, SDL_Rect *rect);

static int Wayland_GetDisplayDPI(_THIS, SDL_VideoDisplay *sdl_display, float *ddpi, float *hdpi, float *vdpi);

static void Wayland_VideoQuit(_THIS);

/* Find out what class name we should use
 * Based on src/video/x11/SDL_x11video.c */
static char *get_classname(void)
{
    /* !!! FIXME: this is probably wrong, albeit harmless in many common cases. From protocol spec:
        "The surface class identifies the general class of applications
        to which the surface belongs. A common convention is to use the
        file name (or the full path if it is a non-standard location) of
        the application's .desktop file as the class." */

    char *spot;
#if defined(__LINUX__) || defined(__FREEBSD__)
    char procfile[1024];
    char linkfile[1024];
    int linksize;
#endif

    /* First allow environment variable override */
    spot = SDL_getenv("SDL_VIDEO_WAYLAND_WMCLASS");
    if (spot) {
        return SDL_strdup(spot);
    } else {
        /* Fallback to the "old" envvar */
        spot = SDL_getenv("SDL_VIDEO_X11_WMCLASS");
        if (spot) {
            return SDL_strdup(spot);
        }
    }

    /* Next look at the application's executable name */
#if defined(__LINUX__) || defined(__FREEBSD__)
#if defined(__LINUX__)
    (void)SDL_snprintf(procfile, SDL_arraysize(procfile), "/proc/%d/exe", getpid());
#elif defined(__FREEBSD__)
    (void)SDL_snprintf(procfile, SDL_arraysize(procfile), "/proc/%d/file", getpid());
#else
#error Where can we find the executable name?
#endif
    linksize = readlink(procfile, linkfile, sizeof(linkfile) - 1);
    if (linksize > 0) {
        linkfile[linksize] = '\0';
        spot = SDL_strrchr(linkfile, '/');
        if (spot) {
            return SDL_strdup(spot + 1);
        } else {
            return SDL_strdup(linkfile);
        }
    }
#endif /* __LINUX__ || __FREEBSD__ */

    /* Finally use the default we've used forever */
    return SDL_strdup("SDL_App");
}

static const char *SDL_WAYLAND_surface_tag = "sdl-window";
static const char *SDL_WAYLAND_output_tag = "sdl-output";

void SDL_WAYLAND_register_surface(struct wl_surface *surface)
{
    wl_proxy_set_tag((struct wl_proxy *)surface, &SDL_WAYLAND_surface_tag);
}

void SDL_WAYLAND_register_output(struct wl_output *output)
{
    wl_proxy_set_tag((struct wl_proxy *)output, &SDL_WAYLAND_output_tag);
}

SDL_bool SDL_WAYLAND_own_surface(struct wl_surface *surface)
{
    return wl_proxy_get_tag((struct wl_proxy *)surface) == &SDL_WAYLAND_surface_tag;
}

SDL_bool SDL_WAYLAND_own_output(struct wl_output *output)
{
    return wl_proxy_get_tag((struct wl_proxy *)output) == &SDL_WAYLAND_output_tag;
}

static void Wayland_DeleteDevice(SDL_VideoDevice *device)
{
    SDL_VideoData *data = (SDL_VideoData *)device->driverdata;
    if (data->display) {
        WAYLAND_wl_display_flush(data->display);
        WAYLAND_wl_display_disconnect(data->display);
    }
    if (device->wakeup_lock) {
        SDL_DestroyMutex(device->wakeup_lock);
    }
    SDL_free(data);
    SDL_free(device);
    SDL_WAYLAND_UnloadSymbols();
}

static SDL_VideoDevice *Wayland_CreateDevice(void)
{
    SDL_VideoDevice *device;
    SDL_VideoData *data;
    struct SDL_WaylandInput *input;
    struct wl_display *display;

    /* Are we trying to connect to or are currently in a Wayland session? */
    if (!getenv("WAYLAND_DISPLAY")) {
        const char *session = getenv("XDG_SESSION_TYPE");
        if (session && SDL_strcasecmp(session, "wayland")) {
            return NULL;
        }
    }

    if (!SDL_WAYLAND_LoadSymbols()) {
        return NULL;
    }

    display = WAYLAND_wl_display_connect(NULL);
    if (!display) {
        SDL_WAYLAND_UnloadSymbols();
        return NULL;
    }

    data = SDL_calloc(1, sizeof(*data));
    if (!data) {
        WAYLAND_wl_display_disconnect(display);
        SDL_WAYLAND_UnloadSymbols();
        SDL_OutOfMemory();
        return NULL;
    }

    input = SDL_calloc(1, sizeof(*input));
    if (!input) {
        SDL_free(data);
        WAYLAND_wl_display_disconnect(display);
        SDL_WAYLAND_UnloadSymbols();
        SDL_OutOfMemory();
        return NULL;
    }

    input->display = data;
    input->sx_w = wl_fixed_from_int(0);
    input->sy_w = wl_fixed_from_int(0);
    input->xkb.current_group = XKB_GROUP_INVALID;

    data->initializing = SDL_TRUE;
    data->display = display;
    data->input = input;

    /* Initialize all variables that we clean on shutdown */
    device = SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        SDL_free(data);
        SDL_free(input);
        WAYLAND_wl_display_disconnect(display);
        SDL_WAYLAND_UnloadSymbols();
        SDL_OutOfMemory();
        return NULL;
    }

    device->driverdata = data;
    device->wakeup_lock = SDL_CreateMutex();

    /* Set the function pointers */
    device->VideoInit = Wayland_VideoInit;
    device->VideoQuit = Wayland_VideoQuit;
    device->GetDisplayBounds = Wayland_GetDisplayBounds;
    device->GetDisplayDPI = Wayland_GetDisplayDPI;
    device->GetWindowWMInfo = Wayland_GetWindowWMInfo;
    device->SuspendScreenSaver = Wayland_SuspendScreenSaver;

    device->PumpEvents = Wayland_PumpEvents;
    device->WaitEventTimeout = Wayland_WaitEventTimeout;
    device->SendWakeupEvent = Wayland_SendWakeupEvent;

#ifdef SDL_VIDEO_OPENGL_EGL
    device->GL_SwapWindow = Wayland_GLES_SwapWindow;
    device->GL_GetSwapInterval = Wayland_GLES_GetSwapInterval;
    device->GL_SetSwapInterval = Wayland_GLES_SetSwapInterval;
    device->GL_MakeCurrent = Wayland_GLES_MakeCurrent;
    device->GL_CreateContext = Wayland_GLES_CreateContext;
    device->GL_LoadLibrary = Wayland_GLES_LoadLibrary;
    device->GL_UnloadLibrary = Wayland_GLES_UnloadLibrary;
    device->GL_GetProcAddress = Wayland_GLES_GetProcAddress;
    device->GL_DeleteContext = Wayland_GLES_DeleteContext;
#endif

    device->CreateSDLWindow = Wayland_CreateWindow;
    device->ShowWindow = Wayland_ShowWindow;
    device->HideWindow = Wayland_HideWindow;
    device->RaiseWindow = Wayland_RaiseWindow;
    device->SetWindowFullscreen = Wayland_SetWindowFullscreen;
    device->MaximizeWindow = Wayland_MaximizeWindow;
    device->MinimizeWindow = Wayland_MinimizeWindow;
    device->SetWindowMouseRect = Wayland_SetWindowMouseRect;
    device->SetWindowMouseGrab = Wayland_SetWindowMouseGrab;
    device->SetWindowKeyboardGrab = Wayland_SetWindowKeyboardGrab;
    device->RestoreWindow = Wayland_RestoreWindow;
    device->SetWindowBordered = Wayland_SetWindowBordered;
    device->SetWindowResizable = Wayland_SetWindowResizable;
    device->SetWindowSize = Wayland_SetWindowSize;
    device->SetWindowMinimumSize = Wayland_SetWindowMinimumSize;
    device->SetWindowMaximumSize = Wayland_SetWindowMaximumSize;
    device->SetWindowModalFor = Wayland_SetWindowModalFor;
    device->SetWindowTitle = Wayland_SetWindowTitle;
    device->GetWindowSizeInPixels = Wayland_GetWindowSizeInPixels;
    device->DestroyWindow = Wayland_DestroyWindow;
    device->SetWindowHitTest = Wayland_SetWindowHitTest;
    device->FlashWindow = Wayland_FlashWindow;
    device->HasScreenKeyboardSupport = Wayland_HasScreenKeyboardSupport;

    device->SetClipboardText = Wayland_SetClipboardText;
    device->GetClipboardText = Wayland_GetClipboardText;
    device->HasClipboardText = Wayland_HasClipboardText;
    device->SetPrimarySelectionText = Wayland_SetPrimarySelectionText;
    device->GetPrimarySelectionText = Wayland_GetPrimarySelectionText;
    device->HasPrimarySelectionText = Wayland_HasPrimarySelectionText;
    device->StartTextInput = Wayland_StartTextInput;
    device->StopTextInput = Wayland_StopTextInput;
    device->SetTextInputRect = Wayland_SetTextInputRect;

#ifdef SDL_VIDEO_VULKAN
    device->Vulkan_LoadLibrary = Wayland_Vulkan_LoadLibrary;
    device->Vulkan_UnloadLibrary = Wayland_Vulkan_UnloadLibrary;
    device->Vulkan_GetInstanceExtensions = Wayland_Vulkan_GetInstanceExtensions;
    device->Vulkan_CreateSurface = Wayland_Vulkan_CreateSurface;
#endif

    device->free = Wayland_DeleteDevice;

    device->quirk_flags = VIDEO_DEVICE_QUIRK_DISABLE_DISPLAY_MODE_SWITCHING |
                          VIDEO_DEVICE_QUIRK_DISABLE_UNSET_FULLSCREEN_ON_MINIMIZE;

    return device;
}

VideoBootStrap Wayland_bootstrap = {
    WAYLANDVID_DRIVER_NAME, "SDL Wayland video driver",
    Wayland_CreateDevice,
    Wayland_ShowMessageBox
};

static void xdg_output_handle_logical_position(void *data, struct zxdg_output_v1 *xdg_output,
                                               int32_t x, int32_t y)
{
    SDL_WaylandOutputData *driverdata = data;

    driverdata->x = x;
    driverdata->y = y;
    driverdata->has_logical_position = SDL_TRUE;
}

static void xdg_output_handle_logical_size(void *data, struct zxdg_output_v1 *xdg_output,
                                           int32_t width, int32_t height)
{
    SDL_WaylandOutputData *driverdata = data;

    if (driverdata->width != 0 && driverdata->height != 0) {
        /* FIXME: GNOME has a bug where the logical size does not account for
         * scale, resulting in bogus viewport sizes.
         *
         * Until this is fixed, validate that _some_ kind of scaling is being
         * done (we can't match exactly because fractional scaling can't be
         * detected otherwise), then override if necessary.
         * -flibit
         */
        const float scale = (float)driverdata->width / (float)width;
        if ((scale == 1.0f) && (driverdata->scale_factor != 1.0f)) {
            SDL_LogWarn(
                SDL_LOG_CATEGORY_VIDEO,
                "xdg_output scale did not match, overriding with wl_output scale");
            return;
        }
    }

    driverdata->width = width;
    driverdata->height = height;
    driverdata->has_logical_size = SDL_TRUE;
}

static void xdg_output_handle_done(void *data, struct zxdg_output_v1 *xdg_output)
{
    SDL_WaylandOutputData *driverdata = data;

    /*
     * xdg-output.done events are deprecated and only apply below version 3 of the protocol.
     * A wl-output.done event will be emitted in version 3 or higher.
     */
    if (zxdg_output_v1_get_version(driverdata->xdg_output) < 3) {
        display_handle_done(data, driverdata->output);
    }
}

static void xdg_output_handle_name(void *data, struct zxdg_output_v1 *xdg_output,
                                   const char *name)
{
}

static void xdg_output_handle_description(void *data, struct zxdg_output_v1 *xdg_output,
                                          const char *description)
{
    SDL_WaylandOutputData *driverdata = data;

    if (driverdata->index == -1) {
        /* xdg-output descriptions, if available, supersede wl-output model names. */
        if (driverdata->placeholder.name) {
            SDL_free(driverdata->placeholder.name);
        }

        driverdata->placeholder.name = SDL_strdup(description);
    }
}

static const struct zxdg_output_v1_listener xdg_output_listener = {
    xdg_output_handle_logical_position,
    xdg_output_handle_logical_size,
    xdg_output_handle_done,
    xdg_output_handle_name,
    xdg_output_handle_description,
};

static void AddEmulatedModes(SDL_VideoDisplay *dpy, SDL_bool rot_90)
{
    struct EmulatedMode
    {
        int w;
        int h;
    };

    /* Resolution lists courtesy of XWayland */
    const struct EmulatedMode mode_list[] = {
        /* 16:9 (1.77) */
        { 7680, 4320 },
        { 6144, 3160 },
        { 5120, 2880 },
        { 4096, 2304 },
        { 3840, 2160 },
        { 3200, 1800 },
        { 2880, 1620 },
        { 2560, 1440 },
        { 2048, 1152 },
        { 1920, 1080 },
        { 1600, 900 },
        { 1368, 768 },
        { 1280, 720 },
        { 864, 486 },

        /* 16:10 (1.6) */
        { 2560, 1600 },
        { 1920, 1200 },
        { 1680, 1050 },
        { 1440, 900 },
        { 1280, 800 },

        /* 3:2 (1.5) */
        { 720, 480 },

        /* 4:3 (1.33) */
        { 2048, 1536 },
        { 1920, 1440 },
        { 1600, 1200 },
        { 1440, 1080 },
        { 1400, 1050 },
        { 1280, 1024 },
        { 1280, 960 },
        { 1152, 864 },
        { 1024, 768 },
        { 800, 600 },
        { 640, 480 }
    };

    int i;
    SDL_DisplayMode mode;
    const int native_width = dpy->display_modes->w;
    const int native_height = dpy->display_modes->h;

    for (i = 0; i < SDL_arraysize(mode_list); ++i) {
        mode = *dpy->display_modes;

        if (rot_90) {
            mode.w = mode_list[i].h;
            mode.h = mode_list[i].w;
        } else {
            mode.w = mode_list[i].w;
            mode.h = mode_list[i].h;
        }

        /* Only add modes that are smaller than the native mode. */
        if ((mode.w < native_width && mode.h < native_height) ||
            (mode.w < native_width && mode.h == native_height) ||
            (mode.w == native_width && mode.h < native_height)) {
            SDL_AddDisplayMode(dpy, &mode);
        }
    }
}

static void display_handle_geometry(void *data,
                                    struct wl_output *output,
                                    int x, int y,
                                    int physical_width,
                                    int physical_height,
                                    int subpixel,
                                    const char *make,
                                    const char *model,
                                    int transform)

{
    SDL_WaylandOutputData *driverdata = data;

    if (driverdata->wl_output_done_count) {
        SDL_ResetDisplayModes(driverdata->index);

        /* The display has officially started over. */
        driverdata->wl_output_done_count = 0;
    }

    /* Apply the change from wl-output only if xdg-output is not supported */
    if (!driverdata->has_logical_position) {
        driverdata->x = x;
        driverdata->y = y;
    }
    driverdata->physical_width = physical_width;
    driverdata->physical_height = physical_height;

    /* The output name is only set if xdg-output hasn't provided a description. */
    if (driverdata->index == -1 && !driverdata->placeholder.name) {
        driverdata->placeholder.name = SDL_strdup(model);
    }

    driverdata->transform = transform;
#define TF_CASE(in, out)                                 \
    case WL_OUTPUT_TRANSFORM_##in:                       \
        driverdata->orientation = SDL_ORIENTATION_##out; \
        break;
    if (driverdata->physical_width >= driverdata->physical_height) {
        switch (transform) {
            TF_CASE(NORMAL, LANDSCAPE)
            TF_CASE(90, PORTRAIT)
            TF_CASE(180, LANDSCAPE_FLIPPED)
            TF_CASE(270, PORTRAIT_FLIPPED)
            TF_CASE(FLIPPED, LANDSCAPE_FLIPPED)
            TF_CASE(FLIPPED_90, PORTRAIT_FLIPPED)
            TF_CASE(FLIPPED_180, LANDSCAPE)
            TF_CASE(FLIPPED_270, PORTRAIT)
        }
    } else {
        switch (transform) {
            TF_CASE(NORMAL, PORTRAIT)
            TF_CASE(90, LANDSCAPE)
            TF_CASE(180, PORTRAIT_FLIPPED)
            TF_CASE(270, LANDSCAPE_FLIPPED)
            TF_CASE(FLIPPED, PORTRAIT_FLIPPED)
            TF_CASE(FLIPPED_90, LANDSCAPE_FLIPPED)
            TF_CASE(FLIPPED_180, PORTRAIT)
            TF_CASE(FLIPPED_270, LANDSCAPE)
        }
    }
#undef TF_CASE
}

static void display_handle_mode(void *data,
                                struct wl_output *output,
                                uint32_t flags,
                                int width,
                                int height,
                                int refresh)
{
    SDL_WaylandOutputData *driverdata = data;

    if (flags & WL_OUTPUT_MODE_CURRENT) {
        driverdata->native_width = width;
        driverdata->native_height = height;

        /*
         * Don't rotate this yet, wl-output coordinates are transformed in
         * handle_done and xdg-output coordinates are pre-transformed.
         */
        if (!driverdata->has_logical_size) {
            driverdata->width = width;
            driverdata->height = height;
        }

        driverdata->refresh = refresh;
    }
}

static void display_handle_done(void *data,
                                struct wl_output *output)
{
    SDL_WaylandOutputData *driverdata = data;
    SDL_VideoData *video = driverdata->videodata;
    SDL_DisplayMode native_mode, desktop_mode;
    SDL_VideoDisplay *dpy;
    const SDL_bool mode_emulation_enabled = SDL_GetHintBoolean(SDL_HINT_VIDEO_WAYLAND_MODE_EMULATION, SDL_TRUE);

    /*
     * When using xdg-output, two wl-output.done events will be emitted:
     * one at the completion of wl-display and one at the completion of xdg-output.
     *
     * All required events must be received before proceeding.
     */
    const int event_await_count = 1 + (driverdata->xdg_output != NULL);

    driverdata->wl_output_done_count = SDL_min(driverdata->wl_output_done_count + 1, event_await_count + 1);

    if (driverdata->wl_output_done_count != event_await_count) {
        return;
    }

    /* The native display resolution */
    SDL_zero(native_mode);
    native_mode.format = SDL_PIXELFORMAT_RGB888;

    if (driverdata->transform & WL_OUTPUT_TRANSFORM_90) {
        native_mode.w = driverdata->native_height;
        native_mode.h = driverdata->native_width;
    } else {
        native_mode.w = driverdata->native_width;
        native_mode.h = driverdata->native_height;
    }
    native_mode.refresh_rate = (int)SDL_round(driverdata->refresh / 1000.0); /* mHz to Hz */

    /* The scaled desktop mode */
    SDL_zero(desktop_mode);
    desktop_mode.format = SDL_PIXELFORMAT_RGB888;

    if (driverdata->has_logical_size) { /* If xdg-output is present, calculate the true scale of the desktop */
        driverdata->scale_factor = (float)native_mode.w / (float)driverdata->width;
    } else { /* Scale the desktop coordinates, if xdg-output isn't present */
        driverdata->width /= driverdata->scale_factor;
        driverdata->height /= driverdata->scale_factor;
    }

    /* xdg-output dimensions are already transformed, so no need to rotate. */
    if (driverdata->has_logical_size || !(driverdata->transform & WL_OUTPUT_TRANSFORM_90)) {
        desktop_mode.w = driverdata->width;
        desktop_mode.h = driverdata->height;
    } else {
        desktop_mode.w = driverdata->height;
        desktop_mode.h = driverdata->width;
    }
    desktop_mode.refresh_rate = (int)SDL_round(driverdata->refresh / 1000.0); /* mHz to Hz */

    /*
     * The native display mode is only exposed separately from the desktop size if the
     * desktop is scaled and the wp_viewporter protocol is supported.
     */
    if (driverdata->scale_factor > 1.0f && video->viewporter) {
        if (driverdata->index > -1) {
            SDL_AddDisplayMode(SDL_GetDisplay(driverdata->index), &native_mode);
        } else {
            SDL_AddDisplayMode(&driverdata->placeholder, &native_mode);
        }
    }

    /* Calculate the display DPI */
    if (driverdata->transform & WL_OUTPUT_TRANSFORM_90) {
        driverdata->hdpi = driverdata->physical_height ? (((float)driverdata->native_height) * 25.4f / driverdata->physical_height) : 0.0f;
        driverdata->vdpi = driverdata->physical_width ? (((float)driverdata->native_width) * 25.4f / driverdata->physical_width) : 0.0f;
        driverdata->ddpi = SDL_ComputeDiagonalDPI(driverdata->native_height,
                                                  driverdata->native_width,
                                                  ((float)driverdata->physical_height) / 25.4f,
                                                  ((float)driverdata->physical_width) / 25.4f);
    } else {
        driverdata->hdpi = driverdata->physical_width ? (((float)driverdata->native_width) * 25.4f / driverdata->physical_width) : 0.0f;
        driverdata->vdpi = driverdata->physical_height ? (((float)driverdata->native_height) * 25.4f / driverdata->physical_height) : 0.0f;
        driverdata->ddpi = SDL_ComputeDiagonalDPI(driverdata->native_width,
                                                  driverdata->native_height,
                                                  ((float)driverdata->physical_width) / 25.4f,
                                                  ((float)driverdata->physical_height) / 25.4f);
    }

    if (driverdata->index > -1) {
        dpy = SDL_GetDisplay(driverdata->index);

        /* XXX: This can never happen, but jump threading during aggressive LTO can generate a warning without this check. */
        if (!dpy) {
            dpy = &driverdata->placeholder;
        }
    } else {
        dpy = &driverdata->placeholder;
    }

    SDL_AddDisplayMode(dpy, &desktop_mode);
    SDL_SetCurrentDisplayMode(dpy, &desktop_mode);
    SDL_SetDesktopDisplayMode(dpy, &desktop_mode);

    /* Add emulated modes if wp_viewporter is supported and mode emulation is enabled. */
    if (video->viewporter && mode_emulation_enabled) {
        const SDL_bool rot_90 = (driverdata->transform & WL_OUTPUT_TRANSFORM_90) ||
                                (driverdata->width < driverdata->height);
        AddEmulatedModes(dpy, rot_90);
    }

    if (driverdata->index == -1) {
        /* First time getting display info, create the VideoDisplay */
        SDL_bool send_event = driverdata->videodata->initializing ? SDL_FALSE : SDL_TRUE;
        driverdata->placeholder.orientation = driverdata->orientation;
        driverdata->placeholder.driverdata = driverdata;
        driverdata->index = SDL_AddVideoDisplay(&driverdata->placeholder, send_event);
        SDL_free(driverdata->placeholder.name);
        SDL_zero(driverdata->placeholder);
    } else {
        SDL_SendDisplayEvent(dpy, SDL_DISPLAYEVENT_ORIENTATION, driverdata->orientation);
    }
}

static void display_handle_scale(void *data,
                                 struct wl_output *output,
                                 int32_t factor)
{
    SDL_WaylandOutputData *driverdata = data;
    driverdata->scale_factor = factor;
}

static const struct wl_output_listener output_listener = {
    display_handle_geometry,
    display_handle_mode,
    display_handle_done,
    display_handle_scale
};

static void Wayland_add_display(SDL_VideoData *d, uint32_t id)
{
    struct wl_output *output;
    SDL_WaylandOutputData *data;

    output = wl_registry_bind(d->registry, id, &wl_output_interface, 2);
    if (!output) {
        SDL_SetError("Failed to retrieve output.");
        return;
    }
    data = (SDL_WaylandOutputData *)SDL_malloc(sizeof(*data));
    SDL_zerop(data);
    data->videodata = d;
    data->output = output;
    data->registry_id = id;
    data->scale_factor = 1.0f;
    data->index = -1;

    wl_output_add_listener(output, &output_listener, data);
    SDL_WAYLAND_register_output(output);

    /* Keep a list of outputs for deferred xdg-output initialization. */
    if (d->output_list) {
        SDL_WaylandOutputData *node = d->output_list;

        while (node->next) {
            node = node->next;
        }

        node->next = (struct SDL_WaylandOutputData *)data;
    } else {
        d->output_list = (struct SDL_WaylandOutputData *)data;
    }

    if (data->videodata->xdg_output_manager) {
        data->xdg_output = zxdg_output_manager_v1_get_xdg_output(data->videodata->xdg_output_manager, output);
        zxdg_output_v1_add_listener(data->xdg_output, &xdg_output_listener, data);
    }
}

static void Wayland_free_display(SDL_VideoData *d, uint32_t id)
{
    int num_displays = SDL_GetNumVideoDisplays();
    SDL_VideoDisplay *display;
    SDL_WaylandOutputData *data;
    int i;

    for (i = 0; i < num_displays; i += 1) {
        display = SDL_GetDisplay(i);
        data = (SDL_WaylandOutputData *)display->driverdata;
        if (data->registry_id == id) {
            if (d->output_list) {
                SDL_WaylandOutputData *node = d->output_list;
                if (node == data) {
                    d->output_list = node->next;
                } else {
                    while (node->next != data && node->next) {
                        node = node->next;
                    }
                    if (node->next) {
                        node->next = node->next->next;
                    }
                }
            }
            if (data->xdg_output) {
                zxdg_output_v1_destroy(data->xdg_output);
            }
            wl_output_destroy(data->output);
            SDL_DelVideoDisplay(i);

            /* Update the index for all remaining displays */
            num_displays -= 1;
            for (; i < num_displays; i += 1) {
                display = SDL_GetDisplay(i);
                data = (SDL_WaylandOutputData *)display->driverdata;
                data->index -= 1;
            }

            return;
        }
    }
}

static void Wayland_init_xdg_output(SDL_VideoData *d)
{
    SDL_WaylandOutputData *node;
    for (node = d->output_list; node; node = node->next) {
        node->xdg_output = zxdg_output_manager_v1_get_xdg_output(node->videodata->xdg_output_manager, node->output);
        zxdg_output_v1_add_listener(node->xdg_output, &xdg_output_listener, node);
    }
}

#ifdef SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH
static void windowmanager_hints(void *data, struct qt_windowmanager *qt_windowmanager,
                                int32_t show_is_fullscreen)
{
}

static void windowmanager_quit(void *data, struct qt_windowmanager *qt_windowmanager)
{
    SDL_SendQuit();
}

static const struct qt_windowmanager_listener windowmanager_listener = {
    windowmanager_hints,
    windowmanager_quit,
};
#endif /* SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH */

static void handle_ping_xdg_wm_base(void *data, struct xdg_wm_base *xdg, uint32_t serial)
{
    xdg_wm_base_pong(xdg, serial);
}

static const struct xdg_wm_base_listener shell_listener_xdg = {
    handle_ping_xdg_wm_base
};

#ifdef HAVE_LIBDECOR_H
static void libdecor_error(struct libdecor *context,
                           enum libdecor_error error,
                           const char *message)
{
    SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "libdecor error (%d): %s\n", error, message);
}

static struct libdecor_interface libdecor_interface = {
    libdecor_error,
};
#endif

static void display_handle_global(void *data, struct wl_registry *registry, uint32_t id,
                                  const char *interface, uint32_t version)
{
    SDL_VideoData *d = data;

    /*printf("WAYLAND INTERFACE: %s\n", interface);*/

    if (SDL_strcmp(interface, "wl_compositor") == 0) {
        d->compositor = wl_registry_bind(d->registry, id, &wl_compositor_interface, SDL_min(4, version));
    } else if (SDL_strcmp(interface, "wl_output") == 0) {
        Wayland_add_display(d, id);
    } else if (SDL_strcmp(interface, "wl_seat") == 0) {
        Wayland_display_add_input(d, id, version);
    } else if (SDL_strcmp(interface, "xdg_wm_base") == 0) {
        d->shell.xdg = wl_registry_bind(d->registry, id, &xdg_wm_base_interface, SDL_min(version, 3));
        xdg_wm_base_add_listener(d->shell.xdg, &shell_listener_xdg, NULL);
    } else if (SDL_strcmp(interface, "wl_shm") == 0) {
        d->shm = wl_registry_bind(registry, id, &wl_shm_interface, 1);
    } else if (SDL_strcmp(interface, "zwp_relative_pointer_manager_v1") == 0) {
        Wayland_display_add_relative_pointer_manager(d, id);
    } else if (SDL_strcmp(interface, "zwp_pointer_constraints_v1") == 0) {
        Wayland_display_add_pointer_constraints(d, id);
    } else if (SDL_strcmp(interface, "zwp_keyboard_shortcuts_inhibit_manager_v1") == 0) {
        d->key_inhibitor_manager = wl_registry_bind(d->registry, id, &zwp_keyboard_shortcuts_inhibit_manager_v1_interface, 1);
    } else if (SDL_strcmp(interface, "zwp_idle_inhibit_manager_v1") == 0) {
        d->idle_inhibit_manager = wl_registry_bind(d->registry, id, &zwp_idle_inhibit_manager_v1_interface, 1);
    } else if (SDL_strcmp(interface, "xdg_activation_v1") == 0) {
        d->activation_manager = wl_registry_bind(d->registry, id, &xdg_activation_v1_interface, 1);
    } else if (SDL_strcmp(interface, "zwp_text_input_manager_v3") == 0) {
        Wayland_add_text_input_manager(d, id, version);
    } else if (SDL_strcmp(interface, "wl_data_device_manager") == 0) {
        Wayland_add_data_device_manager(d, id, version);
    } else if (SDL_strcmp(interface, "zwp_primary_selection_device_manager_v1") == 0) {
        Wayland_add_primary_selection_device_manager(d, id, version);
    } else if (SDL_strcmp(interface, "zxdg_decoration_manager_v1") == 0) {
        d->decoration_manager = wl_registry_bind(d->registry, id, &zxdg_decoration_manager_v1_interface, 1);
    } else if (SDL_strcmp(interface, "zwp_tablet_manager_v2") == 0) {
        d->tablet_manager = wl_registry_bind(d->registry, id, &zwp_tablet_manager_v2_interface, 1);
        Wayland_input_add_tablet(d->input, d->tablet_manager);
    } else if (SDL_strcmp(interface, "zxdg_output_manager_v1") == 0) {
        version = SDL_min(version, 3); /* Versions 1 through 3 are supported. */
        d->xdg_output_manager = wl_registry_bind(d->registry, id, &zxdg_output_manager_v1_interface, version);
        Wayland_init_xdg_output(d);
    } else if (SDL_strcmp(interface, "wp_viewporter") == 0) {
        d->viewporter = wl_registry_bind(d->registry, id, &wp_viewporter_interface, 1);
    } else if (SDL_strcmp(interface, "wp_fractional_scale_manager_v1") == 0) {
        d->fractional_scale_manager = wl_registry_bind(d->registry, id, &wp_fractional_scale_manager_v1_interface, 1);
#ifdef SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH
    } else if (SDL_strcmp(interface, "qt_touch_extension") == 0) {
        Wayland_touch_create(d, id);
    } else if (SDL_strcmp(interface, "qt_surface_extension") == 0) {
        d->surface_extension = wl_registry_bind(registry, id,
                                                &qt_surface_extension_interface, 1);
    } else if (SDL_strcmp(interface, "qt_windowmanager") == 0) {
        d->windowmanager = wl_registry_bind(registry, id,
                                            &qt_windowmanager_interface, 1);
        qt_windowmanager_add_listener(d->windowmanager, &windowmanager_listener, d);
#endif /* SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH */
    }
}

static void display_remove_global(void *data, struct wl_registry *registry, uint32_t id)
{
    SDL_VideoData *d = data;
    /* We don't get an interface, just an ID, so assume it's a wl_output :shrug: */
    Wayland_free_display(d, id);
}

static const struct wl_registry_listener registry_listener = {
    display_handle_global,
    display_remove_global
};

#ifdef HAVE_LIBDECOR_H
static SDL_bool should_use_libdecor(SDL_VideoData *data, SDL_bool ignore_xdg)
{
    if (!SDL_WAYLAND_HAVE_WAYLAND_LIBDECOR) {
        return SDL_FALSE;
    }

    if (!SDL_GetHintBoolean(SDL_HINT_VIDEO_WAYLAND_ALLOW_LIBDECOR, SDL_TRUE)) {
        return SDL_FALSE;
    }

    if (SDL_GetHintBoolean(SDL_HINT_VIDEO_WAYLAND_PREFER_LIBDECOR, SDL_FALSE)) {
        return SDL_TRUE;
    }

    if (ignore_xdg) {
        return SDL_TRUE;
    }

    if (data->decoration_manager) {
        return SDL_FALSE;
    }

    return SDL_TRUE;
}
#endif

SDL_bool Wayland_LoadLibdecor(SDL_VideoData *data, SDL_bool ignore_xdg)
{
#ifdef HAVE_LIBDECOR_H
    if (data->shell.libdecor != NULL) {
        return SDL_TRUE; /* Already loaded! */
    }
    if (should_use_libdecor(data, ignore_xdg)) {
        data->shell.libdecor = libdecor_new(data->display, &libdecor_interface);
        return data->shell.libdecor != NULL;
    }
#endif
    return SDL_FALSE;
}

int Wayland_VideoInit(_THIS)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->driverdata;

    data->xkb_context = WAYLAND_xkb_context_new(0);
    if (!data->xkb_context) {
        return SDL_SetError("Failed to create XKB context");
    }

    data->registry = wl_display_get_registry(data->display);
    if (!data->registry) {
        return SDL_SetError("Failed to get the Wayland registry");
    }

    wl_registry_add_listener(data->registry, &registry_listener, data);

    // First roundtrip to receive all registry objects.
    WAYLAND_wl_display_roundtrip(data->display);

    /* Now that we have all the protocols, load libdecor if applicable */
    Wayland_LoadLibdecor(data, SDL_FALSE);

    // Second roundtrip to receive all output events.
    WAYLAND_wl_display_roundtrip(data->display);

    Wayland_InitMouse();

    /* Get the surface class name, usually the name of the application */
    data->classname = get_classname();

    WAYLAND_wl_display_flush(data->display);

    Wayland_InitKeyboard(_this);
    Wayland_InitWin(data);

    data->initializing = SDL_FALSE;

    return 0;
}

static int Wayland_GetDisplayBounds(_THIS, SDL_VideoDisplay *display, SDL_Rect *rect)
{
    SDL_WaylandOutputData *driverdata = (SDL_WaylandOutputData *)display->driverdata;
    rect->x = driverdata->x;
    rect->y = driverdata->y;
    rect->w = display->current_mode.w;
    rect->h = display->current_mode.h;
    return 0;
}

static int Wayland_GetDisplayDPI(_THIS, SDL_VideoDisplay *sdl_display, float *ddpi, float *hdpi, float *vdpi)
{
    SDL_WaylandOutputData *driverdata = (SDL_WaylandOutputData *)sdl_display->driverdata;

    if (ddpi) {
        *ddpi = driverdata->ddpi;
    }
    if (hdpi) {
        *hdpi = driverdata->hdpi;
    }
    if (vdpi) {
        *vdpi = driverdata->vdpi;
    }

    return driverdata->ddpi != 0.0f ? 0 : SDL_SetError("Couldn't get DPI");
}

static void Wayland_VideoCleanup(_THIS)
{
    SDL_VideoData *data = _this->driverdata;
    int i;

    Wayland_QuitWin(data);
    Wayland_FiniMouse(data);

    for (i = _this->num_displays - 1; i >= 0; --i) {
        SDL_VideoDisplay *display = &_this->displays[i];

        if (((SDL_WaylandOutputData *)display->driverdata)->xdg_output) {
            zxdg_output_v1_destroy(((SDL_WaylandOutputData *)display->driverdata)->xdg_output);
        }

        wl_output_destroy(((SDL_WaylandOutputData *)display->driverdata)->output);
        SDL_DelVideoDisplay(i);
    }
    data->output_list = NULL;

    Wayland_display_destroy_input(data);
    Wayland_display_destroy_pointer_constraints(data);
    Wayland_display_destroy_relative_pointer_manager(data);

    if (data->activation_manager) {
        xdg_activation_v1_destroy(data->activation_manager);
        data->activation_manager = NULL;
    }

    if (data->idle_inhibit_manager) {
        zwp_idle_inhibit_manager_v1_destroy(data->idle_inhibit_manager);
        data->idle_inhibit_manager = NULL;
    }

    if (data->key_inhibitor_manager) {
        zwp_keyboard_shortcuts_inhibit_manager_v1_destroy(data->key_inhibitor_manager);
        data->key_inhibitor_manager = NULL;
    }

    Wayland_QuitKeyboard(_this);

    if (data->text_input_manager) {
        zwp_text_input_manager_v3_destroy(data->text_input_manager);
        data->text_input_manager = NULL;
    }

    if (data->xkb_context) {
        WAYLAND_xkb_context_unref(data->xkb_context);
        data->xkb_context = NULL;
    }
#ifdef SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH
    if (data->windowmanager) {
        qt_windowmanager_destroy(data->windowmanager);
        data->windowmanager = NULL;
    }

    if (data->surface_extension) {
        qt_surface_extension_destroy(data->surface_extension);
        data->surface_extension = NULL;
    }

    Wayland_touch_destroy(data);
#endif /* SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH */

    if (data->tablet_manager) {
        zwp_tablet_manager_v2_destroy((struct zwp_tablet_manager_v2 *)data->tablet_manager);
        data->tablet_manager = NULL;
    }

    if (data->data_device_manager) {
        wl_data_device_manager_destroy(data->data_device_manager);
        data->data_device_manager = NULL;
    }

    if (data->shm) {
        wl_shm_destroy(data->shm);
        data->shm = NULL;
    }

    if (data->shell.xdg) {
        xdg_wm_base_destroy(data->shell.xdg);
        data->shell.xdg = NULL;
    }

    if (data->decoration_manager) {
        zxdg_decoration_manager_v1_destroy(data->decoration_manager);
        data->decoration_manager = NULL;
    }

    if (data->xdg_output_manager) {
        zxdg_output_manager_v1_destroy(data->xdg_output_manager);
        data->xdg_output_manager = NULL;
    }

    if (data->viewporter) {
        wp_viewporter_destroy(data->viewporter);
        data->viewporter = NULL;
    }

    if (data->primary_selection_device_manager) {
        zwp_primary_selection_device_manager_v1_destroy(data->primary_selection_device_manager);
        data->primary_selection_device_manager = NULL;
    }

    if (data->fractional_scale_manager) {
        wp_fractional_scale_manager_v1_destroy(data->fractional_scale_manager);
        data->fractional_scale_manager = NULL;
    }

    if (data->compositor) {
        wl_compositor_destroy(data->compositor);
        data->compositor = NULL;
    }

    if (data->registry) {
        wl_registry_destroy(data->registry);
        data->registry = NULL;
    }
}

SDL_bool Wayland_VideoReconnect(_THIS)
{
#if 0 /* TODO RECONNECT: Uncomment all when https://invent.kde.org/plasma/kwin/-/wikis/Restarting is completed */
    SDL_VideoData *data = _this->driverdata;

    SDL_Window *window = NULL;

    SDL_GLContext current_ctx = SDL_GL_GetCurrentContext();
    SDL_Window *current_window = SDL_GL_GetCurrentWindow();

    SDL_GL_MakeCurrent(NULL, NULL);
    Wayland_VideoCleanup(_this);

    SDL_ResetKeyboard();
    SDL_ResetMouse();
    if (WAYLAND_wl_display_reconnect(data->display) < 0) {
        return SDL_FALSE;
    }

    Wayland_VideoInit(_this);

    window = _this->windows;
    while (window) {
        /* We're going to cheat _just_ for a second and strip the OpenGL flag.
         * The Wayland driver actually forces it in CreateWindow, and
         * RecreateWindow does a bunch of unloading/loading of libGL, so just
         * strip the flag so RecreateWindow doesn't mess with the GL context,
         * and CreateWindow will add it right back!
         * -flibit
         */
        window->flags &= ~SDL_WINDOW_OPENGL;

        SDL_RecreateWindow(window, window->flags);
        window = window->next;
    }

    Wayland_RecreateCursors();

    if (current_window && current_ctx) {
        SDL_GL_MakeCurrent (current_window, current_ctx);
    }
    return SDL_TRUE;
#else
    return SDL_FALSE;
#endif /* 0 */
}

void Wayland_VideoQuit(_THIS)
{
    SDL_VideoData *data = _this->driverdata;

    Wayland_VideoCleanup(_this);

#ifdef HAVE_LIBDECOR_H
    if (data->shell.libdecor) {
        libdecor_unref(data->shell.libdecor);
        data->shell.libdecor = NULL;
    }
#endif

    SDL_free(data->classname);
}

#endif /* SDL_VIDEO_DRIVER_WAYLAND */

/* vi: set ts=4 sw=4 expandtab: */
