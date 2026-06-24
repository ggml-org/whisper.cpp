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

#ifdef SDL_VIDEO_DRIVER_KMSDRM

/* SDL internals */
#include "../SDL_sysvideo.h"
#include "SDL_syswm.h"
#include "SDL_log.h"
#include "SDL_hints.h"
#include "../../events/SDL_events_c.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_keyboard_c.h"

#ifdef SDL_INPUT_LINUXEV
#include "../../core/linux/SDL_evdev.h"
#elif defined SDL_INPUT_WSCONS
#include "../../core/openbsd/SDL_wscons.h"
#endif

/* KMS/DRM declarations */
#include "SDL_kmsdrmvideo.h"
#include "SDL_kmsdrmevents.h"
#include "SDL_kmsdrmopengles.h"
#include "SDL_kmsdrmmouse.h"
#include "SDL_kmsdrmdyn.h"
#include "SDL_kmsdrmvulkan.h"
#include <sys/stat.h>
#include <sys/param.h>
#include <sys/utsname.h>
#include <dirent.h>
#include <poll.h>
#include <errno.h>

#ifdef __OpenBSD__
static SDL_bool moderndri = SDL_FALSE;
#else
static SDL_bool moderndri = SDL_TRUE;
#endif

static char kmsdrm_dri_path[16];
static int kmsdrm_dri_pathsize = 0;
static char kmsdrm_dri_devname[8];
static int kmsdrm_dri_devnamesize = 0;
static char kmsdrm_dri_cardpath[32];

#ifndef EGL_PLATFORM_GBM_MESA
#define EGL_PLATFORM_GBM_MESA 0x31D7
#endif

static int get_driindex(void)
{
    int available = -ENOENT;
    char device[sizeof(kmsdrm_dri_cardpath)];
    int drm_fd;
    int i;
    int devindex = -1;
    DIR *folder;
    const char *hint;
    struct dirent *res;

    hint = SDL_GetHint(SDL_HINT_KMSDRM_DEVICE_INDEX);
    if (hint && *hint) {
        char *endptr = NULL;
        const int idx = (int)SDL_strtol(hint, &endptr, 10);
        if ((*endptr == '\0') && (idx >= 0)) { /* *endptr==0 means "whole string was a valid number" */
            return idx;                        /* we'll take the user's request here. */
        }
    }

    SDL_strlcpy(device, kmsdrm_dri_path, sizeof(device));
    folder = opendir(device);
    if (!folder) {
        SDL_SetError("Failed to open directory '%s'", device);
        return -ENOENT;
    }

    SDL_strlcpy(device + kmsdrm_dri_pathsize, kmsdrm_dri_devname,
                sizeof(device) - kmsdrm_dri_pathsize);
    while((res = readdir(folder)) != NULL && available < 0) {
        if (SDL_memcmp(res->d_name, kmsdrm_dri_devname,
                       kmsdrm_dri_devnamesize) == 0) {
            SDL_strlcpy(device + kmsdrm_dri_pathsize + kmsdrm_dri_devnamesize,
                        res->d_name + kmsdrm_dri_devnamesize,
                        sizeof(device) - kmsdrm_dri_pathsize -
                            kmsdrm_dri_devnamesize);

            drm_fd = open(device, O_RDWR | O_CLOEXEC);
            if (drm_fd >= 0) {
                devindex = SDL_atoi(device + kmsdrm_dri_pathsize +
                                    kmsdrm_dri_devnamesize);
                if (SDL_KMSDRM_LoadSymbols()) {
                    drmModeRes *resources = KMSDRM_drmModeGetResources(drm_fd);
                    if (resources) {
                        SDL_LogDebug(SDL_LOG_CATEGORY_VIDEO,
                                     "%s%d connector, encoder and CRTC counts are: %d %d %d",
                                     kmsdrm_dri_cardpath, devindex,
                                     resources->count_connectors,
                                     resources->count_encoders,
                                     resources->count_crtcs);

                        if (resources->count_connectors > 0 &&
                            resources->count_encoders > 0 &&
                            resources->count_crtcs > 0) {
                            available = -ENOENT;
                            for (i = 0; i < resources->count_connectors && available < 0; i++) {
                                drmModeConnector *conn =
                                    KMSDRM_drmModeGetConnector(
                                        drm_fd, resources->connectors[i]);

                                if (!conn) {
                                    continue;
                                }

                                if (conn->connection == DRM_MODE_CONNECTED &&
                                    conn->count_modes) {
                                    SDL_bool access_denied = SDL_FALSE;
                                    if (SDL_GetHintBoolean(
                                            SDL_HINT_KMSDRM_REQUIRE_DRM_MASTER,
                                            SDL_TRUE)) {
                                        /* Skip this device if we can't obtain
                                         * DRM master */
                                        KMSDRM_drmSetMaster(drm_fd);
                                        if (KMSDRM_drmAuthMagic(drm_fd, 0) == -EACCES) {
                                            access_denied = SDL_TRUE;
                                        }
                                    }

                                    if (!access_denied) {
                                        available = devindex;
                                    }
                                }

                                KMSDRM_drmModeFreeConnector(conn);
                            }
                        }
                        KMSDRM_drmModeFreeResources(resources);
                    }
                    SDL_KMSDRM_UnloadSymbols();
                }
                close(drm_fd);
            } else {
                SDL_LogDebug(SDL_LOG_CATEGORY_VIDEO,
                             "Failed to open KMSDRM device %s, errno: %d\n", device, errno);
            }
        }
    }

    closedir(folder);

    return available;
}

static int KMSDRM_Available(void)
{
#ifdef __OpenBSD__
    struct utsname nameofsystem;
    double releaseversion;
#endif
    int ret = -ENOENT;

#ifdef __OpenBSD__
    if (!(uname(&nameofsystem) < 0)) {
        releaseversion = SDL_atof(nameofsystem.release);
        if (releaseversion >= 6.9) {
            moderndri = SDL_TRUE;
        }
    }
#endif

    if (moderndri) {
        SDL_strlcpy(kmsdrm_dri_path, "/dev/dri/", sizeof(kmsdrm_dri_path));
        SDL_strlcpy(kmsdrm_dri_devname, "card", sizeof(kmsdrm_dri_devname));
    } else {
        SDL_strlcpy(kmsdrm_dri_path, "/dev/", sizeof(kmsdrm_dri_path));
        SDL_strlcpy(kmsdrm_dri_devname, "drm", sizeof(kmsdrm_dri_devname));
    }

    kmsdrm_dri_pathsize = SDL_strlen(kmsdrm_dri_path);
    kmsdrm_dri_devnamesize = SDL_strlen(kmsdrm_dri_devname);
    (void)SDL_snprintf(kmsdrm_dri_cardpath, sizeof(kmsdrm_dri_cardpath), "%s%s",
                       kmsdrm_dri_path, kmsdrm_dri_devname);

    ret = get_driindex();
    if (ret >= 0) {
        return 1;
    }

    return ret;
}

static void KMSDRM_DeleteDevice(SDL_VideoDevice *device)
{
    if (device->driverdata) {
        SDL_free(device->driverdata);
        device->driverdata = NULL;
    }

    SDL_free(device);

    SDL_KMSDRM_UnloadSymbols();
}

static SDL_VideoDevice *KMSDRM_CreateDevice(void)
{
    SDL_VideoDevice *device;
    SDL_VideoData *viddata;
    int devindex;

    if (!KMSDRM_Available()) {
        return NULL;
    }

    devindex = get_driindex();
    if (devindex < 0) {
        SDL_SetError("devindex (%d) must not be negative.", devindex);
        return NULL;
    }

    if (!SDL_KMSDRM_LoadSymbols()) {
        return NULL;
    }

    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        SDL_OutOfMemory();
        return NULL;
    }

    viddata = (SDL_VideoData *)SDL_calloc(1, sizeof(SDL_VideoData));
    if (!viddata) {
        SDL_OutOfMemory();
        goto cleanup;
    }
    viddata->devindex = devindex;
    viddata->drm_fd = -1;

    device->driverdata = viddata;

    /* Setup all functions which we can handle */
    device->VideoInit = KMSDRM_VideoInit;
    device->VideoQuit = KMSDRM_VideoQuit;
    device->GetDisplayModes = KMSDRM_GetDisplayModes;
    device->SetDisplayMode = KMSDRM_SetDisplayMode;
    device->CreateSDLWindow = KMSDRM_CreateWindow;
    device->CreateSDLWindowFrom = KMSDRM_CreateWindowFrom;
    device->SetWindowTitle = KMSDRM_SetWindowTitle;
    device->SetWindowIcon = KMSDRM_SetWindowIcon;
    device->SetWindowPosition = KMSDRM_SetWindowPosition;
    device->SetWindowSize = KMSDRM_SetWindowSize;
    device->SetWindowFullscreen = KMSDRM_SetWindowFullscreen;
    device->GetWindowGammaRamp = KMSDRM_GetWindowGammaRamp;
    device->SetWindowGammaRamp = KMSDRM_SetWindowGammaRamp;
    device->ShowWindow = KMSDRM_ShowWindow;
    device->HideWindow = KMSDRM_HideWindow;
    device->RaiseWindow = KMSDRM_RaiseWindow;
    device->MaximizeWindow = KMSDRM_MaximizeWindow;
    device->MinimizeWindow = KMSDRM_MinimizeWindow;
    device->RestoreWindow = KMSDRM_RestoreWindow;
    device->DestroyWindow = KMSDRM_DestroyWindow;
    device->GetWindowWMInfo = KMSDRM_GetWindowWMInfo;

    device->GL_LoadLibrary = KMSDRM_GLES_LoadLibrary;
    device->GL_GetProcAddress = KMSDRM_GLES_GetProcAddress;
    device->GL_UnloadLibrary = KMSDRM_GLES_UnloadLibrary;
    device->GL_CreateContext = KMSDRM_GLES_CreateContext;
    device->GL_MakeCurrent = KMSDRM_GLES_MakeCurrent;
    device->GL_SetSwapInterval = KMSDRM_GLES_SetSwapInterval;
    device->GL_GetSwapInterval = KMSDRM_GLES_GetSwapInterval;
    device->GL_SwapWindow = KMSDRM_GLES_SwapWindow;
    device->GL_DeleteContext = KMSDRM_GLES_DeleteContext;
    device->GL_DefaultProfileConfig = KMSDRM_GLES_DefaultProfileConfig;

#ifdef SDL_VIDEO_VULKAN
    device->Vulkan_LoadLibrary = KMSDRM_Vulkan_LoadLibrary;
    device->Vulkan_UnloadLibrary = KMSDRM_Vulkan_UnloadLibrary;
    device->Vulkan_GetInstanceExtensions = KMSDRM_Vulkan_GetInstanceExtensions;
    device->Vulkan_CreateSurface = KMSDRM_Vulkan_CreateSurface;
    device->Vulkan_GetDrawableSize = KMSDRM_Vulkan_GetDrawableSize;
#endif

    device->PumpEvents = KMSDRM_PumpEvents;
    device->free = KMSDRM_DeleteDevice;

    return device;

cleanup:
    SDL_free(device);

    if (viddata) {
        SDL_free(viddata);
    }
    return NULL;
}

VideoBootStrap KMSDRM_bootstrap = {
    "KMSDRM",
    "KMS/DRM Video Driver",
    KMSDRM_CreateDevice,
    NULL /* no ShowMessageBox implementation */
};

static void KMSDRM_FBDestroyCallback(struct gbm_bo *bo, void *data)
{
    KMSDRM_FBInfo *fb_info = (KMSDRM_FBInfo *)data;

    if (fb_info && fb_info->drm_fd >= 0 && fb_info->fb_id != 0) {
        KMSDRM_drmModeRmFB(fb_info->drm_fd, fb_info->fb_id);
        SDL_LogDebug(SDL_LOG_CATEGORY_VIDEO, "Delete DRM FB %u", fb_info->fb_id);
    }

    SDL_free(fb_info);
}

KMSDRM_FBInfo *KMSDRM_FBFromBO(_THIS, struct gbm_bo *bo)
{
    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);
    unsigned w, h;
    int rc = -1;
    int num_planes = 0;
    uint32_t format, strides[4] = { 0 }, handles[4] = { 0 }, offsets[4] = { 0 }, flags = 0;
    uint64_t modifiers[4] = { 0 };

    /* Check for an existing framebuffer */
    KMSDRM_FBInfo *fb_info = (KMSDRM_FBInfo *)KMSDRM_gbm_bo_get_user_data(bo);

    if (fb_info) {
        return fb_info;
    }

    /* Create a structure that contains enough info to remove the framebuffer
       when the backing buffer is destroyed */
    fb_info = (KMSDRM_FBInfo *)SDL_calloc(1, sizeof(KMSDRM_FBInfo));

    if (!fb_info) {
        SDL_OutOfMemory();
        return NULL;
    }

    fb_info->drm_fd = viddata->drm_fd;

    /* Create framebuffer object for the buffer using the modifiers requested by GBM.
       Use of the modifiers is necessary on some platforms. */
    w = KMSDRM_gbm_bo_get_width(bo);
    h = KMSDRM_gbm_bo_get_height(bo);
    format = KMSDRM_gbm_bo_get_format(bo);

    if (KMSDRM_drmModeAddFB2WithModifiers &&
        KMSDRM_gbm_bo_get_modifier &&
        KMSDRM_gbm_bo_get_plane_count &&
        KMSDRM_gbm_bo_get_offset &&
        KMSDRM_gbm_bo_get_stride_for_plane &&
        KMSDRM_gbm_bo_get_handle_for_plane) {

        modifiers[0] = KMSDRM_gbm_bo_get_modifier(bo);
        num_planes = KMSDRM_gbm_bo_get_plane_count(bo);
        for (int i = 0; i < num_planes; i++) {
            strides[i] = KMSDRM_gbm_bo_get_stride_for_plane(bo, i);
            handles[i] = KMSDRM_gbm_bo_get_handle_for_plane(bo, i).u32;
            offsets[i] = KMSDRM_gbm_bo_get_offset(bo, i);
            modifiers[i] = modifiers[0];
        }

        if (modifiers[0] && modifiers[0] != DRM_FORMAT_MOD_INVALID) {
            flags = DRM_MODE_FB_MODIFIERS;
        }

        rc = KMSDRM_drmModeAddFB2WithModifiers(viddata->drm_fd, w, h, format, handles, strides, offsets, modifiers, &fb_info->fb_id, flags);
    }

    if (rc < 0) {
        strides[0] = KMSDRM_gbm_bo_get_stride(bo);
        handles[0] = KMSDRM_gbm_bo_get_handle(bo).u32;
        rc = KMSDRM_drmModeAddFB(viddata->drm_fd, w, h, 24, 32, strides[0], handles[0], &fb_info->fb_id);
    }

    if (rc < 0) {
        SDL_free(fb_info);
        return NULL;
    }

    SDL_LogDebug(SDL_LOG_CATEGORY_VIDEO, "New DRM FB (%u): %ux%u, from BO %p",
                 fb_info->fb_id, w, h, (void *)bo);

    /* Associate our DRM framebuffer with this buffer object */
    KMSDRM_gbm_bo_set_user_data(bo, fb_info, KMSDRM_FBDestroyCallback);

    return fb_info;
}

static void KMSDRM_FlipHandler(int fd, unsigned int frame, unsigned int sec, unsigned int usec, void *data)
{
    *((SDL_bool *)data) = SDL_FALSE;
}

SDL_bool KMSDRM_WaitPageflip(_THIS, SDL_WindowData *windata)
{

    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);
    drmEventContext ev = { 0 };
    struct pollfd pfd = { 0 };
    int ret;

    ev.version = DRM_EVENT_CONTEXT_VERSION;
    ev.page_flip_handler = KMSDRM_FlipHandler;

    pfd.fd = viddata->drm_fd;
    pfd.events = POLLIN;

    /* Stay on the while loop until we get the desired event.
       We need the while the loop because we could be in a situation where:
       -We get and event on the FD in time, thus not on exiting on return number 1.
       -The event is not an error, thus not exiting on return number 2.
       -The event is of POLLIN type, but even then, if the event is not a pageflip,
        drmHandleEvent() won't unset wait_for_pageflip, so we have to iterate
        and go polling again.

        If it wasn't for the while loop, we could erroneously exit the function
        without the pageflip event to arrive!

      For example, vblank events hit the FD and they are POLLIN events too (POLLIN
      means "there's data to read on the FD"), but they are not the pageflip event
      we are waiting for, so the drmEventHandle() doesn't run the flip handler, and
      since waiting_for_flip is set on the pageflip handle, it's not set and we stay
      on the loop, until we get the event for the pageflip, which is fine.
    */
    while (windata->waiting_for_flip) {

        pfd.revents = 0;

        /* poll() waits for events arriving on the FD, and returns < 0 if timeout passes
           with no events or a signal occurred before any requested event (-EINTR).
           We wait forever (timeout = -1), but even if we DO get an event, we have yet
           to see if it's of the required type, then if it's a pageflip, etc */
        ret = poll(&pfd, 1, -1);

        if (ret < 0) {
            if (errno == EINTR) {
                /* poll() returning < 0 and setting errno = EINTR means there was a signal before
                   any requested event, so we immediately poll again. */
                continue;
            } else {
                /* There was another error. Don't pull again or we could get into a busy loop. */
                SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "DRM poll error");
                return SDL_FALSE; /* Return number 1. */
            }
        }

        if (pfd.revents & (POLLHUP | POLLERR)) {
            /* An event arrived on the FD in time, but it's an error. */
            SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "DRM poll hup or error");
            return SDL_FALSE; /* Return number 2. */
        }

        if (pfd.revents & POLLIN) {
            /* There is data to read on the FD!
               Is the event a pageflip? We know it is a pageflip if it matches the
               event we are passing in &ev. If it does, drmHandleEvent() will unset
               windata->waiting_for_flip and we will get out of the "while" loop.
               If it's not, we keep iterating on the loop. */
            KMSDRM_drmHandleEvent(viddata->drm_fd, &ev);
        }

        /* If we got to this point in the loop, we may iterate or exit the loop:
           -A legit (non-error) event arrived, and it was a POLLING event, and it was consumed
            by drmHandleEvent().
              -If it was a PAGEFLIP event, waiting_for_flip will be unset by drmHandleEvent()
               and we will exit the loop.
              -If it wasn't a PAGEFLIP, drmHandleEvent() won't unset waiting_for_flip, so we
               iterare back to polling.
           -A legit (non-error) event arrived, but it's not a POLLIN event, so it hasn't to be
            consumed by drmHandleEvent(), so waiting_for_flip isn't set and we iterate back
            to polling. */
    }

    return SDL_TRUE;
}

/* Given w, h and refresh rate, returns the closest DRM video mode
   available on the DRM connector of the display.
   We use the SDL mode list (which we filled in KMSDRM_GetDisplayModes)
   because it's ordered, while the list on the connector is mostly random.*/
static drmModeModeInfo *KMSDRM_GetClosestDisplayMode(SDL_VideoDisplay *display,
                                                     uint32_t width, uint32_t height, uint32_t refresh_rate)
{

    SDL_DisplayData *dispdata = (SDL_DisplayData *)display->driverdata;
    drmModeConnector *connector = dispdata->connector;

    SDL_DisplayMode target, closest;
    drmModeModeInfo *drm_mode;

    target.w = width;
    target.h = height;
    target.format = 0; /* Will use the default mode format. */
    target.refresh_rate = refresh_rate;
    target.driverdata = 0; /* Initialize to 0 */

    if (!SDL_GetClosestDisplayMode(SDL_atoi(display->name), &target, &closest)) {
        return NULL;
    } else {
        SDL_DisplayModeData *modedata = (SDL_DisplayModeData *)closest.driverdata;
        drm_mode = &connector->modes[modedata->mode_index];
        return drm_mode;
    }
}

/*****************************************************************************/
/* SDL Video and Display initialization/handling functions                   */
/* _this is a SDL_VideoDevice *                                              */
/*****************************************************************************/

static SDL_bool KMSDRM_DropMaster(_THIS)
{
    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);

    /* Check if we have DRM master to begin with */
    if (KMSDRM_drmAuthMagic(viddata->drm_fd, 0) == -EACCES) {
        /* Nope, nothing to do then */
        return SDL_TRUE;
    }

    return KMSDRM_drmDropMaster(viddata->drm_fd) < 0 ? SDL_FALSE : SDL_TRUE;
}

/* Deinitializes the driverdata of the SDL Displays in the SDL display list. */
static void KMSDRM_DeinitDisplays(_THIS)
{
    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);
    SDL_DisplayData *dispdata;
    int num_displays, i;

    num_displays = SDL_GetNumVideoDisplays();

    /* Iterate on the SDL Display list. */
    for (i = 0; i < num_displays; i++) {

        /* Get the driverdata for this display */
        dispdata = (SDL_DisplayData *)SDL_GetDisplayDriverData(i);

        /* Free connector */
        if (dispdata && dispdata->connector) {
            KMSDRM_drmModeFreeConnector(dispdata->connector);
            dispdata->connector = NULL;
        }

        /* Free CRTC */
        if (dispdata && dispdata->crtc) {
            KMSDRM_drmModeFreeCrtc(dispdata->crtc);
            dispdata->crtc = NULL;
        }
    }

    if (viddata->drm_fd >= 0) {
        close(viddata->drm_fd);
        viddata->drm_fd = -1;
    }
}

static uint32_t KMSDRM_CrtcGetPropId(uint32_t drm_fd,
                                     drmModeObjectPropertiesPtr props,
                                     char const *name)
{
    uint32_t i, prop_id = 0;

    for (i = 0; !prop_id && i < props->count_props; ++i) {
        drmModePropertyPtr drm_prop =
            KMSDRM_drmModeGetProperty(drm_fd, props->props[i]);

        if (!drm_prop) {
            continue;
        }

        if (SDL_strcmp(drm_prop->name, name) == 0) {
            prop_id = drm_prop->prop_id;
        }

        KMSDRM_drmModeFreeProperty(drm_prop);
    }

    return prop_id;
}

static SDL_bool KMSDRM_VrrPropId(uint32_t drm_fd, uint32_t crtc_id, uint32_t *vrr_prop_id)
{
    drmModeObjectPropertiesPtr drm_props;

    drm_props = KMSDRM_drmModeObjectGetProperties(drm_fd,
                                                  crtc_id,
                                                  DRM_MODE_OBJECT_CRTC);

    if (!drm_props) {
        return SDL_FALSE;
    }

    *vrr_prop_id = KMSDRM_CrtcGetPropId(drm_fd,
                                        drm_props,
                                        "VRR_ENABLED");

    KMSDRM_drmModeFreeObjectProperties(drm_props);

    return SDL_TRUE;
}

static SDL_bool KMSDRM_ConnectorCheckVrrCapable(uint32_t drm_fd,
                                                uint32_t output_id,
                                                char const *name)
{
    uint32_t i;
    SDL_bool found = SDL_FALSE;
    uint64_t prop_value = 0;

    drmModeObjectPropertiesPtr props = KMSDRM_drmModeObjectGetProperties(drm_fd,
                                                                         output_id,
                                                                         DRM_MODE_OBJECT_CONNECTOR);

    if (!props) {
        return SDL_FALSE;
    }

    for (i = 0; !found && i < props->count_props; ++i) {
        drmModePropertyPtr drm_prop = KMSDRM_drmModeGetProperty(drm_fd, props->props[i]);

        if (!drm_prop) {
            continue;
        }

        if (SDL_strcasecmp(drm_prop->name, name) == 0) {
            prop_value = props->prop_values[i];
            found = SDL_TRUE;
        }

        KMSDRM_drmModeFreeProperty(drm_prop);
    }
    if (found) {
        return prop_value ? SDL_TRUE : SDL_FALSE;
    }

    return SDL_FALSE;
}

void KMSDRM_CrtcSetVrr(uint32_t drm_fd, uint32_t crtc_id, SDL_bool enabled)
{
    uint32_t vrr_prop_id;
    if (!KMSDRM_VrrPropId(drm_fd, crtc_id, &vrr_prop_id)) {
        return;
    }

    KMSDRM_drmModeObjectSetProperty(drm_fd,
                                    crtc_id,
                                    DRM_MODE_OBJECT_CRTC,
                                    vrr_prop_id,
                                    enabled);
}

static SDL_bool KMSDRM_CrtcGetVrr(uint32_t drm_fd, uint32_t crtc_id)
{
    uint32_t object_prop_id, vrr_prop_id;
    drmModeObjectPropertiesPtr props;
    SDL_bool object_prop_value;
    int i;

    if (!KMSDRM_VrrPropId(drm_fd, crtc_id, &vrr_prop_id)) {
        return SDL_FALSE;
    }

    props = KMSDRM_drmModeObjectGetProperties(drm_fd,
                                              crtc_id,
                                              DRM_MODE_OBJECT_CRTC);

    if (!props) {
        return SDL_FALSE;
    }

    for (i = 0; i < props->count_props; ++i) {
        drmModePropertyPtr drm_prop = KMSDRM_drmModeGetProperty(drm_fd, props->props[i]);

        if (!drm_prop) {
            continue;
        }

        object_prop_id = drm_prop->prop_id;
        object_prop_value = props->prop_values[i] ? SDL_TRUE : SDL_FALSE;

        KMSDRM_drmModeFreeProperty(drm_prop);

        if (object_prop_id == vrr_prop_id) {
            return object_prop_value;
        }
    }
    return SDL_FALSE;
}

/* Gets a DRM connector, builds an SDL_Display with it, and adds it to the
   list of SDL Displays in _this->displays[]  */
static void KMSDRM_AddDisplay(_THIS, drmModeConnector *connector, drmModeRes *resources)
{
    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);
    SDL_DisplayData *dispdata = NULL;
    SDL_VideoDisplay display = { 0 };
    SDL_DisplayModeData *modedata = NULL;
    drmModeEncoder *encoder = NULL;
    drmModeCrtc *crtc = NULL;
    int mode_index;
    int i, j;
    int ret = 0;

    /* Reserve memory for the new display's driverdata. */
    dispdata = (SDL_DisplayData *)SDL_calloc(1, sizeof(SDL_DisplayData));
    if (!dispdata) {
        ret = SDL_OutOfMemory();
        goto cleanup;
    }

    /* Initialize some of the members of the new display's driverdata
       to sane values. */
    dispdata->cursor_bo = NULL;
    dispdata->cursor_bo_drm_fd = -1;

    /* Since we create and show the default cursor on KMSDRM_InitMouse(),
       and we call KMSDRM_InitMouse() when we create a window, we have to know
       if the display used by the window already has a default cursor or not.
       If we don't, new default cursors would stack up on mouse->cursors and SDL
       would have to hide and delete them at quit, not to mention the memory leak... */
    dispdata->default_cursor_init = SDL_FALSE;

    /* Try to find the connector's current encoder */
    for (i = 0; i < resources->count_encoders; i++) {
        encoder = KMSDRM_drmModeGetEncoder(viddata->drm_fd, resources->encoders[i]);

        if (!encoder) {
            continue;
        }

        if (encoder->encoder_id == connector->encoder_id) {
            break;
        }

        KMSDRM_drmModeFreeEncoder(encoder);
        encoder = NULL;
    }

    if (!encoder) {
        /* No encoder was connected, find the first supported one */
        for (i = 0; i < resources->count_encoders; i++) {
            encoder = KMSDRM_drmModeGetEncoder(viddata->drm_fd,
                                               resources->encoders[i]);

            if (!encoder) {
                continue;
            }

            for (j = 0; j < connector->count_encoders; j++) {
                if (connector->encoders[j] == encoder->encoder_id) {
                    break;
                }
            }

            if (j != connector->count_encoders) {
                break;
            }

            KMSDRM_drmModeFreeEncoder(encoder);
            encoder = NULL;
        }
    }

    if (!encoder) {
        ret = SDL_SetError("No connected encoder found for connector.");
        goto cleanup;
    }

    /* Try to find a CRTC connected to this encoder */
    crtc = KMSDRM_drmModeGetCrtc(viddata->drm_fd, encoder->crtc_id);

    /* If no CRTC was connected to the encoder, find the first CRTC
       that is supported by the encoder, and use that. */
    if (!crtc) {
        for (i = 0; i < resources->count_crtcs; i++) {
            if (encoder->possible_crtcs & (1 << i)) {
                encoder->crtc_id = resources->crtcs[i];
                crtc = KMSDRM_drmModeGetCrtc(viddata->drm_fd, encoder->crtc_id);
                break;
            }
        }
    }

    if (!crtc) {
        ret = SDL_SetError("No CRTC found for connector.");
        goto cleanup;
    }

    /* Find the index of the mode attached to this CRTC */
    mode_index = -1;

    for (i = 0; i < connector->count_modes; i++) {
        drmModeModeInfo *mode = &connector->modes[i];

        if (!SDL_memcmp(mode, &crtc->mode, sizeof(crtc->mode))) {
            mode_index = i;
            break;
        }
    }

    if (mode_index == -1) {
        int current_area, largest_area = 0;

        /* Find the preferred mode or the highest resolution mode */
        for (i = 0; i < connector->count_modes; i++) {
            drmModeModeInfo *mode = &connector->modes[i];

            if (mode->type & DRM_MODE_TYPE_PREFERRED) {
                mode_index = i;
                break;
            }

            current_area = mode->hdisplay * mode->vdisplay;
            if (current_area > largest_area) {
                mode_index = i;
                largest_area = current_area;
            }
        }
        if (mode_index != -1) {
            crtc->mode = connector->modes[mode_index];
        }
    }

    if (mode_index == -1) {
        ret = SDL_SetError("Failed to find index of mode attached to the CRTC.");
        goto cleanup;
    }

    /*********************************************/
    /* Create an SDL Display for this connector. */
    /*********************************************/

    /*********************************************/
    /* Part 1: setup the SDL_Display driverdata. */
    /*********************************************/

    /* Get the mode currently setup for this display,
       which is the mode currently setup on the CRTC
       we found for the active connector. */
    dispdata->mode = crtc->mode;
    dispdata->original_mode = crtc->mode;
    dispdata->fullscreen_mode = crtc->mode;

    if (dispdata->mode.hdisplay == 0 || dispdata->mode.vdisplay == 0) {
        ret = SDL_SetError("Couldn't get a valid connector videomode.");
        goto cleanup;
    }

    /* Store the connector and crtc for this display. */
    dispdata->connector = connector;
    dispdata->crtc = crtc;

    /* save previous vrr state */
    dispdata->saved_vrr = KMSDRM_CrtcGetVrr(viddata->drm_fd, crtc->crtc_id);
    /* try to enable vrr */
    if (KMSDRM_ConnectorCheckVrrCapable(viddata->drm_fd, connector->connector_id, "VRR_CAPABLE")) {
        SDL_LogDebug(SDL_LOG_CATEGORY_VIDEO, "Enabling VRR");
        KMSDRM_CrtcSetVrr(viddata->drm_fd, crtc->crtc_id, SDL_TRUE);
    }

    /*****************************************/
    /* Part 2: setup the SDL_Display itself. */
    /*****************************************/

    /* Setup the display.
       There's no problem with it being still incomplete. */
    modedata = SDL_calloc(1, sizeof(SDL_DisplayModeData));

    if (!modedata) {
        ret = SDL_OutOfMemory();
        goto cleanup;
    }

    modedata->mode_index = mode_index;

    display.driverdata = dispdata;
    display.desktop_mode.w = dispdata->mode.hdisplay;
    display.desktop_mode.h = dispdata->mode.vdisplay;
    display.desktop_mode.refresh_rate = dispdata->mode.vrefresh;
    display.desktop_mode.format = SDL_PIXELFORMAT_ARGB8888;
    display.desktop_mode.driverdata = modedata;
    display.current_mode = display.desktop_mode;

    /* Add the display to the list of SDL displays. */
    SDL_AddVideoDisplay(&display, SDL_FALSE);

cleanup:
    if (encoder) {
        KMSDRM_drmModeFreeEncoder(encoder);
    }
    if (ret) {
        /* Error (complete) cleanup */
        if (dispdata) {
            if (dispdata->connector) {
                KMSDRM_drmModeFreeConnector(dispdata->connector);
                dispdata->connector = NULL;
            }
            if (dispdata->crtc) {
                KMSDRM_drmModeFreeCrtc(dispdata->crtc);
                dispdata->crtc = NULL;
            }
            SDL_free(dispdata);
        }
    }
} /* NOLINT(clang-analyzer-unix.Malloc): If no error `dispdata` is saved in the display */

/* Initializes the list of SDL displays: we build a new display for each
   connecter connector we find.
   This is to be called early, in VideoInit(), because it gets us
   the videomode information, which SDL needs immediately after VideoInit(). */
static int KMSDRM_InitDisplays(_THIS)
{

    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);
    drmModeRes *resources = NULL;

    uint64_t async_pageflip = 0;
    int ret = 0;
    int i;

    /* Open /dev/dri/cardNN (/dev/drmN if on OpenBSD version less than 6.9) */
    (void)SDL_snprintf(viddata->devpath, sizeof(viddata->devpath), "%s%d",
                       kmsdrm_dri_cardpath, viddata->devindex);

    SDL_LogDebug(SDL_LOG_CATEGORY_VIDEO, "Opening device %s", viddata->devpath);
    viddata->drm_fd = open(viddata->devpath, O_RDWR | O_CLOEXEC);

    if (viddata->drm_fd < 0) {
        ret = SDL_SetError("Could not open %s", viddata->devpath);
        goto cleanup;
    }

    SDL_LogDebug(SDL_LOG_CATEGORY_VIDEO, "Opened DRM FD (%d)", viddata->drm_fd);

    /* Get all of the available connectors / devices / crtcs */
    resources = KMSDRM_drmModeGetResources(viddata->drm_fd);
    if (!resources) {
        ret = SDL_SetError("drmModeGetResources(%d) failed", viddata->drm_fd);
        goto cleanup;
    }

    /* Iterate on the available connectors. For every connected connector,
       we create an SDL_Display and add it to the list of SDL Displays. */
    for (i = 0; i < resources->count_connectors; i++) {
        drmModeConnector *connector = KMSDRM_drmModeGetConnector(viddata->drm_fd,
                                                                 resources->connectors[i]);

        if (!connector) {
            continue;
        }

        if (connector->connection == DRM_MODE_CONNECTED && connector->count_modes) {
            /* If it's a connected connector with available videomodes, try to add
               an SDL Display representing it. KMSDRM_AddDisplay() is purposely void,
               so if it fails (no encoder for connector, no valid video mode for
               connector etc...) we can keep looking for connected connectors. */
            KMSDRM_AddDisplay(_this, connector, resources);
        } else {
            /* If it's not, free it now. */
            KMSDRM_drmModeFreeConnector(connector);
        }
    }

    /* Have we added any SDL displays? */
    if (!SDL_GetNumVideoDisplays()) {
        ret = SDL_SetError("No connected displays found.");
        goto cleanup;
    }

    /* Determine if video hardware supports async pageflips. */
    ret = KMSDRM_drmGetCap(viddata->drm_fd, DRM_CAP_ASYNC_PAGE_FLIP, &async_pageflip);
    if (ret) {
        SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "Could not determine async page flip capability.");
    }
    viddata->async_pageflip_support = async_pageflip ? SDL_TRUE : SDL_FALSE;

    /***********************************/
    /* Block for Vulkan compatibility. */
    /***********************************/

    /* Vulkan requires DRM master on its own FD to work, so try to drop master
       on our FD. This will only work without root on kernels v5.8 and later.
       If it doesn't work, just close the FD and we'll reopen it later. */
    if (!KMSDRM_DropMaster(_this)) {
        close(viddata->drm_fd);
        viddata->drm_fd = -1;
    }

cleanup:
    if (resources) {
        KMSDRM_drmModeFreeResources(resources);
    }
    if (ret) {
        if (viddata->drm_fd >= 0) {
            close(viddata->drm_fd);
            viddata->drm_fd = -1;
        }
    }
    return ret;
}

/* Init the Vulkan-INCOMPATIBLE stuff:
   Reopen FD, create gbm dev, create dumb buffer and setup display plane.
   This is to be called late, in WindowCreate(), and ONLY if this is not
   a Vulkan window.
   We are doing this so late to allow Vulkan to work if we build a VK window.
   These things are incompatible with Vulkan, which accesses the same resources
   internally so they must be free when trying to build a Vulkan surface.
*/
static int KMSDRM_GBMInit(_THIS, SDL_DisplayData *dispdata)
{
    SDL_VideoData *viddata = (SDL_VideoData *)_this->driverdata;
    int ret = 0;

    /* Reopen the FD if we weren't able to drop master on the original one */
    if (viddata->drm_fd < 0) {
        viddata->drm_fd = open(viddata->devpath, O_RDWR | O_CLOEXEC);
        if (viddata->drm_fd < 0) {
            return SDL_SetError("Could not reopen %s", viddata->devpath);
        }
    }

    /* Set the FD as current DRM master. */
    KMSDRM_drmSetMaster(viddata->drm_fd);

    /* Create the GBM device. */
    viddata->gbm_dev = KMSDRM_gbm_create_device(viddata->drm_fd);
    if (!viddata->gbm_dev) {
        ret = SDL_SetError("Couldn't create gbm device.");
    }

    viddata->gbm_init = SDL_TRUE;

    return ret;
}

/* Deinit the Vulkan-incompatible KMSDRM stuff. */
static void KMSDRM_GBMDeinit(_THIS, SDL_DisplayData *dispdata)
{
    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);

    /* Destroy GBM device. GBM surface is destroyed by DestroySurfaces(),
       already called when we get here. */
    if (viddata->gbm_dev) {
        KMSDRM_gbm_device_destroy(viddata->gbm_dev);
        viddata->gbm_dev = NULL;
    }

    /* Finally drop DRM master if possible, otherwise close DRM FD.
       May be reopened on next non-vulkan window creation. */
    if (viddata->drm_fd >= 0 && !KMSDRM_DropMaster(_this)) {
        close(viddata->drm_fd);
        viddata->drm_fd = -1;
    }

    viddata->gbm_init = SDL_FALSE;
}

static void KMSDRM_DestroySurfaces(_THIS, SDL_Window *window)
{
    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);
    SDL_WindowData *windata = (SDL_WindowData *)window->driverdata;
    SDL_DisplayData *dispdata = (SDL_DisplayData *)SDL_GetDisplayForWindow(window)->driverdata;
    int ret;

    /**********************************************/
    /* Wait for last issued pageflip to complete. */
    /**********************************************/
    /*KMSDRM_WaitPageflip(_this, windata);*/

    /***********************************************************************/
    /* Restore the original CRTC configuration: configue the crtc with the */
    /* original video mode and make it point to the original TTY buffer.   */
    /***********************************************************************/

    ret = KMSDRM_drmModeSetCrtc(viddata->drm_fd, dispdata->crtc->crtc_id,
                                dispdata->crtc->buffer_id, 0, 0, &dispdata->connector->connector_id, 1,
                                &dispdata->original_mode);

    /* If we failed to set the original mode, try to set the connector prefered mode. */
    if (ret && (dispdata->crtc->mode_valid == 0)) {
        ret = KMSDRM_drmModeSetCrtc(viddata->drm_fd, dispdata->crtc->crtc_id,
                                    dispdata->crtc->buffer_id, 0, 0, &dispdata->connector->connector_id, 1,
                                    &dispdata->original_mode);
    }

    if (ret) {
        SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "Could not restore CRTC");
    }

    /***************************/
    /* Destroy the EGL surface */
    /***************************/

    SDL_EGL_MakeCurrent(_this, EGL_NO_SURFACE, EGL_NO_CONTEXT);

    if (windata->egl_surface != EGL_NO_SURFACE) {
        SDL_EGL_DestroySurface(_this, windata->egl_surface);
        windata->egl_surface = EGL_NO_SURFACE;
    }

    /***************************/
    /* Destroy the GBM buffers */
    /***************************/

    if (windata->bo) {
        KMSDRM_gbm_surface_release_buffer(windata->gs, windata->bo);
        windata->bo = NULL;
    }

    if (windata->next_bo) {
        KMSDRM_gbm_surface_release_buffer(windata->gs, windata->next_bo);
        windata->next_bo = NULL;
    }

    /***************************/
    /* Destroy the GBM surface */
    /***************************/

    if (windata->gs) {
        KMSDRM_gbm_surface_destroy(windata->gs);
        windata->gs = NULL;
    }
}

static void KMSDRM_GetModeToSet(SDL_Window *window, drmModeModeInfo *out_mode)
{
    SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);
    SDL_DisplayData *dispdata = (SDL_DisplayData *)display->driverdata;

    if ((window->flags & SDL_WINDOW_FULLSCREEN) == SDL_WINDOW_FULLSCREEN) {
        *out_mode = dispdata->fullscreen_mode;
    } else {
        drmModeModeInfo *mode;

        mode = KMSDRM_GetClosestDisplayMode(display,
                                            window->windowed.w, window->windowed.h, 0);

        if (mode) {
            *out_mode = *mode;
        } else {
            *out_mode = dispdata->original_mode;
        }
    }
}

static void KMSDRM_DirtySurfaces(SDL_Window *window)
{
    SDL_WindowData *windata = (SDL_WindowData *)window->driverdata;
    drmModeModeInfo mode;

    /* Can't recreate EGL surfaces right now, need to wait until SwapWindow
       so the correct thread-local surface and context state are available */
    windata->egl_surface_dirty = SDL_TRUE;

    /* The app may be waiting for the resize event after calling SetWindowSize
       or SetWindowFullscreen, send a fake event for now since the actual
       recreation is deferred */
    KMSDRM_GetModeToSet(window, &mode);
    SDL_SendWindowEvent(window, SDL_WINDOWEVENT_RESIZED, mode.hdisplay, mode.vdisplay);
}

/* This determines the size of the fb, which comes from the GBM surface
   that we create here. */
int KMSDRM_CreateSurfaces(_THIS, SDL_Window *window)
{
    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);
    SDL_WindowData *windata = (SDL_WindowData *)window->driverdata;
    SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);
    SDL_DisplayData *dispdata = (SDL_DisplayData *)display->driverdata;

    uint32_t surface_fmt = GBM_FORMAT_ARGB8888;
    uint32_t surface_flags = GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING;

    EGLContext egl_context;

    int ret = 0;

    /* If the current window already has surfaces, destroy them before creating other. */
    if (windata->gs) {
        KMSDRM_DestroySurfaces(_this, window);
    }

    if (!KMSDRM_gbm_device_is_format_supported(viddata->gbm_dev,
                                               surface_fmt, surface_flags)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_VIDEO,
                    "GBM surface format not supported. Trying anyway.");
    }

    /* The KMSDRM backend doesn't always set the mode the higher-level code in
       SDL_video.c expects. Hulk-smash the display's current_mode to keep the
       mode that's set in sync with what SDL_video.c thinks is set */
    KMSDRM_GetModeToSet(window, &dispdata->mode);

    display->current_mode.w = dispdata->mode.hdisplay;
    display->current_mode.h = dispdata->mode.vdisplay;
    display->current_mode.refresh_rate = dispdata->mode.vrefresh;
    display->current_mode.format = SDL_PIXELFORMAT_ARGB8888;

    windata->gs = KMSDRM_gbm_surface_create(viddata->gbm_dev,
                                            dispdata->mode.hdisplay, dispdata->mode.vdisplay,
                                            surface_fmt, surface_flags);

    if (!windata->gs) {
        return SDL_SetError("Could not create GBM surface");
    }

    /* We can't get the EGL context yet because SDL_CreateRenderer has not been called,
       but we need an EGL surface NOW, or GL won't be able to render into any surface
       and we won't see the first frame. */
    SDL_EGL_SetRequiredVisualId(_this, surface_fmt);
    windata->egl_surface = SDL_EGL_CreateSurface(_this, (NativeWindowType)windata->gs);

    if (windata->egl_surface == EGL_NO_SURFACE) {
        ret = SDL_SetError("Could not create EGL window surface");
        goto cleanup;
    }

    /* Current context passing to EGL is now done here. If something fails,
       go back to delayed SDL_EGL_MakeCurrent() call in SwapWindow. */
    egl_context = (EGLContext)SDL_GL_GetCurrentContext();
    ret = SDL_EGL_MakeCurrent(_this, windata->egl_surface, egl_context);

    SDL_SendWindowEvent(window, SDL_WINDOWEVENT_RESIZED,
                        dispdata->mode.hdisplay, dispdata->mode.vdisplay);

    windata->egl_surface_dirty = SDL_FALSE;

cleanup:

    if (ret) {
        /* Error (complete) cleanup. */
        if (windata->gs) {
            KMSDRM_gbm_surface_destroy(windata->gs);
            windata->gs = NULL;
        }
    }

    return ret;
}

#ifdef SDL_INPUT_LINUXEV
static void KMSDRM_ReleaseVT(void *userdata)
{
    SDL_VideoDevice *_this = (SDL_VideoDevice *)userdata;
    SDL_VideoData *viddata = _this->driverdata;
    int i;

    for (i = 0; i < viddata->num_windows; i++) {
        SDL_Window *window = viddata->windows[i];
        if (!(window->flags & SDL_WINDOW_VULKAN)) {
            KMSDRM_DestroySurfaces(_this, window);
        }
    }
    KMSDRM_drmDropMaster(viddata->drm_fd);
}

static void KMSDRM_AcquireVT(void *userdata)
{
    SDL_VideoDevice *_this = (SDL_VideoDevice *)userdata;
    SDL_VideoData *viddata = _this->driverdata;
    int i;

    KMSDRM_drmSetMaster(viddata->drm_fd);
    for (i = 0; i < viddata->num_windows; i++) {
        SDL_Window *window = viddata->windows[i];
        if (!(window->flags & SDL_WINDOW_VULKAN)) {
            KMSDRM_CreateSurfaces(_this, window);
        }
    }
}
#endif /* defined SDL_INPUT_LINUXEV */

int KMSDRM_VideoInit(_THIS)
{
    int ret = 0;

    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);
    SDL_LogDebug(SDL_LOG_CATEGORY_VIDEO, "KMSDRM_VideoInit()");

    viddata->video_init = SDL_FALSE;
    viddata->gbm_init = SDL_FALSE;

    /* Get KMSDRM resources info and store what we need. Getting and storing
       this info isn't a problem for VK compatibility.
       For VK-incompatible initializations we have KMSDRM_GBMInit(), which is
       called on window creation, and only when we know it's not a VK window. */
    if (KMSDRM_InitDisplays(_this)) {
        ret = SDL_SetError("error getting KMSDRM displays information");
    }

#ifdef SDL_INPUT_LINUXEV
    SDL_EVDEV_Init();
    SDL_EVDEV_SetVTSwitchCallbacks(KMSDRM_ReleaseVT, _this, KMSDRM_AcquireVT, _this);
#elif defined(SDL_INPUT_WSCONS)
    SDL_WSCONS_Init();
#endif

    viddata->video_init = SDL_TRUE;

    return ret;
}

/* The driverdata pointers, like dispdata, viddata, windata, etc...
   are freed by SDL internals, so not our job. */
void KMSDRM_VideoQuit(_THIS)
{
    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);

    KMSDRM_DeinitDisplays(_this);

#ifdef SDL_INPUT_LINUXEV
    SDL_EVDEV_SetVTSwitchCallbacks(NULL, NULL, NULL, NULL);
    SDL_EVDEV_Quit();
#elif defined(SDL_INPUT_WSCONS)
    SDL_WSCONS_Quit();
#endif

    /* Clear out the window list */
    SDL_free(viddata->windows);
    viddata->windows = NULL;
    viddata->max_windows = 0;
    viddata->num_windows = 0;
    viddata->video_init = SDL_FALSE;
}

/* Read modes from the connector modes, and store them in display->display_modes. */
void KMSDRM_GetDisplayModes(_THIS, SDL_VideoDisplay *display)
{
    SDL_DisplayData *dispdata = display->driverdata;
    drmModeConnector *conn = dispdata->connector;
    SDL_DisplayMode mode;
    int i;

    for (i = 0; i < conn->count_modes; i++) {
        SDL_DisplayModeData *modedata = SDL_calloc(1, sizeof(SDL_DisplayModeData));

        if (modedata) {
            modedata->mode_index = i;
        }

        mode.w = conn->modes[i].hdisplay;
        mode.h = conn->modes[i].vdisplay;
        mode.refresh_rate = conn->modes[i].vrefresh;
        mode.format = SDL_PIXELFORMAT_ARGB8888;
        mode.driverdata = modedata;

        if (!SDL_AddDisplayMode(display, &mode)) {
            SDL_free(modedata);
        }
    }
}

int KMSDRM_SetDisplayMode(_THIS, SDL_VideoDisplay *display, SDL_DisplayMode *mode)
{
    /* Set the dispdata->mode to the new mode and leave actual modesetting
       pending to be done on SwapWindow() via drmModeSetCrtc() */

    SDL_VideoData *viddata = (SDL_VideoData *)_this->driverdata;
    SDL_DisplayData *dispdata = (SDL_DisplayData *)display->driverdata;
    SDL_DisplayModeData *modedata = (SDL_DisplayModeData *)mode->driverdata;
    drmModeConnector *conn = dispdata->connector;
    int i;

    /* Don't do anything if we are in Vulkan mode. */
    if (viddata->vulkan_mode) {
        return 0;
    }

    if (!modedata) {
        return SDL_SetError("Mode doesn't have an associated index");
    }

    /* Take note of the new mode to be set, and leave the CRTC modeset pending
       so it's done in SwapWindow. */
    dispdata->fullscreen_mode = conn->modes[modedata->mode_index];

    for (i = 0; i < viddata->num_windows; i++) {
        KMSDRM_DirtySurfaces(viddata->windows[i]);
    }

    return 0;
}

void KMSDRM_DestroyWindow(_THIS, SDL_Window *window)
{
    SDL_WindowData *windata = (SDL_WindowData *)window->driverdata;
    SDL_DisplayData *dispdata = (SDL_DisplayData *)SDL_GetDisplayForWindow(window)->driverdata;
    SDL_VideoData *viddata;
    SDL_bool is_vulkan = window->flags & SDL_WINDOW_VULKAN; /* Is this a VK window? */
    unsigned int i, j;

    if (!windata) {
        return;
    }

    /* restore vrr state */
    KMSDRM_CrtcSetVrr(windata->viddata->drm_fd, dispdata->crtc->crtc_id, dispdata->saved_vrr);

    viddata = windata->viddata;

    if (!is_vulkan && viddata->gbm_init) {

        /* Destroy cursor GBM BO of the display of this window. */
        KMSDRM_DestroyCursorBO(_this, SDL_GetDisplayForWindow(window));

        /* Destroy GBM surface and buffers. */
        KMSDRM_DestroySurfaces(_this, window);

        /* Unload library and deinit GBM, but only if this is the last window.
           Note that this is the right comparision because num_windows could be 1
           if there is a complete window, or 0 if we got here from SDL_CreateWindow()
           because KMSDRM_CreateWindow() returned an error so the window wasn't
           added to the windows list. */
        if (viddata->num_windows <= 1) {

            /* Unload EGL/GL library and free egl_data.  */
            if (_this->egl_data) {
                SDL_EGL_UnloadLibrary(_this);
                _this->gl_config.driver_loaded = 0;
            }

            /* Free display plane, and destroy GBM device. */
            KMSDRM_GBMDeinit(_this, dispdata);
        }

    } else {

        /* If we were in Vulkan mode, get out of it. */
        if (viddata->vulkan_mode) {
            viddata->vulkan_mode = SDL_FALSE;
        }
    }

    /********************************************/
    /* Remove from the internal SDL window list */
    /********************************************/

    for (i = 0; i < viddata->num_windows; i++) {
        if (viddata->windows[i] == window) {
            viddata->num_windows--;

            for (j = i; j < viddata->num_windows; j++) {
                viddata->windows[j] = viddata->windows[j + 1];
            }

            break;
        }
    }

    /*********************************************************************/
    /* Free the window driverdata. Bye bye, surface and buffer pointers! */
    /*********************************************************************/
    SDL_free(window->driverdata);
    window->driverdata = NULL;
}

/**********************************************************************/
/* We simply IGNORE if it's a fullscreen window, window->flags don't  */
/* reflect it: if it's fullscreen, KMSDRM_SetWindwoFullscreen() will  */
/* be called by SDL later, and we can manage it there.                */
/**********************************************************************/
int KMSDRM_CreateWindow(_THIS, SDL_Window *window)
{
    SDL_WindowData *windata = NULL;
    SDL_VideoData *viddata = (SDL_VideoData *)_this->driverdata;
    SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);
    SDL_DisplayData *dispdata = display->driverdata;
    SDL_bool is_vulkan = window->flags & SDL_WINDOW_VULKAN; /* Is this a VK window? */
    SDL_bool vulkan_mode = viddata->vulkan_mode;            /* Do we have any Vulkan windows? */
    NativeDisplayType egl_display;
    drmModeModeInfo *mode;
    int ret = 0;

    /* Allocate window internal data */
    windata = (SDL_WindowData *)SDL_calloc(1, sizeof(SDL_WindowData));
    if (!windata) {
        return SDL_OutOfMemory();
    }

    /* Setup driver data for this window */
    windata->viddata = viddata;
    window->driverdata = windata;

    /* Do we want a double buffering scheme to get low video lag? */
    windata->double_buffer = SDL_FALSE;
    if (SDL_GetHintBoolean(SDL_HINT_VIDEO_DOUBLE_BUFFER, SDL_FALSE)) {
        windata->double_buffer = SDL_TRUE;
    }

    if (!is_vulkan && !vulkan_mode) { /* NON-Vulkan block. */

        /* Maybe you didn't ask for an OPENGL window, but that's what you will get.
           See following comments on why. */
        window->flags |= SDL_WINDOW_OPENGL;

        if (!(viddata->gbm_init)) {

            /* After SDL_CreateWindow, most SDL2 programs will do SDL_CreateRenderer(),
               which will in turn call GL_CreateRenderer() or GLES2_CreateRenderer().
               In order for the GL_CreateRenderer() or GLES2_CreateRenderer() call to
               succeed without an unnecessary window re-creation, we must:
               -Mark the window as being OPENGL
               -Load the GL library (which can't be done until the GBM device has been
                created, so we have to do it here instead of doing it on VideoInit())
                and mark it as loaded by setting gl_config.driver_loaded to 1.
               So if you ever see KMSDRM_CreateWindow() to be called two times in tests,
               don't be shy to debug GL_CreateRenderer() or GLES2_CreateRenderer()
               to find out why!
             */

            /* Reopen FD, create gbm dev, setup display plane, etc,.
               but only when we come here for the first time,
               and only if it's not a VK window. */
            ret = KMSDRM_GBMInit(_this, dispdata);
            if (ret != 0) {
                return SDL_SetError("Can't init GBM on window creation.");
            }
        }

        /* Manually load the GL library. KMSDRM_EGL_LoadLibrary() has already
           been called by SDL_CreateWindow() but we don't do anything there,
           our KMSDRM_EGL_LoadLibrary() is a dummy precisely to be able to load it here.
           If we let SDL_CreateWindow() load the lib, it would be loaded
           before we call KMSDRM_GBMInit(), causing all GLES programs to fail. */
        if (!_this->egl_data) {
            egl_display = (NativeDisplayType)((SDL_VideoData *)_this->driverdata)->gbm_dev;
            if (SDL_EGL_LoadLibrary(_this, NULL, egl_display, EGL_PLATFORM_GBM_MESA) < 0) {
                /* Try again with OpenGL ES 2.0 */
                _this->gl_config.profile_mask = SDL_GL_CONTEXT_PROFILE_ES;
                _this->gl_config.major_version = 2;
                _this->gl_config.minor_version = 0;
                if (SDL_EGL_LoadLibrary(_this, NULL, egl_display, EGL_PLATFORM_GBM_MESA) < 0) {
                    return SDL_SetError("Can't load EGL/GL library on window creation.");
                }
            }

            _this->gl_config.driver_loaded = 1;
        }

        /* Create the cursor BO for the display of this window,
           now that we know this is not a VK window. */
        KMSDRM_CreateCursorBO(display);

        /* Create and set the default cursor for the display
           of this window, now that we know this is not a VK window. */
        KMSDRM_InitMouse(_this, display);

        /* The FULLSCREEN flags are cut out from window->flags at this point,
           so we can't know if a window is fullscreen or not, hence all windows
           are considered "windowed" at this point of their life.
           If a window is fullscreen, SDL internals will call
           KMSDRM_SetWindowFullscreen() to reconfigure it if necessary. */
        mode = KMSDRM_GetClosestDisplayMode(display,
                                            window->windowed.w, window->windowed.h, 0);

        if (mode) {
            dispdata->fullscreen_mode = *mode;
        } else {
            dispdata->fullscreen_mode = dispdata->original_mode;
        }

        /* Create the window surfaces with the size we have just chosen.
           Needs the window diverdata in place. */
        ret = KMSDRM_CreateSurfaces(_this, window);
        if (ret != 0) {
            return SDL_SetError("Can't window GBM/EGL surfaces on window creation.");
        }
    } /* NON-Vulkan block ends. */

    /* Add window to the internal list of tracked windows. Note, while it may
       seem odd to support multiple fullscreen windows, some apps create an
       extra window as a dummy surface when working with multiple contexts */
    if (viddata->num_windows >= viddata->max_windows) {
        unsigned int new_max_windows = viddata->max_windows + 1;
        viddata->windows = (SDL_Window **)SDL_realloc(viddata->windows,
                                                      new_max_windows * sizeof(SDL_Window *));
        viddata->max_windows = new_max_windows;

        if (!viddata->windows) {
            return SDL_OutOfMemory();
        }
    }

    viddata->windows[viddata->num_windows++] = window;

    /* If we have just created a Vulkan window, establish that we are in Vulkan mode now. */
    viddata->vulkan_mode = is_vulkan;

    /* Focus on the newly created window.
       SDL_SetMouseFocus() also takes care of calling KMSDRM_ShowCursor() if necessary. */
    SDL_SetMouseFocus(window);
    SDL_SetKeyboardFocus(window);

    /* Tell the app that the window has moved to top-left. */
    {
        SDL_Rect display_bounds;
        SDL_zero(display_bounds);
        SDL_GetDisplayBounds(SDL_GetWindowDisplayIndex(window), &display_bounds);
        SDL_SendWindowEvent(window, SDL_WINDOWEVENT_MOVED, display_bounds.x, display_bounds.y);
    }

    /* Allocated windata will be freed in KMSDRM_DestroyWindow,
       and KMSDRM_DestroyWindow() will be called by SDL_CreateWindow()
       if we return error on any of the previous returns of the function. */
    return ret;
}

int KMSDRM_GetWindowGammaRamp(_THIS, SDL_Window * window, Uint16 * ramp)
{
    SDL_WindowData *windata = (SDL_WindowData*)window->driverdata;
    SDL_VideoData *viddata = (SDL_VideoData*)windata->viddata;
    SDL_VideoDisplay *disp = SDL_GetDisplayForWindow(window);
    SDL_DisplayData* dispdata = (SDL_DisplayData*)disp->driverdata;
    if (KMSDRM_drmModeCrtcGetGamma(viddata->drm_fd, dispdata->crtc->crtc_id, 256, &ramp[0*256], &ramp[1*256], &ramp[2*256]) == -1)
    {
        return SDL_SetError("Failed to get gamma ramp");
    }
    return 0;
}

int KMSDRM_SetWindowGammaRamp(_THIS, SDL_Window * window, const Uint16 * ramp)
{
    SDL_WindowData *windata = (SDL_WindowData*)window->driverdata;
    SDL_VideoData *viddata = (SDL_VideoData*)windata->viddata;
    SDL_VideoDisplay *disp = SDL_GetDisplayForWindow(window);
    SDL_DisplayData* dispdata = (SDL_DisplayData*)disp->driverdata;
    Uint16* tempRamp = SDL_calloc(3 * sizeof(Uint16), 256);
    if (!tempRamp)
    {
        return SDL_OutOfMemory();
    }
    SDL_memcpy(tempRamp, ramp, 3 * sizeof(Uint16) * 256);
    if (KMSDRM_drmModeCrtcSetGamma(viddata->drm_fd, dispdata->crtc->crtc_id, 256, &tempRamp[0*256], &tempRamp[1*256], &tempRamp[2*256]) == -1)
    {
        SDL_free(tempRamp);
        return SDL_SetError("Failed to set gamma ramp");
    }
    SDL_free(tempRamp);
    return 0;
}

int KMSDRM_CreateWindowFrom(_THIS, SDL_Window * window, const void *data)
{
    return -1;
}

void KMSDRM_SetWindowTitle(_THIS, SDL_Window *window)
{
}
void KMSDRM_SetWindowIcon(_THIS, SDL_Window *window, SDL_Surface *icon)
{
}
void KMSDRM_SetWindowPosition(_THIS, SDL_Window *window)
{
}
void KMSDRM_SetWindowSize(_THIS, SDL_Window *window)
{
    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);
    if (!viddata->vulkan_mode) {
        KMSDRM_DirtySurfaces(window);
    }
}
void KMSDRM_SetWindowFullscreen(_THIS, SDL_Window *window, SDL_VideoDisplay *display, SDL_bool fullscreen)

{
    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);
    if (!viddata->vulkan_mode) {
        KMSDRM_DirtySurfaces(window);
    }
}
void KMSDRM_ShowWindow(_THIS, SDL_Window *window)
{
}
void KMSDRM_HideWindow(_THIS, SDL_Window *window)
{
}
void KMSDRM_RaiseWindow(_THIS, SDL_Window *window)
{
}
void KMSDRM_MaximizeWindow(_THIS, SDL_Window *window)
{
}
void KMSDRM_MinimizeWindow(_THIS, SDL_Window *window)
{
}
void KMSDRM_RestoreWindow(_THIS, SDL_Window *window)
{
}

/*****************************************************************************/
/* SDL Window Manager function                                               */
/*****************************************************************************/
SDL_bool KMSDRM_GetWindowWMInfo(_THIS, SDL_Window * window, struct SDL_SysWMinfo *info)
{
     SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);
     const Uint32 version = SDL_VERSIONNUM((Uint32)info->version.major,
                                           (Uint32)info->version.minor,
                                           (Uint32)info->version.patch);

     if (version < SDL_VERSIONNUM(2, 0, 15)) {
         SDL_SetError("Version must be 2.0.15 or newer");
         return SDL_FALSE;
     }

    info->subsystem = SDL_SYSWM_KMSDRM;
    info->info.kmsdrm.dev_index = viddata->devindex;
    info->info.kmsdrm.drm_fd = viddata->drm_fd;
    info->info.kmsdrm.gbm_dev = viddata->gbm_dev;

     return SDL_TRUE;
}

#endif /* SDL_VIDEO_DRIVER_KMSDRM */

/* vi: set ts=4 sw=4 expandtab: */
