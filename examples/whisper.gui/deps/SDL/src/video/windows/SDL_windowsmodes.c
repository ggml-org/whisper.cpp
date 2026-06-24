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

#if defined(SDL_VIDEO_DRIVER_WINDOWS) && !defined(__XBOXONE__) && !defined(__XBOXSERIES__)

#include "SDL_windowsvideo.h"
#include "../../events/SDL_displayevents_c.h"

/* Windows CE compatibility */
#ifndef CDS_FULLSCREEN
#define CDS_FULLSCREEN 0
#endif

/* #define DEBUG_MODES */
/* #define HIGHDPI_DEBUG_VERBOSE */

static void WIN_UpdateDisplayMode(_THIS, LPCWSTR deviceName, DWORD index, SDL_DisplayMode *mode)
{
    SDL_DisplayModeData *data = (SDL_DisplayModeData *)mode->driverdata;
    HDC hdc;

    data->DeviceMode.dmFields =
        (DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY |
         DM_DISPLAYFLAGS);

    /* NOLINTNEXTLINE(bugprone-assignment-in-if-condition): No simple way to extract the assignment */
    if (index == ENUM_CURRENT_SETTINGS && (hdc = CreateDC(deviceName, NULL, NULL, NULL)) != NULL) {
        char bmi_data[sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD)];
        LPBITMAPINFO bmi;
        HBITMAP hbm;
        int logical_width = GetDeviceCaps(hdc, HORZRES);
        int logical_height = GetDeviceCaps(hdc, VERTRES);

        /* High-DPI notes:

           If DPI-unaware:
             - GetDeviceCaps( hdc, HORZRES ) will return the monitor width in points.
             - DeviceMode.dmPelsWidth is actual pixels (unlike almost all other Windows API's,
               it's not virtualized when DPI unaware).

           If DPI-aware:
            - GetDeviceCaps( hdc, HORZRES ) will return pixels, same as DeviceMode.dmPelsWidth */
        mode->w = logical_width;
        mode->h = logical_height;

        SDL_zeroa(bmi_data);
        bmi = (LPBITMAPINFO)bmi_data;
        bmi->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);

        hbm = CreateCompatibleBitmap(hdc, 1, 1);
        GetDIBits(hdc, hbm, 0, 1, NULL, bmi, DIB_RGB_COLORS);
        GetDIBits(hdc, hbm, 0, 1, NULL, bmi, DIB_RGB_COLORS);
        DeleteObject(hbm);
        DeleteDC(hdc);
        if (bmi->bmiHeader.biCompression == BI_BITFIELDS) {
            switch (*(Uint32 *)bmi->bmiColors) {
            case 0x00FF0000:
                mode->format = SDL_PIXELFORMAT_RGB888;
                break;
            case 0x000000FF:
                mode->format = SDL_PIXELFORMAT_BGR888;
                break;
            case 0xF800:
                mode->format = SDL_PIXELFORMAT_RGB565;
                break;
            case 0x7C00:
                mode->format = SDL_PIXELFORMAT_RGB555;
                break;
            }
        } else if (bmi->bmiHeader.biBitCount == 8) {
            mode->format = SDL_PIXELFORMAT_INDEX8;
        } else if (bmi->bmiHeader.biBitCount == 4) {
            mode->format = SDL_PIXELFORMAT_INDEX4LSB;
        }
    } else if (mode->format == SDL_PIXELFORMAT_UNKNOWN) {
        /* FIXME: Can we tell what this will be? */
        if ((data->DeviceMode.dmFields & DM_BITSPERPEL) == DM_BITSPERPEL) {
            switch (data->DeviceMode.dmBitsPerPel) {
            case 32:
                mode->format = SDL_PIXELFORMAT_RGB888;
                break;
            case 24:
                mode->format = SDL_PIXELFORMAT_RGB24;
                break;
            case 16:
                mode->format = SDL_PIXELFORMAT_RGB565;
                break;
            case 15:
                mode->format = SDL_PIXELFORMAT_RGB555;
                break;
            case 8:
                mode->format = SDL_PIXELFORMAT_INDEX8;
                break;
            case 4:
                mode->format = SDL_PIXELFORMAT_INDEX4LSB;
                break;
            }
        }
    }
}

static SDL_DisplayOrientation WIN_GetDisplayOrientation(DEVMODE *mode)
{
    int width = mode->dmPelsWidth;
    int height = mode->dmPelsHeight;

    /* Use unrotated width/height to guess orientation */
    if (mode->dmDisplayOrientation == DMDO_90 || mode->dmDisplayOrientation == DMDO_270) {
        int temp = width;
        width = height;
        height = temp;
    }

    if (width >= height) {
        switch (mode->dmDisplayOrientation) {
        case DMDO_DEFAULT:
            return SDL_ORIENTATION_LANDSCAPE;
        case DMDO_90:
            return SDL_ORIENTATION_PORTRAIT;
        case DMDO_180:
            return SDL_ORIENTATION_LANDSCAPE_FLIPPED;
        case DMDO_270:
            return SDL_ORIENTATION_PORTRAIT_FLIPPED;
        default:
            return SDL_ORIENTATION_UNKNOWN;
        }
    } else {
        switch (mode->dmDisplayOrientation) {
        case DMDO_DEFAULT:
            return SDL_ORIENTATION_PORTRAIT;
        case DMDO_90:
            return SDL_ORIENTATION_LANDSCAPE_FLIPPED;
        case DMDO_180:
            return SDL_ORIENTATION_PORTRAIT_FLIPPED;
        case DMDO_270:
            return SDL_ORIENTATION_LANDSCAPE;
        default:
            return SDL_ORIENTATION_UNKNOWN;
        }
    }
}

static SDL_bool WIN_GetDisplayMode(_THIS, LPCWSTR deviceName, DWORD index, SDL_DisplayMode *mode, SDL_DisplayOrientation *orientation)
{
    SDL_DisplayModeData *data;
    DEVMODE devmode;

    devmode.dmSize = sizeof(devmode);
    devmode.dmDriverExtra = 0;
    if (!EnumDisplaySettingsW(deviceName, index, &devmode)) {
        return SDL_FALSE;
    }

    data = (SDL_DisplayModeData *)SDL_malloc(sizeof(*data));
    if (!data) {
        return SDL_FALSE;
    }

    mode->driverdata = data;
    data->DeviceMode = devmode;

    mode->format = SDL_PIXELFORMAT_UNKNOWN;
    mode->w = data->DeviceMode.dmPelsWidth;
    mode->h = data->DeviceMode.dmPelsHeight;
    mode->refresh_rate = data->DeviceMode.dmDisplayFrequency;

    /* Fill in the mode information */
    WIN_UpdateDisplayMode(_this, deviceName, index, mode);

    if (orientation) {
        *orientation = WIN_GetDisplayOrientation(&devmode);
    }

    return SDL_TRUE;
}

/* The win32 API calls in this function require Windows Vista or later. */
/* *INDENT-OFF* */ /* clang-format off */
typedef LONG (WINAPI *SDL_WIN32PROC_GetDisplayConfigBufferSizes)(UINT32 flags, UINT32* numPathArrayElements, UINT32* numModeInfoArrayElements);
typedef LONG (WINAPI *SDL_WIN32PROC_QueryDisplayConfig)(UINT32 flags, UINT32* numPathArrayElements, DISPLAYCONFIG_PATH_INFO* pathArray, UINT32* numModeInfoArrayElements, DISPLAYCONFIG_MODE_INFO* modeInfoArray, DISPLAYCONFIG_TOPOLOGY_ID* currentTopologyId);
typedef LONG (WINAPI *SDL_WIN32PROC_DisplayConfigGetDeviceInfo)(DISPLAYCONFIG_DEVICE_INFO_HEADER* requestPacket);
/* *INDENT-ON* */ /* clang-format on */

static char *WIN_GetDisplayNameVista(const WCHAR *deviceName)
{
    void *dll;
    SDL_WIN32PROC_GetDisplayConfigBufferSizes pGetDisplayConfigBufferSizes;
    SDL_WIN32PROC_QueryDisplayConfig pQueryDisplayConfig;
    SDL_WIN32PROC_DisplayConfigGetDeviceInfo pDisplayConfigGetDeviceInfo;
    DISPLAYCONFIG_PATH_INFO *paths = NULL;
    DISPLAYCONFIG_MODE_INFO *modes = NULL;
    char *retval = NULL;
    UINT32 pathCount = 0;
    UINT32 modeCount = 0;
    UINT32 i;
    LONG rc;

    dll = SDL_LoadObject("USER32.DLL");
    if (!dll) {
        return NULL;
    }

    pGetDisplayConfigBufferSizes = (SDL_WIN32PROC_GetDisplayConfigBufferSizes)SDL_LoadFunction(dll, "GetDisplayConfigBufferSizes");
    pQueryDisplayConfig = (SDL_WIN32PROC_QueryDisplayConfig)SDL_LoadFunction(dll, "QueryDisplayConfig");
    pDisplayConfigGetDeviceInfo = (SDL_WIN32PROC_DisplayConfigGetDeviceInfo)SDL_LoadFunction(dll, "DisplayConfigGetDeviceInfo");

    if (!pGetDisplayConfigBufferSizes || !pQueryDisplayConfig || !pDisplayConfigGetDeviceInfo) {
        goto WIN_GetDisplayNameVista_failed;
    }

    do {
        rc = pGetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &pathCount, &modeCount);
        if (rc != ERROR_SUCCESS) {
            goto WIN_GetDisplayNameVista_failed;
        }

        SDL_free(paths);
        SDL_free(modes);

        paths = (DISPLAYCONFIG_PATH_INFO *)SDL_malloc(sizeof(DISPLAYCONFIG_PATH_INFO) * pathCount);
        modes = (DISPLAYCONFIG_MODE_INFO *)SDL_malloc(sizeof(DISPLAYCONFIG_MODE_INFO) * modeCount);
        if ((!paths) || (!modes)) {
            goto WIN_GetDisplayNameVista_failed;
        }

        rc = pQueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &pathCount, paths, &modeCount, modes, 0);
    } while (rc == ERROR_INSUFFICIENT_BUFFER);

    if (rc == ERROR_SUCCESS) {
        for (i = 0; i < pathCount; i++) {
            DISPLAYCONFIG_SOURCE_DEVICE_NAME sourceName;
            DISPLAYCONFIG_TARGET_DEVICE_NAME targetName;

            SDL_zero(sourceName);
            sourceName.header.adapterId = paths[i].targetInfo.adapterId;
            sourceName.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
            sourceName.header.size = sizeof(sourceName);
            sourceName.header.id = paths[i].sourceInfo.id;
            rc = pDisplayConfigGetDeviceInfo(&sourceName.header);
            if (rc != ERROR_SUCCESS) {
                break;
            } else if (SDL_wcscmp(deviceName, sourceName.viewGdiDeviceName) != 0) {
                continue;
            }

            SDL_zero(targetName);
            targetName.header.adapterId = paths[i].targetInfo.adapterId;
            targetName.header.id = paths[i].targetInfo.id;
            targetName.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
            targetName.header.size = sizeof(targetName);
            rc = pDisplayConfigGetDeviceInfo(&targetName.header);
            if (rc == ERROR_SUCCESS) {
                retval = WIN_StringToUTF8W(targetName.monitorFriendlyDeviceName);
                /* if we got an empty string, treat it as failure so we'll fallback
                   to getting the generic name. */
                if (retval && (*retval == '\0')) {
                    SDL_free(retval);
                    retval = NULL;
                }
            }
            break;
        }
    }

    SDL_free(paths);
    SDL_free(modes);
    SDL_UnloadObject(dll);
    return retval;

WIN_GetDisplayNameVista_failed:
    SDL_free(retval);
    SDL_free(paths);
    SDL_free(modes);
    SDL_UnloadObject(dll);
    return NULL;
}

static void WIN_AddDisplay(_THIS, HMONITOR hMonitor, const MONITORINFOEXW *info, int *display_index, SDL_bool send_event)
{
    int i, index = *display_index;
    SDL_VideoDisplay display;
    SDL_DisplayData *displaydata;
    SDL_DisplayMode mode;
    SDL_DisplayOrientation orientation;

#ifdef DEBUG_MODES
    SDL_Log("Display: %s\n", WIN_StringToUTF8W(info->szDevice));
#endif

    if (!WIN_GetDisplayMode(_this, info->szDevice, ENUM_CURRENT_SETTINGS, &mode, &orientation)) {
        return;
    }

    // Prevent adding duplicate displays. Do this after we know the display is
    // ready to be added to allow any displays that we can't fully query to be
    // removed
    for (i = 0; i < _this->num_displays; ++i) {
        SDL_DisplayData *driverdata = (SDL_DisplayData *)_this->displays[i].driverdata;
        if (SDL_wcscmp(driverdata->DeviceName, info->szDevice) == 0) {
            SDL_bool moved = (index != i);

            if (index >= _this->num_displays) {
                return;
            }

            if (moved) {
                SDL_VideoDisplay tmp;

                SDL_memcpy(&tmp, &_this->displays[index], sizeof(tmp));
                SDL_memcpy(&_this->displays[index], &_this->displays[i], sizeof(tmp));
                SDL_memcpy(&_this->displays[i], &tmp, sizeof(tmp));
                i = index;
            }

            driverdata->MonitorHandle = hMonitor;
            driverdata->IsValid = SDL_TRUE;

            if (!_this->setting_display_mode) {
                SDL_Rect bounds;

                SDL_ResetDisplayModes(i);
                SDL_SetCurrentDisplayMode(&_this->displays[i], &mode);
                SDL_SetDesktopDisplayMode(&_this->displays[i], &mode);
                if (WIN_GetDisplayBounds(_this, &_this->displays[i], &bounds) == 0) {
                    if (SDL_memcmp(&driverdata->bounds, &bounds, sizeof(bounds)) != 0 || moved) {
                        SDL_SendDisplayEvent(&_this->displays[i], SDL_DISPLAYEVENT_MOVED, 0);
                    }
                }
                SDL_SendDisplayEvent(&_this->displays[i], SDL_DISPLAYEVENT_ORIENTATION, orientation);
            }
            goto done;
        }
    }

    displaydata = (SDL_DisplayData *)SDL_calloc(1, sizeof(*displaydata));
    if (!displaydata) {
        return;
    }
    SDL_memcpy(displaydata->DeviceName, info->szDevice, sizeof(displaydata->DeviceName));
    displaydata->MonitorHandle = hMonitor;
    displaydata->IsValid = SDL_TRUE;

    SDL_zero(display);
    display.name = WIN_GetDisplayNameVista(info->szDevice);
    if (!display.name) {
        DISPLAY_DEVICEW device;
        SDL_zero(device);
        device.cb = sizeof(device);
        if (EnumDisplayDevicesW(info->szDevice, 0, &device, 0)) {
            display.name = WIN_StringToUTF8W(device.DeviceString);
        }
    }

    display.desktop_mode = mode;
    display.current_mode = mode;
    display.orientation = orientation;
    display.device = _this;
    display.driverdata = displaydata;
    WIN_GetDisplayBounds(_this, &display, &displaydata->bounds);
    index = SDL_AddVideoDisplay(&display, send_event);
    SDL_free(display.name);

done:
    *display_index = index + 1;
}

typedef struct _WIN_AddDisplaysData
{
    SDL_VideoDevice *video_device;
    int display_index;
    SDL_bool send_event;
    SDL_bool want_primary;
} WIN_AddDisplaysData;

static BOOL CALLBACK WIN_AddDisplaysCallback(HMONITOR hMonitor,
                                             HDC hdcMonitor,
                                             LPRECT lprcMonitor,
                                             LPARAM dwData)
{
    WIN_AddDisplaysData *data = (WIN_AddDisplaysData *)dwData;
    MONITORINFOEXW info;

    SDL_zero(info);
    info.cbSize = sizeof(info);

    if (GetMonitorInfoW(hMonitor, (LPMONITORINFO)&info) != 0) {
        const SDL_bool is_primary = ((info.dwFlags & MONITORINFOF_PRIMARY) == MONITORINFOF_PRIMARY);

        if (is_primary == data->want_primary) {
            WIN_AddDisplay(data->video_device, hMonitor, &info, &data->display_index, data->send_event);
        }
    }

    // continue enumeration
    return TRUE;
}

static void WIN_AddDisplays(_THIS, SDL_bool send_event)
{
    WIN_AddDisplaysData callback_data;
    callback_data.video_device = _this;
    callback_data.display_index = 0;
    callback_data.send_event = send_event;

    callback_data.want_primary = SDL_TRUE;
    EnumDisplayMonitors(NULL, NULL, WIN_AddDisplaysCallback, (LPARAM)&callback_data);

    callback_data.want_primary = SDL_FALSE;
    EnumDisplayMonitors(NULL, NULL, WIN_AddDisplaysCallback, (LPARAM)&callback_data);
}

int WIN_InitModes(_THIS)
{
    WIN_AddDisplays(_this, SDL_FALSE);

    if (_this->num_displays == 0) {
        return SDL_SetError("No displays available");
    }
    return 0;
}

/**
 * Convert the monitor rect and work rect from pixels to the SDL coordinate system (monitor origins are in pixels,
 * monitor size in DPI-scaled points).
 *
 * No-op if DPI scaling is not enabled.
 */
static void WIN_MonitorInfoToSDL(const SDL_VideoData *videodata, HMONITOR monitor, MONITORINFO *info)
{
    UINT xdpi, ydpi;

    if (!videodata->dpi_scaling_enabled) {
        return;
    }

    /* Check for Windows < 8.1*/
    if (!videodata->GetDpiForMonitor) {
        return;
    }
    if (videodata->GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &xdpi, &ydpi) != S_OK) {
        /* Shouldn't happen? */
        return;
    }

    /* Convert monitor size to points, leaving the monitor position in pixels */
    info->rcMonitor.right = info->rcMonitor.left + MulDiv(info->rcMonitor.right - info->rcMonitor.left, 96, xdpi);
    info->rcMonitor.bottom = info->rcMonitor.top + MulDiv(info->rcMonitor.bottom - info->rcMonitor.top, 96, ydpi);

    /* Convert monitor work rect to points */
    info->rcWork.left = info->rcMonitor.left + MulDiv(info->rcWork.left - info->rcMonitor.left, 96, xdpi);
    info->rcWork.right = info->rcMonitor.left + MulDiv(info->rcWork.right - info->rcMonitor.left, 96, xdpi);
    info->rcWork.top = info->rcMonitor.top + MulDiv(info->rcWork.top - info->rcMonitor.top, 96, ydpi);
    info->rcWork.bottom = info->rcMonitor.top + MulDiv(info->rcWork.bottom - info->rcMonitor.top, 96, ydpi);
}

int WIN_GetDisplayBounds(_THIS, SDL_VideoDisplay *display, SDL_Rect *rect)
{
    const SDL_DisplayData *data = (const SDL_DisplayData *)display->driverdata;
    const SDL_VideoData *videodata = (SDL_VideoData *)display->device->driverdata;
    MONITORINFO minfo;
    BOOL rc;

    SDL_zero(minfo);
    minfo.cbSize = sizeof(MONITORINFO);
    rc = GetMonitorInfo(data->MonitorHandle, &minfo);

    if (!rc) {
        return SDL_SetError("Couldn't find monitor data");
    }

    WIN_MonitorInfoToSDL(videodata, data->MonitorHandle, &minfo);
    rect->x = minfo.rcMonitor.left;
    rect->y = minfo.rcMonitor.top;
    rect->w = minfo.rcMonitor.right - minfo.rcMonitor.left;
    rect->h = minfo.rcMonitor.bottom - minfo.rcMonitor.top;

    return 0;
}

int WIN_GetDisplayDPI(_THIS, SDL_VideoDisplay *display, float *ddpi_out, float *hdpi_out, float *vdpi_out)
{
    const SDL_DisplayData *displaydata = (SDL_DisplayData *)display->driverdata;
    const SDL_VideoData *videodata = (SDL_VideoData *)display->device->driverdata;
    float hdpi = 0, vdpi = 0, ddpi = 0;

    if (videodata->GetDpiForMonitor) {
        UINT hdpi_uint, vdpi_uint;
        // Windows 8.1+ codepath
        if (videodata->GetDpiForMonitor(displaydata->MonitorHandle, MDT_EFFECTIVE_DPI, &hdpi_uint, &vdpi_uint) == S_OK) {
            // GetDpiForMonitor docs promise to return the same hdpi/vdpi
            hdpi = (float)hdpi_uint;
            vdpi = (float)hdpi_uint;
            ddpi = (float)hdpi_uint;
        } else {
            return SDL_SetError("GetDpiForMonitor failed");
        }
    } else {
        // Window 8.0 and below: same DPI for all monitors.
        HDC hdc;
        int hdpi_int, vdpi_int, hpoints, vpoints, hpix, vpix;
        float hinches, vinches;

        hdc = GetDC(NULL);
        if (!hdc) {
            return SDL_SetError("GetDC failed");
        }
        hdpi_int = GetDeviceCaps(hdc, LOGPIXELSX);
        vdpi_int = GetDeviceCaps(hdc, LOGPIXELSY);
        ReleaseDC(NULL, hdc);

        hpoints = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        vpoints = GetSystemMetrics(SM_CYVIRTUALSCREEN);

        hpix = MulDiv(hpoints, hdpi_int, 96);
        vpix = MulDiv(vpoints, vdpi_int, 96);

        hinches = (float)hpoints / 96.0f;
        vinches = (float)vpoints / 96.0f;

        hdpi = (float)hdpi_int;
        vdpi = (float)vdpi_int;
        ddpi = SDL_ComputeDiagonalDPI(hpix, vpix, hinches, vinches);
    }

    if (ddpi_out) {
        *ddpi_out = ddpi;
    }
    if (hdpi_out) {
        *hdpi_out = hdpi;
    }
    if (vdpi_out) {
        *vdpi_out = vdpi;
    }

    return ddpi != 0.0f ? 0 : SDL_SetError("Couldn't get DPI");
}

int WIN_GetDisplayUsableBounds(_THIS, SDL_VideoDisplay *display, SDL_Rect *rect)
{
    const SDL_DisplayData *data = (const SDL_DisplayData *)display->driverdata;
    const SDL_VideoData *videodata = (SDL_VideoData *)display->device->driverdata;
    MONITORINFO minfo;
    BOOL rc;

    SDL_zero(minfo);
    minfo.cbSize = sizeof(MONITORINFO);
    rc = GetMonitorInfo(data->MonitorHandle, &minfo);

    if (!rc) {
        return SDL_SetError("Couldn't find monitor data");
    }

    WIN_MonitorInfoToSDL(videodata, data->MonitorHandle, &minfo);
    rect->x = minfo.rcWork.left;
    rect->y = minfo.rcWork.top;
    rect->w = minfo.rcWork.right - minfo.rcWork.left;
    rect->h = minfo.rcWork.bottom - minfo.rcWork.top;

    return 0;
}

/**
 * Convert a point from the SDL coordinate system (monitor origins are in pixels,
 * offset within a monitor in DPI-scaled points) to Windows virtual screen coordinates (pixels).
 *
 * No-op if DPI scaling is not enabled (returns 96 dpi).
 *
 * Returns the DPI of the monitor that was closest to x, y and used for the conversion.
 */
void WIN_ScreenPointFromSDL(int *x, int *y, int *dpiOut)
{
    const SDL_VideoDevice *videodevice = SDL_GetVideoDevice();
    const SDL_VideoData *videodata;
    int displayIndex;
    SDL_Rect bounds;
    float ddpi, hdpi, vdpi;
    int x_sdl, y_sdl;
    SDL_Point point;
    point.x = *x;
    point.y = *y;

    if (dpiOut) {
        *dpiOut = 96;
    }

    if (!videodevice || !videodevice->driverdata) {
        return;
    }

    videodata = (SDL_VideoData *)videodevice->driverdata;
    if (!videodata->dpi_scaling_enabled) {
        return;
    }

    /* Can't use MonitorFromPoint for this because we currently have SDL coordinates, not pixels */
    displayIndex = SDL_GetPointDisplayIndex(&point);

    if (displayIndex < 0) {
        return;
    }

    if (SDL_GetDisplayBounds(displayIndex, &bounds) < 0 || SDL_GetDisplayDPI(displayIndex, &ddpi, &hdpi, &vdpi) < 0) {
        return;
    }

    if (dpiOut) {
        *dpiOut = (int)ddpi;
    }

    /* Undo the DPI-scaling within the monitor bounds to convert back to pixels */
    x_sdl = *x;
    y_sdl = *y;
    *x = bounds.x + MulDiv(x_sdl - bounds.x, (int)ddpi, 96);
    *y = bounds.y + MulDiv(y_sdl - bounds.y, (int)ddpi, 96);

#ifdef HIGHDPI_DEBUG_VERBOSE
    SDL_Log("WIN_ScreenPointFromSDL: (%d, %d) points -> (%d x %d) pixels, using %d DPI monitor",
            x_sdl, y_sdl, *x, *y, (int)ddpi);
#endif
}

/**
 * Convert a point from Windows virtual screen coordinates (pixels) to the SDL
 * coordinate system (monitor origins are in pixels, offset within a monitor in DPI-scaled points).
 *
 * No-op if DPI scaling is not enabled.
 */
void WIN_ScreenPointToSDL(int *x, int *y)
{
    const SDL_VideoDevice *videodevice = SDL_GetVideoDevice();
    const SDL_VideoData *videodata;
    POINT point;
    HMONITOR monitor;
    int i, displayIndex;
    SDL_Rect bounds;
    float ddpi, hdpi, vdpi;
    int x_pixels, y_pixels;

    if (!videodevice || !videodevice->driverdata) {
        return;
    }

    videodata = (SDL_VideoData *)videodevice->driverdata;
    if (!videodata->dpi_scaling_enabled) {
        return;
    }

    point.x = *x;
    point.y = *y;
    monitor = MonitorFromPoint(point, MONITOR_DEFAULTTONEAREST);

    /* Search for the corresponding SDL monitor */
    displayIndex = -1;
    for (i = 0; i < videodevice->num_displays; ++i) {
        SDL_DisplayData *driverdata = (SDL_DisplayData *)videodevice->displays[i].driverdata;
        if (driverdata->MonitorHandle == monitor) {
            displayIndex = i;
        }
    }
    if (displayIndex == -1) {
        return;
    }

    /* Get SDL display properties */
    if (SDL_GetDisplayBounds(displayIndex, &bounds) < 0 || SDL_GetDisplayDPI(displayIndex, &ddpi, &hdpi, &vdpi) < 0) {
        return;
    }

    /* Convert the point's offset within the monitor from pixels to DPI-scaled points */
    x_pixels = *x;
    y_pixels = *y;
    *x = bounds.x + MulDiv(x_pixels - bounds.x, 96, (int)ddpi);
    *y = bounds.y + MulDiv(y_pixels - bounds.y, 96, (int)ddpi);

#ifdef HIGHDPI_DEBUG_VERBOSE
    SDL_Log("WIN_ScreenPointToSDL: (%d, %d) pixels -> (%d x %d) points, using %d DPI monitor",
            x_pixels, y_pixels, *x, *y, (int)ddpi);
#endif
}

void WIN_GetDisplayModes(_THIS, SDL_VideoDisplay *display)
{
    SDL_DisplayData *data = (SDL_DisplayData *)display->driverdata;
    DWORD i;
    SDL_DisplayMode mode;

    for (i = 0;; ++i) {
        if (!WIN_GetDisplayMode(_this, data->DeviceName, i, &mode, NULL)) {
            break;
        }
        if (SDL_ISPIXELFORMAT_INDEXED(mode.format)) {
            /* We don't support palettized modes now */
            SDL_free(mode.driverdata);
            continue;
        }
        if (mode.format != SDL_PIXELFORMAT_UNKNOWN) {
            if (!SDL_AddDisplayMode(display, &mode)) {
                SDL_free(mode.driverdata);
            }
        } else {
            SDL_free(mode.driverdata);
        }
    }
}

#ifdef DEBUG_MODES
static void WIN_LogMonitor(_THIS, HMONITOR mon)
{
    const SDL_VideoData *vid_data = (const SDL_VideoData *)_this->driverdata;
    MONITORINFOEX minfo;
    UINT xdpi = 0, ydpi = 0;
    char *name_utf8;

    if (vid_data->GetDpiForMonitor) {
        vid_data->GetDpiForMonitor(mon, MDT_EFFECTIVE_DPI, &xdpi, &ydpi);
    }

    SDL_zero(minfo);
    minfo.cbSize = sizeof(minfo);
    GetMonitorInfo(mon, (LPMONITORINFO)&minfo);

    name_utf8 = WIN_StringToUTF8(minfo.szDevice);

    SDL_Log("WIN_LogMonitor: monitor \"%s\": dpi: %d windows screen coordinates: %d, %d, %dx%d",
            name_utf8,
            xdpi,
            minfo.rcMonitor.left,
            minfo.rcMonitor.top,
            minfo.rcMonitor.right - minfo.rcMonitor.left,
            minfo.rcMonitor.bottom - minfo.rcMonitor.top);

    SDL_free(name_utf8);
}
#endif

int WIN_SetDisplayMode(_THIS, SDL_VideoDisplay *display, SDL_DisplayMode *mode)
{
    SDL_DisplayData *displaydata = (SDL_DisplayData *)display->driverdata;
    SDL_DisplayModeData *data = (SDL_DisplayModeData *)mode->driverdata;
    LONG status;

#ifdef DEBUG_MODES
    SDL_Log("WIN_SetDisplayMode: monitor state before mode change:");
    WIN_LogMonitor(_this, displaydata->MonitorHandle);
#endif

    /* High-DPI notes:

       - ChangeDisplaySettingsEx always takes pixels.
       - e.g. if the display is set to 2880x1800 with 200% scaling in Display Settings
         - calling ChangeDisplaySettingsEx with a dmPelsWidth/Height other than 2880x1800 will
           change the monitor DPI to 96. (100% scaling)
         - calling ChangeDisplaySettingsEx with a dmPelsWidth/Height of 2880x1800 (or a NULL DEVMODE*) will
           reset the monitor DPI to 192. (200% scaling)

       NOTE: these are temporary changes in DPI, not modifications to the Control Panel setting. */
    if (mode->driverdata == display->desktop_mode.driverdata) {
#ifdef DEBUG_MODES
        SDL_Log("WIN_SetDisplayMode: resetting to original resolution");
#endif
        status = ChangeDisplaySettingsExW(displaydata->DeviceName, NULL, NULL, CDS_FULLSCREEN, NULL);
    } else {
#ifdef DEBUG_MODES
        SDL_Log("WIN_SetDisplayMode: changing to %dx%d pixels", data->DeviceMode.dmPelsWidth, data->DeviceMode.dmPelsHeight);
#endif
        status = ChangeDisplaySettingsExW(displaydata->DeviceName, &data->DeviceMode, NULL, CDS_FULLSCREEN, NULL);
    }
    if (status != DISP_CHANGE_SUCCESSFUL) {
        const char *reason = "Unknown reason";
        switch (status) {
        case DISP_CHANGE_BADFLAGS:
            reason = "DISP_CHANGE_BADFLAGS";
            break;
        case DISP_CHANGE_BADMODE:
            reason = "DISP_CHANGE_BADMODE";
            break;
        case DISP_CHANGE_BADPARAM:
            reason = "DISP_CHANGE_BADPARAM";
            break;
        case DISP_CHANGE_FAILED:
            reason = "DISP_CHANGE_FAILED";
            break;
        }
        return SDL_SetError("ChangeDisplaySettingsEx() failed: %s", reason);
    }

#ifdef DEBUG_MODES
    SDL_Log("WIN_SetDisplayMode: monitor state after mode change:");
    WIN_LogMonitor(_this, displaydata->MonitorHandle);
#endif

    EnumDisplaySettingsW(displaydata->DeviceName, ENUM_CURRENT_SETTINGS, &data->DeviceMode);
    WIN_UpdateDisplayMode(_this, displaydata->DeviceName, ENUM_CURRENT_SETTINGS, mode);
    return 0;
}

void WIN_RefreshDisplays(_THIS)
{
    int i;

    // Mark all displays as potentially invalid to detect
    // entries that have actually been removed
    for (i = 0; i < _this->num_displays; ++i) {
        SDL_DisplayData *driverdata = (SDL_DisplayData *)_this->displays[i].driverdata;
        driverdata->IsValid = SDL_FALSE;
    }

    // Enumerate displays to add any new ones and mark still
    // connected entries as valid
    WIN_AddDisplays(_this, SDL_TRUE);

    // Delete any entries still marked as invalid, iterate
    // in reverse as each delete takes effect immediately
    for (i = _this->num_displays - 1; i >= 0; --i) {
        SDL_DisplayData *driverdata = (SDL_DisplayData *)_this->displays[i].driverdata;
        if (driverdata->IsValid == SDL_FALSE) {
            SDL_DelVideoDisplay(i);
        }
    }
}

void WIN_QuitModes(_THIS)
{
    /* All fullscreen windows should have restored modes by now */
}

#endif /* SDL_VIDEO_DRIVER_WINDOWS */

/* vi: set ts=4 sw=4 expandtab: */
