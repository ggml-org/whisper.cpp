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

#ifndef SDL_windowsvideo_h_
#define SDL_windowsvideo_h_

#include "../../core/windows/SDL_windows.h"

#include "../SDL_sysvideo.h"

#if defined(_MSC_VER) && (_MSC_VER >= 1500) && !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
#include <msctf.h>
#else
#include "SDL_msctf.h"
#endif

#include <imm.h>

#define MAX_CANDLIST   10
#define MAX_CANDLENGTH 256
#define MAX_CANDSIZE   (sizeof(WCHAR) * MAX_CANDLIST * MAX_CANDLENGTH)

#include "SDL_windowsclipboard.h"
#include "SDL_windowsevents.h"
#include "SDL_windowsopengl.h"

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
#include "SDL_windowskeyboard.h"
#include "SDL_windowsmodes.h"
#include "SDL_windowsmouse.h"
#include "SDL_windowsopengles.h"
#endif

#include "SDL_windowswindow.h"
#include "SDL_events.h"
#include "SDL_loadso.h"

#if WINVER < 0x0601
/* Touch input definitions */
#define TWF_FINETOUCH 1
#define TWF_WANTPALM  2

#define TOUCHEVENTF_MOVE 0x0001
#define TOUCHEVENTF_DOWN 0x0002
#define TOUCHEVENTF_UP   0x0004

DECLARE_HANDLE(HTOUCHINPUT);

typedef struct _TOUCHINPUT
{
    LONG x;
    LONG y;
    HANDLE hSource;
    DWORD dwID;
    DWORD dwFlags;
    DWORD dwMask;
    DWORD dwTime;
    ULONG_PTR dwExtraInfo;
    DWORD cxContact;
    DWORD cyContact;
} TOUCHINPUT, *PTOUCHINPUT;

/* More-robust display information in Vista... */
/* This is a huge amount of data to be stuffing into three API calls. :( */
typedef struct DISPLAYCONFIG_PATH_SOURCE_INFO
{
    LUID adapterId;
    UINT32 id;
    union
    {
        UINT32 modeInfoIdx;
        struct
        {
            UINT32 cloneGroupId : 16;
            UINT32 sourceModeInfoIdx : 16;
        } DUMMYSTRUCTNAME;
    } DUMMYUNIONNAME;

    UINT32 statusFlags;
} DISPLAYCONFIG_PATH_SOURCE_INFO;

typedef struct DISPLAYCONFIG_RATIONAL
{
    UINT32 Numerator;
    UINT32 Denominator;
} DISPLAYCONFIG_RATIONAL;

typedef struct DISPLAYCONFIG_PATH_TARGET_INFO
{
    LUID adapterId;
    UINT32 id;
    union
    {
        UINT32 modeInfoIdx;
        struct
        {
            UINT32 desktopModeInfoIdx : 16;
            UINT32 targetModeInfoIdx : 16;
        } DUMMYSTRUCTNAME;
    } DUMMYUNIONNAME;
    UINT32 /*DISPLAYCONFIG_VIDEO_OUTPUT_TECHNOLOGY*/ outputTechnology;
    UINT32 /*DISPLAYCONFIG_ROTATION*/ rotation;
    UINT32 /*DISPLAYCONFIG_SCALING*/ scaling;
    DISPLAYCONFIG_RATIONAL refreshRate;
    UINT32 /*DISPLAYCONFIG_SCANLINE_ORDERING*/ scanLineOrdering;
    BOOL targetAvailable;
    UINT32 statusFlags;
} DISPLAYCONFIG_PATH_TARGET_INFO;

typedef struct DISPLAYCONFIG_PATH_INFO
{
    DISPLAYCONFIG_PATH_SOURCE_INFO sourceInfo;
    DISPLAYCONFIG_PATH_TARGET_INFO targetInfo;
    UINT32 flags;
} DISPLAYCONFIG_PATH_INFO;

typedef enum
{
    DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE = 1,
    DISPLAYCONFIG_MODE_INFO_TYPE_TARGET = 2,
    DISPLAYCONFIG_MODE_INFO_TYPE_DESKTOP_IMAGE = 3,
    DISPLAYCONFIG_MODE_INFO_TYPE_FORCE_UINT32 = 0xFFFFFFFF
} DISPLAYCONFIG_MODE_INFO_TYPE;

typedef struct DISPLAYCONFIG_2DREGION
{
    UINT32 cx;
    UINT32 cy;
} DISPLAYCONFIG_2DREGION;

typedef struct DISPLAYCONFIG_VIDEO_SIGNAL_INFO
{
    UINT64 pixelRate;
    DISPLAYCONFIG_RATIONAL hSyncFreq;
    DISPLAYCONFIG_RATIONAL vSyncFreq;
    DISPLAYCONFIG_2DREGION activeSize;
    DISPLAYCONFIG_2DREGION totalSize;

    union
    {
        struct
        {
            UINT32 videoStandard : 16;

            // Vertical refresh frequency divider
            UINT32 vSyncFreqDivider : 6;

            UINT32 reserved : 10;
        } AdditionalSignalInfo;

        UINT32 videoStandard;
    } DUMMYUNIONNAME;

    // Scan line ordering (e.g. progressive, interlaced).
    UINT32 /*DISPLAYCONFIG_SCANLINE_ORDERING*/ scanLineOrdering;
} DISPLAYCONFIG_VIDEO_SIGNAL_INFO;

typedef struct DISPLAYCONFIG_SOURCE_MODE
{
    UINT32 width;
    UINT32 height;
    UINT32 /*DISPLAYCONFIG_PIXELFORMAT*/ pixelFormat;
    POINTL position;
} DISPLAYCONFIG_SOURCE_MODE;

typedef struct DISPLAYCONFIG_TARGET_MODE
{
    DISPLAYCONFIG_VIDEO_SIGNAL_INFO targetVideoSignalInfo;
} DISPLAYCONFIG_TARGET_MODE;

typedef struct DISPLAYCONFIG_DESKTOP_IMAGE_INFO
{
    POINTL PathSourceSize;
    RECTL DesktopImageRegion;
    RECTL DesktopImageClip;
} DISPLAYCONFIG_DESKTOP_IMAGE_INFO;

typedef struct DISPLAYCONFIG_MODE_INFO
{
    DISPLAYCONFIG_MODE_INFO_TYPE infoType;
    UINT32 id;
    LUID adapterId;
    union
    {
        DISPLAYCONFIG_TARGET_MODE targetMode;
        DISPLAYCONFIG_SOURCE_MODE sourceMode;
        DISPLAYCONFIG_DESKTOP_IMAGE_INFO desktopImageInfo;
    } DUMMYUNIONNAME;
} DISPLAYCONFIG_MODE_INFO;

typedef enum DISPLAYCONFIG_TOPOLOGY_ID
{
    DISPLAYCONFIG_TOPOLOGY_INTERNAL = 0x00000001,
    DISPLAYCONFIG_TOPOLOGY_CLONE = 0x00000002,
    DISPLAYCONFIG_TOPOLOGY_EXTEND = 0x00000004,
    DISPLAYCONFIG_TOPOLOGY_EXTERNAL = 0x00000008,
    DISPLAYCONFIG_TOPOLOGY_FORCE_UINT32 = 0xFFFFFFFF
} DISPLAYCONFIG_TOPOLOGY_ID;

typedef enum
{
    DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME = 1,
    DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME = 2,
    DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_PREFERRED_MODE = 3,
    DISPLAYCONFIG_DEVICE_INFO_GET_ADAPTER_NAME = 4,
    DISPLAYCONFIG_DEVICE_INFO_SET_TARGET_PERSISTENCE = 5,
    DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_BASE_TYPE = 6,
    DISPLAYCONFIG_DEVICE_INFO_GET_SUPPORT_VIRTUAL_RESOLUTION = 7,
    DISPLAYCONFIG_DEVICE_INFO_SET_SUPPORT_VIRTUAL_RESOLUTION = 8,
    DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO = 9,
    DISPLAYCONFIG_DEVICE_INFO_SET_ADVANCED_COLOR_STATE = 10,
    DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL = 11,
    DISPLAYCONFIG_DEVICE_INFO_FORCE_UINT32 = 0xFFFFFFFF
} DISPLAYCONFIG_DEVICE_INFO_TYPE;

typedef struct DISPLAYCONFIG_DEVICE_INFO_HEADER
{
    DISPLAYCONFIG_DEVICE_INFO_TYPE type;
    UINT32 size;
    LUID adapterId;
    UINT32 id;
} DISPLAYCONFIG_DEVICE_INFO_HEADER;

typedef struct DISPLAYCONFIG_SOURCE_DEVICE_NAME
{
    DISPLAYCONFIG_DEVICE_INFO_HEADER header;
    WCHAR viewGdiDeviceName[CCHDEVICENAME];
} DISPLAYCONFIG_SOURCE_DEVICE_NAME;

typedef struct DISPLAYCONFIG_TARGET_DEVICE_NAME_FLAGS
{
    union
    {
        struct
        {
            UINT32 friendlyNameFromEdid : 1;
            UINT32 friendlyNameForced : 1;
            UINT32 edidIdsValid : 1;
            UINT32 reserved : 29;
        } DUMMYSTRUCTNAME;
        UINT32 value;
    } DUMMYUNIONNAME;
} DISPLAYCONFIG_TARGET_DEVICE_NAME_FLAGS;

typedef struct DISPLAYCONFIG_TARGET_DEVICE_NAME
{
    DISPLAYCONFIG_DEVICE_INFO_HEADER header;
    DISPLAYCONFIG_TARGET_DEVICE_NAME_FLAGS flags;
    UINT32 /*DISPLAYCONFIG_VIDEO_OUTPUT_TECHNOLOGY*/ outputTechnology;
    UINT16 edidManufactureId;
    UINT16 edidProductCodeId;
    UINT32 connectorInstance;
    WCHAR monitorFriendlyDeviceName[64];
    WCHAR monitorDevicePath[128];
} DISPLAYCONFIG_TARGET_DEVICE_NAME;

#define QDC_ONLY_ACTIVE_PATHS 0x00000002

#endif /* WINVER < 0x0601 */

#ifndef HAVE_SHELLSCALINGAPI_H

typedef enum MONITOR_DPI_TYPE
{
    MDT_EFFECTIVE_DPI = 0,
    MDT_ANGULAR_DPI = 1,
    MDT_RAW_DPI = 2,
    MDT_DEFAULT = MDT_EFFECTIVE_DPI
} MONITOR_DPI_TYPE;

typedef enum PROCESS_DPI_AWARENESS
{
    PROCESS_DPI_UNAWARE = 0,
    PROCESS_SYSTEM_DPI_AWARE = 1,
    PROCESS_PER_MONITOR_DPI_AWARE = 2
} PROCESS_DPI_AWARENESS;

#else
#include <shellscalingapi.h>
#endif

#ifndef _DPI_AWARENESS_CONTEXTS_

typedef enum DPI_AWARENESS
{
    DPI_AWARENESS_INVALID = -1,
    DPI_AWARENESS_UNAWARE = 0,
    DPI_AWARENESS_SYSTEM_AWARE = 1,
    DPI_AWARENESS_PER_MONITOR_AWARE = 2
} DPI_AWARENESS;

DECLARE_HANDLE(DPI_AWARENESS_CONTEXT);

#define DPI_AWARENESS_CONTEXT_UNAWARE           ((DPI_AWARENESS_CONTEXT)-1)
#define DPI_AWARENESS_CONTEXT_SYSTEM_AWARE      ((DPI_AWARENESS_CONTEXT)-2)
#define DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE ((DPI_AWARENESS_CONTEXT)-3)

#endif /* _DPI_AWARENESS_CONTEXTS_ */

/* Windows 10 Creators Update */
#if NTDDI_VERSION < 0x0A000003
#define DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 ((DPI_AWARENESS_CONTEXT)-4)
#endif /* NTDDI_VERSION < 0x0A000003 */

/* Windows 10 version 1809 */
#if NTDDI_VERSION < 0x0A000006
#define DPI_AWARENESS_CONTEXT_UNAWARE_GDISCALED ((DPI_AWARENESS_CONTEXT)-5)
#endif /* NTDDI_VERSION < 0x0A000006 */

typedef BOOL (*PFNSHFullScreen)(HWND, DWORD);
typedef void (*PFCoordTransform)(SDL_Window *, POINT *);

typedef struct
{
    void **lpVtbl;
    int refcount;
    void *data;
} TSFSink;

#ifndef SDL_DISABLE_WINDOWS_IME
/* Definition from Win98DDK version of IMM.H */
typedef struct tagINPUTCONTEXT2
{
    HWND hWnd;
    BOOL fOpen;
    POINT ptStatusWndPos;
    POINT ptSoftKbdPos;
    DWORD fdwConversion;
    DWORD fdwSentence;
    union
    {
        LOGFONTA A;
        LOGFONTW W;
    } lfFont;
    COMPOSITIONFORM cfCompForm;
    CANDIDATEFORM cfCandForm[4];
    HIMCC hCompStr;
    HIMCC hCandInfo;
    HIMCC hGuideLine;
    HIMCC hPrivate;
    DWORD dwNumMsgBuf;
    HIMCC hMsgBuf;
    DWORD fdwInit;
    DWORD dwReserve[3];
} INPUTCONTEXT2, *PINPUTCONTEXT2, NEAR *NPINPUTCONTEXT2, FAR *LPINPUTCONTEXT2;
#endif /* !SDL_DISABLE_WINDOWS_IME */

/* Private display data */

typedef struct SDL_VideoData
{
    int render;

    DWORD clipboard_count;

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__) /* Xbox doesn't support user32/shcore*/
    /* Touch input functions */
    void *userDLL;
    /* *INDENT-OFF* */ /* clang-format off */
    BOOL (WINAPI *CloseTouchInputHandle)( HTOUCHINPUT );
    BOOL (WINAPI *GetTouchInputInfo)( HTOUCHINPUT, UINT, PTOUCHINPUT, int );
    BOOL (WINAPI *RegisterTouchWindow)( HWND, ULONG );
    BOOL (WINAPI *SetProcessDPIAware)( void );
    BOOL (WINAPI *SetProcessDpiAwarenessContext)( DPI_AWARENESS_CONTEXT );
    DPI_AWARENESS_CONTEXT (WINAPI *SetThreadDpiAwarenessContext)( DPI_AWARENESS_CONTEXT );
    DPI_AWARENESS_CONTEXT (WINAPI *GetThreadDpiAwarenessContext)( void );
    DPI_AWARENESS (WINAPI *GetAwarenessFromDpiAwarenessContext)( DPI_AWARENESS_CONTEXT );
    BOOL (WINAPI *EnableNonClientDpiScaling)( HWND );
    BOOL (WINAPI *AdjustWindowRectExForDpi)( LPRECT, DWORD, BOOL, DWORD, UINT );
    UINT (WINAPI *GetDpiForWindow)( HWND );
    BOOL (WINAPI *AreDpiAwarenessContextsEqual)(DPI_AWARENESS_CONTEXT, DPI_AWARENESS_CONTEXT);
    BOOL (WINAPI *IsValidDpiAwarenessContext)(DPI_AWARENESS_CONTEXT);
    /* *INDENT-ON* */ /* clang-format on */

    void *shcoreDLL;
    /* *INDENT-OFF* */ /* clang-format off */
    HRESULT (WINAPI *GetDpiForMonitor)( HMONITOR         hmonitor,
                                        MONITOR_DPI_TYPE dpiType,
                                        UINT             *dpiX,
                                        UINT             *dpiY );
    HRESULT (WINAPI *SetProcessDpiAwareness)(PROCESS_DPI_AWARENESS dpiAwareness);
    /* *INDENT-ON* */ /* clang-format on */
#endif                /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/

    SDL_bool dpi_scaling_enabled;
    SDL_bool cleared;

#ifndef SDL_DISABLE_WINDOWS_IME
    SDL_bool ime_com_initialized;
    struct ITfThreadMgr *ime_threadmgr;
    SDL_bool ime_initialized;
    SDL_bool ime_enabled;
    SDL_bool ime_available;
    HWND ime_hwnd_main;
    HWND ime_hwnd_current;
    SDL_bool ime_suppress_endcomposition_event;
    HIMC ime_himc;

    WCHAR *ime_composition;
    int ime_composition_length;
    WCHAR ime_readingstring[16];
    int ime_cursor;

    SDL_bool ime_candlist;
    WCHAR *ime_candidates;
    DWORD ime_candcount;
    DWORD ime_candref;
    DWORD ime_candsel;
    UINT ime_candpgsize;
    int ime_candlistindexbase;
    SDL_bool ime_candvertical;

    SDL_bool ime_dirty;
    SDL_Rect ime_rect;
    SDL_Rect ime_candlistrect;
    int ime_winwidth;
    int ime_winheight;

    HKL ime_hkl;
    void *ime_himm32;
    /* *INDENT-OFF* */ /* clang-format off */
    UINT (WINAPI *GetReadingString)(HIMC himc, UINT uReadingBufLen, LPWSTR lpwReadingBuf, PINT pnErrorIndex, BOOL *pfIsVertical, PUINT puMaxReadingLen);
    BOOL (WINAPI *ShowReadingWindow)(HIMC himc, BOOL bShow);
    LPINPUTCONTEXT2 (WINAPI *ImmLockIMC)(HIMC himc);
    BOOL (WINAPI *ImmUnlockIMC)(HIMC himc);
    LPVOID (WINAPI *ImmLockIMCC)(HIMCC himcc);
    BOOL (WINAPI *ImmUnlockIMCC)(HIMCC himcc);
    /* *INDENT-ON* */ /* clang-format on */

    SDL_bool ime_uiless;
    struct ITfThreadMgrEx *ime_threadmgrex;
    DWORD ime_uielemsinkcookie;
    DWORD ime_alpnsinkcookie;
    DWORD ime_openmodesinkcookie;
    DWORD ime_convmodesinkcookie;
    TSFSink *ime_uielemsink;
    TSFSink *ime_ippasink;
    LONG ime_uicontext;
#endif /* !SDL_DISABLE_WINDOWS_IME */

    BYTE pre_hook_key_state[256];
    UINT _SDL_WAKEUP;
} SDL_VideoData;

extern SDL_bool g_WindowsEnableMessageLoop;
extern SDL_bool g_WindowsEnableMenuMnemonics;
extern SDL_bool g_WindowFrameUsableWhileCursorHidden;

typedef struct IDirect3D9 IDirect3D9;
extern SDL_bool D3D_LoadDLL(void **pD3DDLL, IDirect3D9 **pDirect3D9Interface);

extern SDL_bool WIN_IsPerMonitorV2DPIAware(_THIS);

#endif /* SDL_windowsvideo_h_ */

/* vi: set ts=4 sw=4 expandtab: */
