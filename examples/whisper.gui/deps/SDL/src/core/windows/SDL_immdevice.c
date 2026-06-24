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

#if (defined(__WIN32__) || defined(__GDK__)) && defined(HAVE_MMDEVICEAPI_H)

#include "SDL_windows.h"
#include "SDL_immdevice.h"
#include "SDL_timer.h"
#include "../../audio/SDL_sysaudio.h"
#include <objbase.h> /* For CLSIDFromString */

static const ERole SDL_IMMDevice_role = eConsole; /* !!! FIXME: should this be eMultimedia? Should be a hint? */

/* This is global to the WASAPI target, to handle hotplug and default device lookup. */
static IMMDeviceEnumerator *enumerator = NULL;

/* PropVariantInit() is an inline function/macro in PropIdl.h that calls the C runtime's memset() directly. Use ours instead, to avoid dependency. */
#ifdef PropVariantInit
#undef PropVariantInit
#endif
#define PropVariantInit(p) SDL_zerop(p)

/* Some GUIDs we need to know without linking to libraries that aren't available before Vista. */
/* *INDENT-OFF* */ /* clang-format off */
static const CLSID SDL_CLSID_MMDeviceEnumerator = { 0xbcde0395, 0xe52f, 0x467c,{ 0x8e, 0x3d, 0xc4, 0x57, 0x92, 0x91, 0x69, 0x2e } };
static const IID SDL_IID_IMMDeviceEnumerator = { 0xa95664d2, 0x9614, 0x4f35,{ 0xa7, 0x46, 0xde, 0x8d, 0xb6, 0x36, 0x17, 0xe6 } };
static const IID SDL_IID_IMMNotificationClient = { 0x7991eec9, 0x7e89, 0x4d85,{ 0x83, 0x90, 0x6c, 0x70, 0x3c, 0xec, 0x60, 0xc0 } };
static const IID SDL_IID_IMMEndpoint = { 0x1be09788, 0x6894, 0x4089,{ 0x85, 0x86, 0x9a, 0x2a, 0x6c, 0x26, 0x5a, 0xc5 } };
static const PROPERTYKEY SDL_PKEY_Device_FriendlyName = { { 0xa45c254e, 0xdf1c, 0x4efd,{ 0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0, } }, 14 };
static const PROPERTYKEY SDL_PKEY_AudioEngine_DeviceFormat = { { 0xf19f064d, 0x82c, 0x4e27,{ 0xbc, 0x73, 0x68, 0x82, 0xa1, 0xbb, 0x8e, 0x4c, } }, 0 };
static const PROPERTYKEY SDL_PKEY_AudioEndpoint_GUID = { { 0x1da5d803, 0xd492, 0x4edd,{ 0x8c, 0x23, 0xe0, 0xc0, 0xff, 0xee, 0x7f, 0x0e, } }, 4 };
static const GUID SDL_KSDATAFORMAT_SUBTYPE_PCM = { 0x00000001, 0x0000, 0x0010,{ 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 } };
static const GUID SDL_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = { 0x00000003, 0x0000, 0x0010,{ 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 } };
/* *INDENT-ON* */ /* clang-format on */

/* these increment as default devices change. Opened default devices pick up changes in their threads. */
SDL_atomic_t SDL_IMMDevice_DefaultPlaybackGeneration;
SDL_atomic_t SDL_IMMDevice_DefaultCaptureGeneration;

static void GetMMDeviceInfo(IMMDevice *device, char **utf8dev, WAVEFORMATEXTENSIBLE *fmt, GUID *guid)
{
    /* PKEY_Device_FriendlyName gives you "Speakers (SoundBlaster Pro)" which drives me nuts. I'd rather it be
       "SoundBlaster Pro (Speakers)" but I guess that's developers vs users. Windows uses the FriendlyName in
       its own UIs, like Volume Control, etc. */
    IPropertyStore *props = NULL;
    *utf8dev = NULL;
    SDL_zerop(fmt);
    if (SUCCEEDED(IMMDevice_OpenPropertyStore(device, STGM_READ, &props))) {
        PROPVARIANT var;
        PropVariantInit(&var);
        if (SUCCEEDED(IPropertyStore_GetValue(props, &SDL_PKEY_Device_FriendlyName, &var))) {
            *utf8dev = WIN_StringToUTF8W(var.pwszVal);
        }
        PropVariantClear(&var);
        if (SUCCEEDED(IPropertyStore_GetValue(props, &SDL_PKEY_AudioEngine_DeviceFormat, &var))) {
            SDL_memcpy(fmt, var.blob.pBlobData, SDL_min(var.blob.cbSize, sizeof(WAVEFORMATEXTENSIBLE)));
        }
        PropVariantClear(&var);
        if (SUCCEEDED(IPropertyStore_GetValue(props, &SDL_PKEY_AudioEndpoint_GUID, &var))) {
            CLSIDFromString(var.pwszVal, guid);
        }
        PropVariantClear(&var);
        IPropertyStore_Release(props);
    }
}

/* This is a list of device id strings we have inflight, so we have consistent pointers to the same device. */
typedef struct DevIdList
{
    LPWSTR str;
    LPGUID guid;
    struct DevIdList *next;
} DevIdList;

static DevIdList *deviceid_list = NULL;

static void SDL_IMMDevice_Remove(const SDL_bool iscapture, LPCWSTR devid, SDL_bool useguid)
{
    DevIdList *i;
    DevIdList *next;
    DevIdList *prev = NULL;
    for (i = deviceid_list; i; i = next) {
        next = i->next;
        if (SDL_wcscmp(i->str, devid) == 0) {
            if (prev) {
                prev->next = next;
            } else {
                deviceid_list = next;
            }
            SDL_RemoveAudioDevice(iscapture, useguid ? ((void *)i->guid) : ((void *)i->str));
            SDL_free(i->str);
            SDL_free(i);
        } else {
            prev = i;
        }
    }
}

static void SDL_IMMDevice_Add(const SDL_bool iscapture, const char *devname, WAVEFORMATEXTENSIBLE *fmt, LPCWSTR devid, GUID *dsoundguid, SDL_bool useguid)
{
    DevIdList *devidlist;
    SDL_AudioSpec spec;
    LPWSTR devidcopy;
    LPGUID cpyguid;
    LPVOID driverdata;

    /* You can have multiple endpoints on a device that are mutually exclusive ("Speakers" vs "Line Out" or whatever).
       In a perfect world, things that are unplugged won't be in this collection. The only gotcha is probably for
       phones and tablets, where you might have an internal speaker and a headphone jack and expect both to be
       available and switch automatically. (!!! FIXME...?) */

    /* see if we already have this one. */
    for (devidlist = deviceid_list; devidlist; devidlist = devidlist->next) {
        if (SDL_wcscmp(devidlist->str, devid) == 0) {
            return; /* we already have this. */
        }
    }

    devidlist = (DevIdList *)SDL_malloc(sizeof(*devidlist));
    if (!devidlist) {
        return; /* oh well. */
    }

    devidcopy = SDL_wcsdup(devid);
    if (!devidcopy) {
        SDL_free(devidlist);
        return; /* oh well. */
    }

    if (useguid) {
        /* This is freed by DSOUND_FreeDeviceData! */
        cpyguid = (LPGUID)SDL_malloc(sizeof(GUID));
        if (!cpyguid) {
            SDL_free(devidlist);
            SDL_free(devidcopy);
            return; /* oh well. */
        }
        SDL_memcpy(cpyguid, dsoundguid, sizeof(GUID));
        driverdata = cpyguid;
    } else {
        cpyguid = NULL;
        driverdata = devidcopy;
    }

    devidlist->str = devidcopy;
    devidlist->guid = cpyguid;
    devidlist->next = deviceid_list;
    deviceid_list = devidlist;

    SDL_zero(spec);
    spec.channels = (Uint8)fmt->Format.nChannels;
    spec.freq = fmt->Format.nSamplesPerSec;
    spec.format = WaveFormatToSDLFormat((WAVEFORMATEX *)fmt);
    SDL_AddAudioDevice(iscapture, devname, &spec, driverdata);
}

/* We need a COM subclass of IMMNotificationClient for hotplug support, which is
   easy in C++, but we have to tapdance more to make work in C.
   Thanks to this page for coaching on how to make this work:
     https://www.codeproject.com/Articles/13601/COM-in-plain-C */

typedef struct SDLMMNotificationClient
{
    const IMMNotificationClientVtbl *lpVtbl;
    SDL_atomic_t refcount;
    SDL_bool useguid;
} SDLMMNotificationClient;

static HRESULT STDMETHODCALLTYPE SDLMMNotificationClient_QueryInterface(IMMNotificationClient *this, REFIID iid, void **ppv)
{
    if ((WIN_IsEqualIID(iid, &IID_IUnknown)) || (WIN_IsEqualIID(iid, &SDL_IID_IMMNotificationClient))) {
        *ppv = this;
        this->lpVtbl->AddRef(this);
        return S_OK;
    }

    *ppv = NULL;
    return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE SDLMMNotificationClient_AddRef(IMMNotificationClient *ithis)
{
    SDLMMNotificationClient *this = (SDLMMNotificationClient *)ithis;
    return (ULONG)(SDL_AtomicIncRef(&this->refcount) + 1);
}

static ULONG STDMETHODCALLTYPE SDLMMNotificationClient_Release(IMMNotificationClient *ithis)
{
    /* this is a static object; we don't ever free it. */
    SDLMMNotificationClient *this = (SDLMMNotificationClient *)ithis;
    const ULONG retval = SDL_AtomicDecRef(&this->refcount);
    if (retval == 0) {
        SDL_AtomicSet(&this->refcount, 0); /* uhh... */
        return 0;
    }
    return retval - 1;
}

/* These are the entry points called when WASAPI device endpoints change. */
static HRESULT STDMETHODCALLTYPE SDLMMNotificationClient_OnDefaultDeviceChanged(IMMNotificationClient *ithis, EDataFlow flow, ERole role, LPCWSTR pwstrDeviceId)
{
    if (role != SDL_IMMDevice_role) {
        return S_OK; /* ignore it. */
    }

    /* Increment the "generation," so opened devices will pick this up in their threads. */
    switch (flow) {
    case eRender:
        SDL_AtomicAdd(&SDL_IMMDevice_DefaultPlaybackGeneration, 1);
        break;

    case eCapture:
        SDL_AtomicAdd(&SDL_IMMDevice_DefaultCaptureGeneration, 1);
        break;

    case eAll:
        SDL_AtomicAdd(&SDL_IMMDevice_DefaultPlaybackGeneration, 1);
        SDL_AtomicAdd(&SDL_IMMDevice_DefaultCaptureGeneration, 1);
        break;

    default:
        SDL_assert(!"uhoh, unexpected OnDefaultDeviceChange flow!");
        break;
    }

    return S_OK;
}

static HRESULT STDMETHODCALLTYPE SDLMMNotificationClient_OnDeviceAdded(IMMNotificationClient *ithis, LPCWSTR pwstrDeviceId)
{
    /* we ignore this; devices added here then progress to ACTIVE, if appropriate, in
       OnDeviceStateChange, making that a better place to deal with device adds. More
       importantly: the first time you plug in a USB audio device, this callback will
       fire, but when you unplug it, it isn't removed (it's state changes to NOTPRESENT).
       Plugging it back in won't fire this callback again. */
    return S_OK;
}

static HRESULT STDMETHODCALLTYPE SDLMMNotificationClient_OnDeviceRemoved(IMMNotificationClient *ithis, LPCWSTR pwstrDeviceId)
{
    /* See notes in OnDeviceAdded handler about why we ignore this. */
    return S_OK;
}

static HRESULT STDMETHODCALLTYPE SDLMMNotificationClient_OnDeviceStateChanged(IMMNotificationClient *ithis, LPCWSTR pwstrDeviceId, DWORD dwNewState)
{
    IMMDevice *device = NULL;

    if (SUCCEEDED(IMMDeviceEnumerator_GetDevice(enumerator, pwstrDeviceId, &device))) {
        IMMEndpoint *endpoint = NULL;
        if (SUCCEEDED(IMMDevice_QueryInterface(device, &SDL_IID_IMMEndpoint, (void **)&endpoint))) {
            EDataFlow flow;
            if (SUCCEEDED(IMMEndpoint_GetDataFlow(endpoint, &flow))) {
                const SDL_bool iscapture = (flow == eCapture);
                const SDLMMNotificationClient *client = (SDLMMNotificationClient *)ithis;
                if (dwNewState == DEVICE_STATE_ACTIVE) {
                    char *utf8dev;
                    WAVEFORMATEXTENSIBLE fmt;
                    GUID dsoundguid;
                    GetMMDeviceInfo(device, &utf8dev, &fmt, &dsoundguid);
                    if (utf8dev) {
                        SDL_IMMDevice_Add(iscapture, utf8dev, &fmt, pwstrDeviceId, &dsoundguid, client->useguid);
                        SDL_free(utf8dev);
                    }
                } else {
                    SDL_IMMDevice_Remove(iscapture, pwstrDeviceId, client->useguid);
                }
            }
            IMMEndpoint_Release(endpoint);
        }
        IMMDevice_Release(device);
    }

    return S_OK;
}

static HRESULT STDMETHODCALLTYPE SDLMMNotificationClient_OnPropertyValueChanged(IMMNotificationClient *this, LPCWSTR pwstrDeviceId, const PROPERTYKEY key)
{
    return S_OK; /* we don't care about these. */
}

static const IMMNotificationClientVtbl notification_client_vtbl = {
    SDLMMNotificationClient_QueryInterface,
    SDLMMNotificationClient_AddRef,
    SDLMMNotificationClient_Release,
    SDLMMNotificationClient_OnDeviceStateChanged,
    SDLMMNotificationClient_OnDeviceAdded,
    SDLMMNotificationClient_OnDeviceRemoved,
    SDLMMNotificationClient_OnDefaultDeviceChanged,
    SDLMMNotificationClient_OnPropertyValueChanged
};

static SDLMMNotificationClient notification_client = { &notification_client_vtbl, { 1 } };

int SDL_IMMDevice_Init(void)
{
    HRESULT ret;

    SDL_AtomicSet(&SDL_IMMDevice_DefaultPlaybackGeneration, 1);
    SDL_AtomicSet(&SDL_IMMDevice_DefaultCaptureGeneration, 1);

    /* just skip the discussion with COM here. */
    if (!WIN_IsWindowsVistaOrGreater()) {
        return SDL_SetError("WASAPI support requires Windows Vista or later");
    }

    if (FAILED(WIN_CoInitialize())) {
        return SDL_SetError("WASAPI: CoInitialize() failed");
    }

    ret = CoCreateInstance(&SDL_CLSID_MMDeviceEnumerator, NULL, CLSCTX_INPROC_SERVER, &SDL_IID_IMMDeviceEnumerator, (LPVOID *)&enumerator);
    if (FAILED(ret)) {
        WIN_CoUninitialize();
        return WIN_SetErrorFromHRESULT("WASAPI CoCreateInstance(MMDeviceEnumerator)", ret);
    }
    return 0;
}

void SDL_IMMDevice_Quit(void)
{
    DevIdList *devidlist;
    DevIdList *next;

    if (enumerator) {
        IMMDeviceEnumerator_UnregisterEndpointNotificationCallback(enumerator, (IMMNotificationClient *)&notification_client);
        IMMDeviceEnumerator_Release(enumerator);
        enumerator = NULL;
    }

    WIN_CoUninitialize();

    for (devidlist = deviceid_list; devidlist; devidlist = next) {
        next = devidlist->next;
        SDL_free(devidlist->str);
        SDL_free(devidlist);
    }
    deviceid_list = NULL;
}

int SDL_IMMDevice_Get(LPCWSTR devid, IMMDevice **device, SDL_bool iscapture)
{
    const Uint64 timeout = SDL_GetTicks64() + 8000;  /* intel's audio drivers can fail for up to EIGHT SECONDS after a device is connected or we wake from sleep. */
    HRESULT ret;

    SDL_assert(device != NULL);

    while (SDL_TRUE) {
        if (!devid) {
            const EDataFlow dataflow = iscapture ? eCapture : eRender;
            ret = IMMDeviceEnumerator_GetDefaultAudioEndpoint(enumerator, dataflow, SDL_IMMDevice_role, device);
        } else {
            ret = IMMDeviceEnumerator_GetDevice(enumerator, devid, device);
        }

        if (SUCCEEDED(ret)) {
            break;
        }

        if (ret == E_NOTFOUND) {
            const Uint64 now = SDL_GetTicks64();
            if (timeout > now) {
                const Uint64 ticksleft = timeout - now;
                SDL_Delay((Uint32)SDL_min(ticksleft, 300));   /* wait awhile and try again. */
                continue;
            }
        }

        return WIN_SetErrorFromHRESULT("WASAPI can't find requested audio endpoint", ret);
    }
    return 0;
}

typedef struct
{
    LPWSTR devid;
    char *devname;
    WAVEFORMATEXTENSIBLE fmt;
    GUID dsoundguid;
} EndpointItem;

static int SDLCALL sort_endpoints(const void *_a, const void *_b)
{
    LPWSTR a = ((const EndpointItem *)_a)->devid;
    LPWSTR b = ((const EndpointItem *)_b)->devid;
    if (!a && !b) {
        return 0;
    } else if (!a && b) {
        return -1;
    } else if (a && !b) {
        return 1;
    }

    while (SDL_TRUE) {
        if (*a < *b) {
            return -1;
        } else if (*a > *b) {
            return 1;
        } else if (*a == 0) {
            break;
        }
        a++;
        b++;
    }

    return 0;
}

static void EnumerateEndpointsForFlow(const SDL_bool iscapture)
{
    IMMDeviceCollection *collection = NULL;
    EndpointItem *items;
    UINT i, total;

    /* Note that WASAPI separates "adapter devices" from "audio endpoint devices"
       ...one adapter device ("SoundBlaster Pro") might have multiple endpoint devices ("Speakers", "Line-Out"). */

    if (FAILED(IMMDeviceEnumerator_EnumAudioEndpoints(enumerator, iscapture ? eCapture : eRender, DEVICE_STATE_ACTIVE, &collection))) {
        return;
    }

    if (FAILED(IMMDeviceCollection_GetCount(collection, &total))) {
        IMMDeviceCollection_Release(collection);
        return;
    }

    items = (EndpointItem *)SDL_calloc(total, sizeof(EndpointItem));
    if (!items) {
        return; /* oh well. */
    }

    for (i = 0; i < total; i++) {
        EndpointItem *item = items + i;
        IMMDevice *device = NULL;
        if (SUCCEEDED(IMMDeviceCollection_Item(collection, i, &device))) {
            if (SUCCEEDED(IMMDevice_GetId(device, &item->devid))) {
                GetMMDeviceInfo(device, &item->devname, &item->fmt, &item->dsoundguid);
            }
            IMMDevice_Release(device);
        }
    }

    /* sort the list of devices by their guid so list is consistent between runs */
    SDL_qsort(items, total, sizeof(*items), sort_endpoints);

    /* Send the sorted list on to the SDL's higher level. */
    for (i = 0; i < total; i++) {
        EndpointItem *item = items + i;
        if ((item->devid) && (item->devname)) {
            SDL_IMMDevice_Add(iscapture, item->devname, &item->fmt, item->devid, &item->dsoundguid, notification_client.useguid);
        }
        SDL_free(item->devname);
        CoTaskMemFree(item->devid);
    }

    SDL_free(items);
    IMMDeviceCollection_Release(collection);
}

void SDL_IMMDevice_EnumerateEndpoints(SDL_bool useguid)
{
    notification_client.useguid = useguid;

    EnumerateEndpointsForFlow(SDL_FALSE); /* playback */
    EnumerateEndpointsForFlow(SDL_TRUE);  /* capture */

    /* if this fails, we just won't get hotplug events. Carry on anyhow. */
    IMMDeviceEnumerator_RegisterEndpointNotificationCallback(enumerator, (IMMNotificationClient *)&notification_client);
}

int SDL_IMMDevice_GetDefaultAudioInfo(char **name, SDL_AudioSpec *spec, int iscapture)
{
    WAVEFORMATEXTENSIBLE fmt;
    IMMDevice *device = NULL;
    char *filler;
    GUID morefiller;
    const EDataFlow dataflow = iscapture ? eCapture : eRender;
    HRESULT ret = IMMDeviceEnumerator_GetDefaultAudioEndpoint(enumerator, dataflow, SDL_IMMDevice_role, &device);

    if (FAILED(ret)) {
        SDL_assert(device == NULL);
        return WIN_SetErrorFromHRESULT("WASAPI can't find default audio endpoint", ret);
    }

    if (!name) {
        name = &filler;
    }

    SDL_zero(fmt);
    GetMMDeviceInfo(device, name, &fmt, &morefiller);
    IMMDevice_Release(device);

    if (name == &filler) {
        SDL_free(filler);
    }

    SDL_zerop(spec);
    spec->channels = (Uint8)fmt.Format.nChannels;
    spec->freq = fmt.Format.nSamplesPerSec;
    spec->format = WaveFormatToSDLFormat((WAVEFORMATEX *)&fmt);
    return 0;
}

SDL_AudioFormat WaveFormatToSDLFormat(WAVEFORMATEX *waveformat)
{
    if ((waveformat->wFormatTag == WAVE_FORMAT_IEEE_FLOAT) && (waveformat->wBitsPerSample == 32)) {
        return AUDIO_F32SYS;
    } else if ((waveformat->wFormatTag == WAVE_FORMAT_PCM) && (waveformat->wBitsPerSample == 16)) {
        return AUDIO_S16SYS;
    } else if ((waveformat->wFormatTag == WAVE_FORMAT_PCM) && (waveformat->wBitsPerSample == 32)) {
        return AUDIO_S32SYS;
    } else if (waveformat->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
        const WAVEFORMATEXTENSIBLE *ext = (const WAVEFORMATEXTENSIBLE *)waveformat;
        if ((SDL_memcmp(&ext->SubFormat, &SDL_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, sizeof(GUID)) == 0) && (waveformat->wBitsPerSample == 32)) {
            return AUDIO_F32SYS;
        } else if ((SDL_memcmp(&ext->SubFormat, &SDL_KSDATAFORMAT_SUBTYPE_PCM, sizeof(GUID)) == 0) && (waveformat->wBitsPerSample == 16)) {
            return AUDIO_S16SYS;
        } else if ((SDL_memcmp(&ext->SubFormat, &SDL_KSDATAFORMAT_SUBTYPE_PCM, sizeof(GUID)) == 0) && (waveformat->wBitsPerSample == 32)) {
            return AUDIO_S32SYS;
        }
    }
    return 0;
}

#endif /* (defined(__WIN32__) || defined(__GDK__)) && HAVE_MMDEVICEAPI_H */
