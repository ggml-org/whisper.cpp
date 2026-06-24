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

#ifdef SDL_HAPTIC_IOKIT

#include "SDL_stdinc.h"
#include "SDL_haptic.h"
#include "../SDL_syshaptic.h"
#include "SDL_joystick.h"
#include "../../joystick/SDL_sysjoystick.h"     /* For the real SDL_Joystick */
#include "../../joystick/darwin/SDL_iokitjoystick_c.h"    /* For joystick hwdata */
#include "SDL_syshaptic_c.h"

#include <IOKit/IOKitLib.h>
#include <IOKit/hid/IOHIDKeys.h>
#include <IOKit/hid/IOHIDUsageTables.h>
#include <ForceFeedback/ForceFeedback.h>
#include <ForceFeedback/ForceFeedbackConstants.h>

#ifndef IO_OBJECT_NULL
#define IO_OBJECT_NULL ((io_service_t)0)
#endif

/*
 * List of available haptic devices.
 */
typedef struct SDL_hapticlist_item
{
    char name[256]; /* Name of the device. */

    io_service_t dev;   /* Node we use to create the device. */
    SDL_Haptic *haptic; /* Haptic currently associated with it. */

    /* Usage pages for determining if it's a mouse or not. */
    long usage;
    long usagePage;

    struct SDL_hapticlist_item *next;
} SDL_hapticlist_item;

/*
 * Haptic system hardware data.
 */
struct haptic_hwdata
{
    FFDeviceObjectReference device; /* Hardware device. */
    UInt8 axes[3];
};

/*
 * Haptic system effect data.
 */
struct haptic_hweffect
{
    FFEffectObjectReference ref; /* Reference. */
    struct FFEFFECT effect;      /* Hardware effect. */
};

/*
 * Prototypes.
 */
static void SDL_SYS_HapticFreeFFEFFECT(FFEFFECT *effect, int type);
static int HIDGetDeviceProduct(io_service_t dev, char *name);

static SDL_hapticlist_item *SDL_hapticlist = NULL;
static SDL_hapticlist_item *SDL_hapticlist_tail = NULL;
static int numhaptics = -1;

/*
 * Like strerror but for force feedback errors.
 */
static const char *FFStrError(unsigned int err)
{
    switch (err) {
    case FFERR_DEVICEFULL:
        return "device full";
    /* This should be valid, but for some reason isn't defined... */
    /* case FFERR_DEVICENOTREG:
        return "device not registered"; */
    case FFERR_DEVICEPAUSED:
        return "device paused";
    case FFERR_DEVICERELEASED:
        return "device released";
    case FFERR_EFFECTPLAYING:
        return "effect playing";
    case FFERR_EFFECTTYPEMISMATCH:
        return "effect type mismatch";
    case FFERR_EFFECTTYPENOTSUPPORTED:
        return "effect type not supported";
    case FFERR_GENERIC:
        return "undetermined error";
    case FFERR_HASEFFECTS:
        return "device has effects";
    case FFERR_INCOMPLETEEFFECT:
        return "incomplete effect";
    case FFERR_INTERNAL:
        return "internal fault";
    case FFERR_INVALIDDOWNLOADID:
        return "invalid download id";
    case FFERR_INVALIDPARAM:
        return "invalid parameter";
    case FFERR_MOREDATA:
        return "more data";
    case FFERR_NOINTERFACE:
        return "interface not supported";
    case FFERR_NOTDOWNLOADED:
        return "effect is not downloaded";
    case FFERR_NOTINITIALIZED:
        return "object has not been initialized";
    case FFERR_OUTOFMEMORY:
        return "out of memory";
    case FFERR_UNPLUGGED:
        return "device is unplugged";
    case FFERR_UNSUPPORTED:
        return "function call unsupported";
    case FFERR_UNSUPPORTEDAXIS:
        return "axis unsupported";

    default:
        return "unknown error";
    }
}

/*
 * Initializes the haptic subsystem.
 */
int SDL_SYS_HapticInit(void)
{
    IOReturn result;
    io_iterator_t iter;
    CFDictionaryRef match;
    io_service_t device;

    if (numhaptics != -1) {
        return SDL_SetError("Haptic subsystem already initialized!");
    }
    numhaptics = 0;

    /* Get HID devices. */
    match = IOServiceMatching(kIOHIDDeviceKey);
    if (!match) {
        return SDL_SetError("Haptic: Failed to get IOServiceMatching.");
    }

    /* Now search I/O Registry for matching devices. */
    result = IOServiceGetMatchingServices(kIOMainPortDefault, match, &iter);
    if (result != kIOReturnSuccess) {
        return SDL_SetError("Haptic: Couldn't create a HID object iterator.");
    }
    /* IOServiceGetMatchingServices consumes dictionary. */

    if (!IOIteratorIsValid(iter)) { /* No iterator. */
        return 0;
    }

    while ((device = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
        MacHaptic_MaybeAddDevice(device);
        /* always release as the AddDevice will retain IF it's a forcefeedback device */
        IOObjectRelease(device);
    }
    IOObjectRelease(iter);

    return numhaptics;
}

int SDL_SYS_NumHaptics(void)
{
    return numhaptics;
}

static SDL_hapticlist_item *HapticByDevIndex(int device_index)
{
    SDL_hapticlist_item *item = SDL_hapticlist;

    if ((device_index < 0) || (device_index >= numhaptics)) {
        return NULL;
    }

    while (device_index > 0) {
        SDL_assert(item != NULL);
        --device_index;
        item = item->next;
    }

    return item;
}

int MacHaptic_MaybeAddDevice(io_object_t device)
{
    IOReturn result;
    CFMutableDictionaryRef hidProperties;
    CFTypeRef refCF;
    SDL_hapticlist_item *item;

    if (numhaptics == -1) {
        return -1; /* not initialized. We'll pick these up on enumeration if we init later. */
    }

    /* Check for force feedback. */
    if (FFIsForceFeedback(device) != FF_OK) {
        return -1;
    }

    /* Make sure we don't already have it */
    for (item = SDL_hapticlist; item; item = item->next) {
        if (IOObjectIsEqualTo((io_object_t)item->dev, device)) {
            /* Already added */
            return -1;
        }
    }

    item = (SDL_hapticlist_item *)SDL_calloc(1, sizeof(SDL_hapticlist_item));
    if (!item) {
        return SDL_SetError("Could not allocate haptic storage");
    }

    /* retain it as we are going to keep it around a while */
    IOObjectRetain(device);

    /* Set basic device data. */
    HIDGetDeviceProduct(device, item->name);
    item->dev = device;

    /* Set usage pages. */
    hidProperties = 0;
    refCF = 0;
    result = IORegistryEntryCreateCFProperties(device,
                                               &hidProperties,
                                               kCFAllocatorDefault,
                                               kNilOptions);
    if ((result == KERN_SUCCESS) && hidProperties) {
        refCF = CFDictionaryGetValue(hidProperties,
                                     CFSTR(kIOHIDPrimaryUsagePageKey));
        if (refCF) {
            if (!CFNumberGetValue(refCF, kCFNumberLongType, &item->usagePage)) {
                SDL_SetError("Haptic: Receiving device's usage page.");
            }
            refCF = CFDictionaryGetValue(hidProperties,
                                         CFSTR(kIOHIDPrimaryUsageKey));
            if (refCF) {
                if (!CFNumberGetValue(refCF, kCFNumberLongType, &item->usage)) {
                    SDL_SetError("Haptic: Receiving device's usage.");
                }
            }
        }
        CFRelease(hidProperties);
    }

    if (!SDL_hapticlist_tail) {
        SDL_hapticlist = SDL_hapticlist_tail = item;
    } else {
        SDL_hapticlist_tail->next = item;
        SDL_hapticlist_tail = item;
    }

    /* Device has been added. */
    ++numhaptics;

    return numhaptics;
}

int MacHaptic_MaybeRemoveDevice(io_object_t device)
{
    SDL_hapticlist_item *item;
    SDL_hapticlist_item *prev = NULL;

    if (numhaptics == -1) {
        return -1; /* not initialized. ignore this. */
    }

    for (item = SDL_hapticlist; item; item = item->next) {
        /* found it, remove it. */
        if (IOObjectIsEqualTo((io_object_t)item->dev, device)) {
            const int retval = item->haptic ? item->haptic->index : -1;

            if (prev) {
                prev->next = item->next;
            } else {
                SDL_assert(SDL_hapticlist == item);
                SDL_hapticlist = item->next;
            }
            if (item == SDL_hapticlist_tail) {
                SDL_hapticlist_tail = prev;
            }

            /* Need to decrement the haptic count */
            --numhaptics;
            /* !!! TODO: Send a haptic remove event? */

            IOObjectRelease(item->dev);
            SDL_free(item);
            return retval;
        }
        prev = item;
    }

    return -1;
}

/*
 * Return the name of a haptic device, does not need to be opened.
 */
const char *SDL_SYS_HapticName(int index)
{
    SDL_hapticlist_item *item;
    item = HapticByDevIndex(index);
    return item->name;
}

/*
 * Gets the device's product name.
 */
static int HIDGetDeviceProduct(io_service_t dev, char *name)
{
    CFMutableDictionaryRef hidProperties, usbProperties;
    io_registry_entry_t parent1, parent2;
    kern_return_t ret;

    hidProperties = usbProperties = 0;

    ret = IORegistryEntryCreateCFProperties(dev, &hidProperties,
                                            kCFAllocatorDefault, kNilOptions);
    if ((ret != KERN_SUCCESS) || !hidProperties) {
        return SDL_SetError("Haptic: Unable to create CFProperties.");
    }

    /* Mac OS X currently is not mirroring all USB properties to HID page so need to look at USB device page also
     * get dictionary for USB properties: step up two levels and get CF dictionary for USB properties
     */
    if ((KERN_SUCCESS ==
         IORegistryEntryGetParentEntry(dev, kIOServicePlane, &parent1)) &&
        (KERN_SUCCESS ==
         IORegistryEntryGetParentEntry(parent1, kIOServicePlane, &parent2)) &&
        (KERN_SUCCESS ==
         IORegistryEntryCreateCFProperties(parent2, &usbProperties,
                                           kCFAllocatorDefault,
                                           kNilOptions))) {
        if (usbProperties) {
            CFTypeRef refCF = 0;
            /* get device info
             * try hid dictionary first, if fail then go to USB dictionary
             */

            /* Get product name */
            refCF = CFDictionaryGetValue(hidProperties, CFSTR(kIOHIDProductKey));
            if (!refCF) {
                refCF = CFDictionaryGetValue(usbProperties,
                                             CFSTR("USB Product Name"));
            }
            if (refCF) {
                if (!CFStringGetCString(refCF, name, 256,
                                        CFStringGetSystemEncoding())) {
                    return SDL_SetError("Haptic: CFStringGetCString error retrieving pDevice->product.");
                }
            }

            CFRelease(usbProperties);
        } else {
            return SDL_SetError("Haptic: IORegistryEntryCreateCFProperties failed to create usbProperties.");
        }

        /* Release stuff. */
        if (kIOReturnSuccess != IOObjectRelease(parent2)) {
            SDL_SetError("Haptic: IOObjectRelease error with parent2.");
        }
        if (kIOReturnSuccess != IOObjectRelease(parent1)) {
            SDL_SetError("Haptic: IOObjectRelease error with parent1.");
        }
    } else {
        return SDL_SetError("Haptic: Error getting registry entries.");
    }

    return 0;
}

#define FF_TEST(ff, s)                    \
    if (features.supportedEffects & (ff)) \
    supported |= (s)
/*
 * Gets supported features.
 */
static unsigned int GetSupportedFeatures(SDL_Haptic *haptic)
{
    HRESULT ret;
    FFDeviceObjectReference device;
    FFCAPABILITIES features;
    unsigned int supported;
    Uint32 val;

    device = haptic->hwdata->device;

    ret = FFDeviceGetForceFeedbackCapabilities(device, &features);
    if (ret != FF_OK) {
        return SDL_SetError("Haptic: Unable to get device's supported features.");
    }

    supported = 0;

    /* Get maximum effects. */
    haptic->neffects = features.storageCapacity;
    haptic->nplaying = features.playbackCapacity;

    /* Test for effects. */
    FF_TEST(FFCAP_ET_CONSTANTFORCE, SDL_HAPTIC_CONSTANT);
    FF_TEST(FFCAP_ET_RAMPFORCE, SDL_HAPTIC_RAMP);
    /* !!! FIXME: put this back when we have more bits in 2.1 */
    /* FF_TEST(FFCAP_ET_SQUARE, SDL_HAPTIC_SQUARE); */
    FF_TEST(FFCAP_ET_SINE, SDL_HAPTIC_SINE);
    FF_TEST(FFCAP_ET_TRIANGLE, SDL_HAPTIC_TRIANGLE);
    FF_TEST(FFCAP_ET_SAWTOOTHUP, SDL_HAPTIC_SAWTOOTHUP);
    FF_TEST(FFCAP_ET_SAWTOOTHDOWN, SDL_HAPTIC_SAWTOOTHDOWN);
    FF_TEST(FFCAP_ET_SPRING, SDL_HAPTIC_SPRING);
    FF_TEST(FFCAP_ET_DAMPER, SDL_HAPTIC_DAMPER);
    FF_TEST(FFCAP_ET_INERTIA, SDL_HAPTIC_INERTIA);
    FF_TEST(FFCAP_ET_FRICTION, SDL_HAPTIC_FRICTION);
    FF_TEST(FFCAP_ET_CUSTOMFORCE, SDL_HAPTIC_CUSTOM);

    /* Check if supports gain. */
    ret = FFDeviceGetForceFeedbackProperty(device, FFPROP_FFGAIN,
                                           &val, sizeof(val));
    if (ret == FF_OK) {
        supported |= SDL_HAPTIC_GAIN;
    } else if (ret != FFERR_UNSUPPORTED) {
        return SDL_SetError("Haptic: Unable to get if device supports gain: %s.",
                            FFStrError(ret));
    }

    /* Checks if supports autocenter. */
    ret = FFDeviceGetForceFeedbackProperty(device, FFPROP_AUTOCENTER,
                                           &val, sizeof(val));
    if (ret == FF_OK) {
        supported |= SDL_HAPTIC_AUTOCENTER;
    } else if (ret != FFERR_UNSUPPORTED) {
        return SDL_SetError("Haptic: Unable to get if device supports autocenter: %s.",
                            FFStrError(ret));
    }

    /* Check for axes, we have an artificial limit on axes */
    haptic->naxes = ((features.numFfAxes) > 3) ? 3 : features.numFfAxes;
    /* Actually store the axes we want to use */
    SDL_memcpy(haptic->hwdata->axes, features.ffAxes,
               haptic->naxes * sizeof(Uint8));

    /* Always supported features. */
    supported |= SDL_HAPTIC_STATUS | SDL_HAPTIC_PAUSE;

    haptic->supported = supported;
    return 0;
}

/*
 * Opens the haptic device from the file descriptor.
 */
static int SDL_SYS_HapticOpenFromService(SDL_Haptic *haptic, io_service_t service)
{
    HRESULT ret;
    int ret2;

    /* Allocate the hwdata */
    haptic->hwdata = (struct haptic_hwdata *)
        SDL_malloc(sizeof(*haptic->hwdata));
    if (!haptic->hwdata) {
        SDL_OutOfMemory();
        goto creat_err;
    }
    SDL_memset(haptic->hwdata, 0, sizeof(*haptic->hwdata));

    /* Open the device */
    ret = FFCreateDevice(service, &haptic->hwdata->device);
    if (ret != FF_OK) {
        SDL_SetError("Haptic: Unable to create device from service: %s.",
                     FFStrError(ret));
        goto creat_err;
    }

    /* Get supported features. */
    ret2 = GetSupportedFeatures(haptic);
    if (ret2 < 0) {
        goto open_err;
    }

    /* Reset and then enable actuators. */
    ret = FFDeviceSendForceFeedbackCommand(haptic->hwdata->device,
                                           FFSFFC_RESET);
    if (ret != FF_OK) {
        SDL_SetError("Haptic: Unable to reset device: %s.", FFStrError(ret));
        goto open_err;
    }
    ret = FFDeviceSendForceFeedbackCommand(haptic->hwdata->device,
                                           FFSFFC_SETACTUATORSON);
    if (ret != FF_OK) {
        SDL_SetError("Haptic: Unable to enable actuators: %s.",
                     FFStrError(ret));
        goto open_err;
    }

    /* Allocate effects memory. */
    haptic->effects = (struct haptic_effect *)
        SDL_malloc(sizeof(struct haptic_effect) * haptic->neffects);
    if (!haptic->effects) {
        SDL_OutOfMemory();
        goto open_err;
    }
    /* Clear the memory */
    SDL_memset(haptic->effects, 0,
               sizeof(struct haptic_effect) * haptic->neffects);

    return 0;

    /* Error handling */
open_err:
    FFReleaseDevice(haptic->hwdata->device);
creat_err:
    if (haptic->hwdata) {
        SDL_free(haptic->hwdata);
        haptic->hwdata = NULL;
    }
    return -1;
}

/*
 * Opens a haptic device for usage.
 */
int SDL_SYS_HapticOpen(SDL_Haptic *haptic)
{
    SDL_hapticlist_item *item;
    item = HapticByDevIndex(haptic->index);

    return SDL_SYS_HapticOpenFromService(haptic, item->dev);
}

/*
 * Opens a haptic device from first mouse it finds for usage.
 */
int SDL_SYS_HapticMouse(void)
{
    int device_index = 0;
    SDL_hapticlist_item *item;

    for (item = SDL_hapticlist; item; item = item->next) {
        if ((item->usagePage == kHIDPage_GenericDesktop) &&
            (item->usage == kHIDUsage_GD_Mouse)) {
            return device_index;
        }
        ++device_index;
    }

    return -1;
}

/*
 * Checks to see if a joystick has haptic features.
 */
int SDL_SYS_JoystickIsHaptic(SDL_Joystick *joystick)
{
#ifdef SDL_JOYSTICK_IOKIT
    if (joystick->driver != &SDL_DARWIN_JoystickDriver) {
        return SDL_FALSE;
    }
    if (joystick->hwdata->ffservice != 0) {
        return SDL_TRUE;
    }
#endif
    return SDL_FALSE;
}

/*
 * Checks to see if the haptic device and joystick are in reality the same.
 */
int SDL_SYS_JoystickSameHaptic(SDL_Haptic *haptic, SDL_Joystick *joystick)
{
#ifdef SDL_JOYSTICK_IOKIT
    if (joystick->driver != &SDL_DARWIN_JoystickDriver) {
        return 0;
    }
    if (IOObjectIsEqualTo((io_object_t)((size_t)haptic->hwdata->device),
                          joystick->hwdata->ffservice)) {
        return 1;
    }
#endif
    return 0;
}

/*
 * Opens a SDL_Haptic from a SDL_Joystick.
 */
int SDL_SYS_HapticOpenFromJoystick(SDL_Haptic *haptic, SDL_Joystick *joystick)
{
#ifdef SDL_JOYSTICK_IOKIT
    int device_index = 0;
    SDL_hapticlist_item *item;

    if (joystick->driver != &SDL_DARWIN_JoystickDriver) {
        return -1;
    }
    for (item = SDL_hapticlist; item; item = item->next) {
        if (IOObjectIsEqualTo((io_object_t)item->dev,
                              joystick->hwdata->ffservice)) {
            haptic->index = device_index;
            break;
        }
        ++device_index;
    }

    return SDL_SYS_HapticOpenFromService(haptic, joystick->hwdata->ffservice);
#else
    return -1;
#endif
}

/*
 * Closes the haptic device.
 */
void SDL_SYS_HapticClose(SDL_Haptic *haptic)
{
    if (haptic->hwdata) {

        /* Free Effects. */
        SDL_free(haptic->effects);
        haptic->effects = NULL;
        haptic->neffects = 0;

        /* Clean up */
        FFReleaseDevice(haptic->hwdata->device);

        /* Free */
        SDL_free(haptic->hwdata);
        haptic->hwdata = NULL;
    }
}

/*
 * Clean up after system specific haptic stuff
 */
void SDL_SYS_HapticQuit(void)
{
    SDL_hapticlist_item *item;
    SDL_hapticlist_item *next = NULL;

    for (item = SDL_hapticlist; item; item = next) {
        next = item->next;
        /* Opened and not closed haptics are leaked, this is on purpose.
         * Close your haptic devices after usage. */

        /* Free the io_service_t */
        IOObjectRelease(item->dev);
        SDL_free(item);
    }

    numhaptics = -1;
    SDL_hapticlist = NULL;
    SDL_hapticlist_tail = NULL;
}

/*
 * Converts an SDL trigger button to an FFEFFECT trigger button.
 */
static DWORD FFGetTriggerButton(Uint16 button)
{
    DWORD dwTriggerButton;

    dwTriggerButton = FFEB_NOTRIGGER;

    if (button != 0) {
        dwTriggerButton = FFJOFS_BUTTON(button - 1);
    }

    return dwTriggerButton;
}

/*
 * Sets the direction.
 */
static int SDL_SYS_SetDirection(FFEFFECT *effect, SDL_HapticDirection *dir, int naxes)
{
    LONG *rglDir;

    /* Handle no axes a part. */
    if (naxes == 0) {
        effect->dwFlags |= FFEFF_SPHERICAL; /* Set as default. */
        effect->rglDirection = NULL;
        return 0;
    }

    /* Has axes. */
    rglDir = SDL_malloc(sizeof(LONG) * naxes);
    if (!rglDir) {
        return SDL_OutOfMemory();
    }
    SDL_memset(rglDir, 0, sizeof(LONG) * naxes);
    effect->rglDirection = rglDir;

    switch (dir->type) {
    case SDL_HAPTIC_POLAR:
        effect->dwFlags |= FFEFF_POLAR;
        rglDir[0] = dir->dir[0];
        return 0;
    case SDL_HAPTIC_CARTESIAN:
        effect->dwFlags |= FFEFF_CARTESIAN;
        rglDir[0] = dir->dir[0];
        if (naxes > 1) {
            rglDir[1] = dir->dir[1];
        }
        if (naxes > 2) {
            rglDir[2] = dir->dir[2];
        }
        return 0;
    case SDL_HAPTIC_SPHERICAL:
        effect->dwFlags |= FFEFF_SPHERICAL;
        rglDir[0] = dir->dir[0];
        if (naxes > 1) {
            rglDir[1] = dir->dir[1];
        }
        if (naxes > 2) {
            rglDir[2] = dir->dir[2];
        }
        return 0;
    case SDL_HAPTIC_STEERING_AXIS:
        effect->dwFlags |= FFEFF_CARTESIAN;
        rglDir[0] = 0;
        return 0;

    default:
        return SDL_SetError("Haptic: Unknown direction type.");
    }
}

/* Clamps and converts. */
#define CCONVERT(x) (((x) > 0x7FFF) ? 10000 : ((x)*10000) / 0x7FFF)
/* Just converts. */
#define CONVERT(x) (((x)*10000) / 0x7FFF)
/*
 * Creates the FFEFFECT from a SDL_HapticEffect.
 */
static int SDL_SYS_ToFFEFFECT(SDL_Haptic *haptic, FFEFFECT *dest, SDL_HapticEffect *src)
{
    int i;
    FFCONSTANTFORCE *constant = NULL;
    FFPERIODIC *periodic = NULL;
    FFCONDITION *condition = NULL; /* Actually an array of conditions - one per axis. */
    FFRAMPFORCE *ramp = NULL;
    FFCUSTOMFORCE *custom = NULL;
    FFENVELOPE *envelope = NULL;
    SDL_HapticConstant *hap_constant = NULL;
    SDL_HapticPeriodic *hap_periodic = NULL;
    SDL_HapticCondition *hap_condition = NULL;
    SDL_HapticRamp *hap_ramp = NULL;
    SDL_HapticCustom *hap_custom = NULL;
    DWORD *axes = NULL;

    /* Set global stuff. */
    SDL_memset(dest, 0, sizeof(FFEFFECT));
    dest->dwSize = sizeof(FFEFFECT);     /* Set the structure size. */
    dest->dwSamplePeriod = 0;            /* Not used by us. */
    dest->dwGain = 10000;                /* Gain is set globally, not locally. */
    dest->dwFlags = FFEFF_OBJECTOFFSETS; /* Seems obligatory. */

    /* Envelope. */
    envelope = SDL_malloc(sizeof(FFENVELOPE));
    if (!envelope) {
        return SDL_OutOfMemory();
    }
    SDL_memset(envelope, 0, sizeof(FFENVELOPE));
    dest->lpEnvelope = envelope;
    envelope->dwSize = sizeof(FFENVELOPE); /* Always should be this. */

    /* Axes. */
    if (src->constant.direction.type == SDL_HAPTIC_STEERING_AXIS) {
        dest->cAxes = 1;
    } else {
        dest->cAxes = haptic->naxes;
    }
    if (dest->cAxes > 0) {
        axes = SDL_malloc(sizeof(DWORD) * dest->cAxes);
        if (!axes) {
            return SDL_OutOfMemory();
        }
        axes[0] = haptic->hwdata->axes[0]; /* Always at least one axis. */
        if (dest->cAxes > 1) {
            axes[1] = haptic->hwdata->axes[1];
        }
        if (dest->cAxes > 2) {
            axes[2] = haptic->hwdata->axes[2];
        }
        dest->rgdwAxes = axes;
    }

    /* The big type handling switch, even bigger then Linux's version. */
    switch (src->type) {
    case SDL_HAPTIC_CONSTANT:
        hap_constant = &src->constant;
        constant = SDL_malloc(sizeof(FFCONSTANTFORCE));
        if (!constant) {
            return SDL_OutOfMemory();
        }
        SDL_memset(constant, 0, sizeof(FFCONSTANTFORCE));

        /* Specifics */
        constant->lMagnitude = CONVERT(hap_constant->level);
        dest->cbTypeSpecificParams = sizeof(FFCONSTANTFORCE);
        dest->lpvTypeSpecificParams = constant;

        /* Generics */
        dest->dwDuration = hap_constant->length * 1000; /* In microseconds. */
        dest->dwTriggerButton = FFGetTriggerButton(hap_constant->button);
        dest->dwTriggerRepeatInterval = hap_constant->interval;
        dest->dwStartDelay = hap_constant->delay * 1000; /* In microseconds. */

        /* Direction. */
        if (SDL_SYS_SetDirection(dest, &hap_constant->direction, dest->cAxes) < 0) {
            return -1;
        }

        /* Envelope */
        if ((hap_constant->attack_length == 0) && (hap_constant->fade_length == 0)) {
            SDL_free(envelope);
            dest->lpEnvelope = NULL;
        } else {
            envelope->dwAttackLevel = CCONVERT(hap_constant->attack_level);
            envelope->dwAttackTime = hap_constant->attack_length * 1000;
            envelope->dwFadeLevel = CCONVERT(hap_constant->fade_level);
            envelope->dwFadeTime = hap_constant->fade_length * 1000;
        }

        break;

    case SDL_HAPTIC_SINE:
    /* !!! FIXME: put this back when we have more bits in 2.1 */
    /* case SDL_HAPTIC_SQUARE: */
    case SDL_HAPTIC_TRIANGLE:
    case SDL_HAPTIC_SAWTOOTHUP:
    case SDL_HAPTIC_SAWTOOTHDOWN:
        hap_periodic = &src->periodic;
        periodic = SDL_malloc(sizeof(FFPERIODIC));
        if (!periodic) {
            return SDL_OutOfMemory();
        }
        SDL_memset(periodic, 0, sizeof(FFPERIODIC));

        /* Specifics */
        periodic->dwMagnitude = CONVERT(SDL_abs(hap_periodic->magnitude));
        periodic->lOffset = CONVERT(hap_periodic->offset);
        periodic->dwPhase =
            (hap_periodic->phase + (hap_periodic->magnitude < 0 ? 18000 : 0)) % 36000;
        periodic->dwPeriod = hap_periodic->period * 1000;
        dest->cbTypeSpecificParams = sizeof(FFPERIODIC);
        dest->lpvTypeSpecificParams = periodic;

        /* Generics */
        dest->dwDuration = hap_periodic->length * 1000; /* In microseconds. */
        dest->dwTriggerButton = FFGetTriggerButton(hap_periodic->button);
        dest->dwTriggerRepeatInterval = hap_periodic->interval;
        dest->dwStartDelay = hap_periodic->delay * 1000; /* In microseconds. */

        /* Direction. */
        if (SDL_SYS_SetDirection(dest, &hap_periodic->direction, dest->cAxes) < 0) {
            return -1;
        }

        /* Envelope */
        if ((hap_periodic->attack_length == 0) && (hap_periodic->fade_length == 0)) {
            SDL_free(envelope);
            dest->lpEnvelope = NULL;
        } else {
            envelope->dwAttackLevel = CCONVERT(hap_periodic->attack_level);
            envelope->dwAttackTime = hap_periodic->attack_length * 1000;
            envelope->dwFadeLevel = CCONVERT(hap_periodic->fade_level);
            envelope->dwFadeTime = hap_periodic->fade_length * 1000;
        }

        break;

    case SDL_HAPTIC_SPRING:
    case SDL_HAPTIC_DAMPER:
    case SDL_HAPTIC_INERTIA:
    case SDL_HAPTIC_FRICTION:
        hap_condition = &src->condition;
        if (dest->cAxes > 0) {
            condition = SDL_malloc(sizeof(FFCONDITION) * dest->cAxes);
            if (!condition) {
                return SDL_OutOfMemory();
            }
            SDL_memset(condition, 0, sizeof(FFCONDITION));

            /* Specifics */
            for (i = 0; i < dest->cAxes; i++) {
                condition[i].lOffset = CONVERT(hap_condition->center[i]);
                condition[i].lPositiveCoefficient =
                    CONVERT(hap_condition->right_coeff[i]);
                condition[i].lNegativeCoefficient =
                    CONVERT(hap_condition->left_coeff[i]);
                condition[i].dwPositiveSaturation =
                    CCONVERT(hap_condition->right_sat[i] / 2);
                condition[i].dwNegativeSaturation =
                    CCONVERT(hap_condition->left_sat[i] / 2);
                condition[i].lDeadBand = CCONVERT(hap_condition->deadband[i] / 2);
            }
        }

        dest->cbTypeSpecificParams = sizeof(FFCONDITION) * dest->cAxes;
        dest->lpvTypeSpecificParams = condition;

        /* Generics */
        dest->dwDuration = hap_condition->length * 1000; /* In microseconds. */
        dest->dwTriggerButton = FFGetTriggerButton(hap_condition->button);
        dest->dwTriggerRepeatInterval = hap_condition->interval;
        dest->dwStartDelay = hap_condition->delay * 1000; /* In microseconds. */

        /* Direction. */
        if (SDL_SYS_SetDirection(dest, &hap_condition->direction, dest->cAxes) < 0) {
            return -1;
        }

        /* Envelope - Not actually supported by most CONDITION implementations. */
        SDL_free(dest->lpEnvelope);
        dest->lpEnvelope = NULL;

        break;

    case SDL_HAPTIC_RAMP:
        hap_ramp = &src->ramp;
        ramp = SDL_malloc(sizeof(FFRAMPFORCE));
        if (!ramp) {
            return SDL_OutOfMemory();
        }
        SDL_memset(ramp, 0, sizeof(FFRAMPFORCE));

        /* Specifics */
        ramp->lStart = CONVERT(hap_ramp->start);
        ramp->lEnd = CONVERT(hap_ramp->end);
        dest->cbTypeSpecificParams = sizeof(FFRAMPFORCE);
        dest->lpvTypeSpecificParams = ramp;

        /* Generics */
        dest->dwDuration = hap_ramp->length * 1000; /* In microseconds. */
        dest->dwTriggerButton = FFGetTriggerButton(hap_ramp->button);
        dest->dwTriggerRepeatInterval = hap_ramp->interval;
        dest->dwStartDelay = hap_ramp->delay * 1000; /* In microseconds. */

        /* Direction. */
        if (SDL_SYS_SetDirection(dest, &hap_ramp->direction, dest->cAxes) < 0) {
            return -1;
        }

        /* Envelope */
        if ((hap_ramp->attack_length == 0) && (hap_ramp->fade_length == 0)) {
            SDL_free(envelope);
            dest->lpEnvelope = NULL;
        } else {
            envelope->dwAttackLevel = CCONVERT(hap_ramp->attack_level);
            envelope->dwAttackTime = hap_ramp->attack_length * 1000;
            envelope->dwFadeLevel = CCONVERT(hap_ramp->fade_level);
            envelope->dwFadeTime = hap_ramp->fade_length * 1000;
        }

        break;

    case SDL_HAPTIC_CUSTOM:
        hap_custom = &src->custom;
        custom = SDL_malloc(sizeof(FFCUSTOMFORCE));
        if (!custom) {
            return SDL_OutOfMemory();
        }
        SDL_memset(custom, 0, sizeof(FFCUSTOMFORCE));

        /* Specifics */
        custom->cChannels = hap_custom->channels;
        custom->dwSamplePeriod = hap_custom->period * 1000;
        custom->cSamples = hap_custom->samples;
        custom->rglForceData =
            SDL_malloc(sizeof(LONG) * custom->cSamples * custom->cChannels);
        for (i = 0; i < hap_custom->samples * hap_custom->channels; i++) { /* Copy data. */
            custom->rglForceData[i] = CCONVERT(hap_custom->data[i]);
        }
        dest->cbTypeSpecificParams = sizeof(FFCUSTOMFORCE);
        dest->lpvTypeSpecificParams = custom;

        /* Generics */
        dest->dwDuration = hap_custom->length * 1000; /* In microseconds. */
        dest->dwTriggerButton = FFGetTriggerButton(hap_custom->button);
        dest->dwTriggerRepeatInterval = hap_custom->interval;
        dest->dwStartDelay = hap_custom->delay * 1000; /* In microseconds. */

        /* Direction. */
        if (SDL_SYS_SetDirection(dest, &hap_custom->direction, dest->cAxes) <
            0) {
            return -1;
        }

        /* Envelope */
        if ((hap_custom->attack_length == 0) && (hap_custom->fade_length == 0)) {
            SDL_free(envelope);
            dest->lpEnvelope = NULL;
        } else {
            envelope->dwAttackLevel = CCONVERT(hap_custom->attack_level);
            envelope->dwAttackTime = hap_custom->attack_length * 1000;
            envelope->dwFadeLevel = CCONVERT(hap_custom->fade_level);
            envelope->dwFadeTime = hap_custom->fade_length * 1000;
        }

        break;

    default:
        return SDL_SetError("Haptic: Unknown effect type.");
    }

    return 0;
}

/*
 * Frees an FFEFFECT allocated by SDL_SYS_ToFFEFFECT.
 */
static void SDL_SYS_HapticFreeFFEFFECT(FFEFFECT *effect, int type)
{
    FFCUSTOMFORCE *custom;

    SDL_free(effect->lpEnvelope);
    effect->lpEnvelope = NULL;
    SDL_free(effect->rgdwAxes);
    effect->rgdwAxes = NULL;
    if (effect->lpvTypeSpecificParams) {
        if (type == SDL_HAPTIC_CUSTOM) { /* Must free the custom data. */
            custom = (FFCUSTOMFORCE *)effect->lpvTypeSpecificParams;
            SDL_free(custom->rglForceData);
            custom->rglForceData = NULL;
        }
        SDL_free(effect->lpvTypeSpecificParams);
        effect->lpvTypeSpecificParams = NULL;
    }
    SDL_free(effect->rglDirection);
    effect->rglDirection = NULL;
}

/*
 * Gets the effect type from the generic SDL haptic effect wrapper.
 */
CFUUIDRef
SDL_SYS_HapticEffectType(Uint16 type)
{
    switch (type) {
    case SDL_HAPTIC_CONSTANT:
        return kFFEffectType_ConstantForce_ID;

    case SDL_HAPTIC_RAMP:
        return kFFEffectType_RampForce_ID;

        /* !!! FIXME: put this back when we have more bits in 2.1 */
        /* case SDL_HAPTIC_SQUARE:
            return kFFEffectType_Square_ID; */

    case SDL_HAPTIC_SINE:
        return kFFEffectType_Sine_ID;

    case SDL_HAPTIC_TRIANGLE:
        return kFFEffectType_Triangle_ID;

    case SDL_HAPTIC_SAWTOOTHUP:
        return kFFEffectType_SawtoothUp_ID;

    case SDL_HAPTIC_SAWTOOTHDOWN:
        return kFFEffectType_SawtoothDown_ID;

    case SDL_HAPTIC_SPRING:
        return kFFEffectType_Spring_ID;

    case SDL_HAPTIC_DAMPER:
        return kFFEffectType_Damper_ID;

    case SDL_HAPTIC_INERTIA:
        return kFFEffectType_Inertia_ID;

    case SDL_HAPTIC_FRICTION:
        return kFFEffectType_Friction_ID;

    case SDL_HAPTIC_CUSTOM:
        return kFFEffectType_CustomForce_ID;

    default:
        SDL_SetError("Haptic: Unknown effect type.");
        return NULL;
    }
}

/*
 * Creates a new haptic effect.
 */
int SDL_SYS_HapticNewEffect(SDL_Haptic *haptic, struct haptic_effect *effect,
                            SDL_HapticEffect *base)
{
    HRESULT ret;
    CFUUIDRef type;

    /* Alloc the effect. */
    effect->hweffect = (struct haptic_hweffect *)
        SDL_malloc(sizeof(struct haptic_hweffect));
    if (!effect->hweffect) {
        SDL_OutOfMemory();
        goto err_hweffect;
    }

    /* Get the type. */
    type = SDL_SYS_HapticEffectType(base->type);
    if (!type) {
        goto err_hweffect;
    }

    /* Get the effect. */
    if (SDL_SYS_ToFFEFFECT(haptic, &effect->hweffect->effect, base) < 0) {
        goto err_effectdone;
    }

    /* Create the actual effect. */
    ret = FFDeviceCreateEffect(haptic->hwdata->device, type,
                               &effect->hweffect->effect,
                               &effect->hweffect->ref);
    if (ret != FF_OK) {
        SDL_SetError("Haptic: Unable to create effect: %s.", FFStrError(ret));
        goto err_effectdone;
    }

    return 0;

err_effectdone:
    SDL_SYS_HapticFreeFFEFFECT(&effect->hweffect->effect, base->type);
err_hweffect:
    SDL_free(effect->hweffect);
    effect->hweffect = NULL;
    return -1;
}

/*
 * Updates an effect.
 */
int SDL_SYS_HapticUpdateEffect(SDL_Haptic *haptic,
                               struct haptic_effect *effect,
                               SDL_HapticEffect *data)
{
    HRESULT ret;
    FFEffectParameterFlag flags;
    FFEFFECT temp;

    /* Get the effect. */
    SDL_memset(&temp, 0, sizeof(FFEFFECT));
    if (SDL_SYS_ToFFEFFECT(haptic, &temp, data) < 0) {
        goto err_update;
    }

    /* Set the flags.  Might be worthwhile to diff temp with loaded effect and
     *  only change those parameters. */
    flags = FFEP_DIRECTION |
            FFEP_DURATION |
            FFEP_ENVELOPE |
            FFEP_STARTDELAY |
            FFEP_TRIGGERBUTTON |
            FFEP_TRIGGERREPEATINTERVAL | FFEP_TYPESPECIFICPARAMS;

    /* Create the actual effect. */
    ret = FFEffectSetParameters(effect->hweffect->ref, &temp, flags);
    if (ret != FF_OK) {
        SDL_SetError("Haptic: Unable to update effect: %s.", FFStrError(ret));
        goto err_update;
    }

    /* Copy it over. */
    SDL_SYS_HapticFreeFFEFFECT(&effect->hweffect->effect, data->type);
    SDL_memcpy(&effect->hweffect->effect, &temp, sizeof(FFEFFECT));

    return 0;

err_update:
    SDL_SYS_HapticFreeFFEFFECT(&temp, data->type);
    return -1;
}

/*
 * Runs an effect.
 */
int SDL_SYS_HapticRunEffect(SDL_Haptic *haptic, struct haptic_effect *effect,
                            Uint32 iterations)
{
    HRESULT ret;
    Uint32 iter;

    /* Check if it's infinite. */
    if (iterations == SDL_HAPTIC_INFINITY) {
        iter = FF_INFINITE;
    } else {
        iter = iterations;
    }

    /* Run the effect. */
    ret = FFEffectStart(effect->hweffect->ref, iter, 0);
    if (ret != FF_OK) {
        return SDL_SetError("Haptic: Unable to run the effect: %s.",
                            FFStrError(ret));
    }

    return 0;
}

/*
 * Stops an effect.
 */
int SDL_SYS_HapticStopEffect(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    HRESULT ret;

    ret = FFEffectStop(effect->hweffect->ref);
    if (ret != FF_OK) {
        return SDL_SetError("Haptic: Unable to stop the effect: %s.",
                            FFStrError(ret));
    }

    return 0;
}

/*
 * Frees the effect.
 */
void SDL_SYS_HapticDestroyEffect(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    HRESULT ret;

    ret = FFDeviceReleaseEffect(haptic->hwdata->device, effect->hweffect->ref);
    if (ret != FF_OK) {
        SDL_SetError("Haptic: Error removing the effect from the device: %s.",
                     FFStrError(ret));
    }
    SDL_SYS_HapticFreeFFEFFECT(&effect->hweffect->effect,
                               effect->effect.type);
    SDL_free(effect->hweffect);
    effect->hweffect = NULL;
}

/*
 * Gets the status of a haptic effect.
 */
int SDL_SYS_HapticGetEffectStatus(SDL_Haptic *haptic,
                                  struct haptic_effect *effect)
{
    HRESULT ret;
    FFEffectStatusFlag status;

    ret = FFEffectGetEffectStatus(effect->hweffect->ref, &status);
    if (ret != FF_OK) {
        SDL_SetError("Haptic: Unable to get effect status: %s.",
                     FFStrError(ret));
        return -1;
    }

    if (status == 0) {
        return SDL_FALSE;
    }
    return SDL_TRUE; /* Assume it's playing or emulated. */
}

/*
 * Sets the gain.
 */
int SDL_SYS_HapticSetGain(SDL_Haptic *haptic, int gain)
{
    HRESULT ret;
    Uint32 val;

    val = gain * 100;           /* Mac OS X uses 0 to 10,000 */
    ret = FFDeviceSetForceFeedbackProperty(haptic->hwdata->device,
                                           FFPROP_FFGAIN, &val);
    if (ret != FF_OK) {
        return SDL_SetError("Haptic: Error setting gain: %s.", FFStrError(ret));
    }

    return 0;
}

/*
 * Sets the autocentering.
 */
int SDL_SYS_HapticSetAutocenter(SDL_Haptic *haptic, int autocenter)
{
    HRESULT ret;
    Uint32 val;

    /* Mac OS X only has 0 (off) and 1 (on) */
    if (autocenter == 0) {
        val = 0;
    } else {
        val = 1;
    }

    ret = FFDeviceSetForceFeedbackProperty(haptic->hwdata->device,
                                           FFPROP_AUTOCENTER, &val);
    if (ret != FF_OK) {
        return SDL_SetError("Haptic: Error setting autocenter: %s.",
                            FFStrError(ret));
    }

    return 0;
}

/*
 * Pauses the device.
 */
int SDL_SYS_HapticPause(SDL_Haptic *haptic)
{
    HRESULT ret;

    ret = FFDeviceSendForceFeedbackCommand(haptic->hwdata->device,
                                           FFSFFC_PAUSE);
    if (ret != FF_OK) {
        return SDL_SetError("Haptic: Error pausing device: %s.", FFStrError(ret));
    }

    return 0;
}

/*
 * Unpauses the device.
 */
int SDL_SYS_HapticUnpause(SDL_Haptic *haptic)
{
    HRESULT ret;

    ret = FFDeviceSendForceFeedbackCommand(haptic->hwdata->device,
                                           FFSFFC_CONTINUE);
    if (ret != FF_OK) {
        return SDL_SetError("Haptic: Error pausing device: %s.", FFStrError(ret));
    }

    return 0;
}

/*
 * Stops all currently playing effects.
 */
int SDL_SYS_HapticStopAll(SDL_Haptic *haptic)
{
    HRESULT ret;

    ret = FFDeviceSendForceFeedbackCommand(haptic->hwdata->device,
                                           FFSFFC_STOPALL);
    if (ret != FF_OK) {
        return SDL_SetError("Haptic: Error stopping device: %s.", FFStrError(ret));
    }

    return 0;
}

#endif /* SDL_HAPTIC_IOKIT */

/* vi: set ts=4 sw=4 expandtab: */
