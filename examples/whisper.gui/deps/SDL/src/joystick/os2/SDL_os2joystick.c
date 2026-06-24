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

#ifdef SDL_JOYSTICK_OS2

/* OS/2 Joystick driver, contributed by Daniel Caetano */

#define INCL_DOSDEVICES
#define INCL_DOSDEVIOCTL
#define INCL_DOSMEMMGR
#include <os2.h>

/*****************************************************************
 * OS/2 Joystick driver defs. Based on docs at edm2.com and in old
 * drivers available at hobbes.nmsu.edu and www.os2site.com
 *****************************************************************/

#define GAME_GET_VERSION 0x01
#define GAME_GET_PARMS   0x02
#define GAME_GET_CALIB   0x04
#define GAME_GET_STATUS  0x10

#define IOCTL_CAT_USER   0x80
#define GAME_PORT_GET    0x20
#define GAME_PORT_RESET  0x60

#pragma pack(push,1)
typedef struct {
  USHORT  uJs_AxCnt, uJs_AyCnt; /* A joystick X/Y pos */
  USHORT  uJs_BxCnt, uJs_ByCnt; /* B joystick X/Y pos */
  USHORT  usJs_ButtonA1Cnt, usJs_ButtonA2Cnt;/* A1/A2 press cnts */
  USHORT  usJs_ButtonB1Cnt, usJs_ButtonB2Cnt;/* B1/B2 press cnts */
  UCHAR   ucJs_JoyStickMask;  /* mask of connected joystick pots */
  UCHAR   ucJs_ButtonStatus;  /* bits of switches down */
  ULONG   ulJs_Ticks;         /* total clock ticks (60 Hz) */
} GAME_PORT_STRUCT;
#pragma pack(pop)

typedef struct {
  USHORT  useA, useB;
  USHORT  mode;
  USHORT  format;
  USHORT  sampDiv;
  USHORT  scale;
  USHORT  res1, res2;
} GAME_PARM_STRUCT;

typedef struct {
  SHORT   x, y;
} GAME_2DPOS_STRUCT;

typedef struct {
  SHORT  lower, centre, upper;
} GAME_3POS_STRUCT;

typedef struct {
  GAME_3POS_STRUCT  Ax, Ay, Bx, By;
} GAME_CALIB_STRUCT;

typedef struct {
  GAME_2DPOS_STRUCT A, B;
  USHORT  butMask;
} GAME_DATA_STRUCT;

typedef struct {
  GAME_DATA_STRUCT  curdata;
  USHORT  b1cnt, b2cnt, b3cnt, b4cnt;
} GAME_STATUS_STRUCT;

/*****************************************************************/

#include "SDL_joystick.h"
#include "SDL_events.h"
#include "../SDL_sysjoystick.h"
#include "../SDL_joystick_c.h"

static HFILE hJoyPort = NULLHANDLE; /* Joystick GAME$ Port Address */
#define MAX_JOYSTICKS   2       /* Maximum of two joysticks */
#define MAX_AXES        4       /* each joystick can have up to 4 axes */
#define MAX_BUTTONS     8       /* 8 buttons */
#define MAX_HATS        0       /* 0 hats - OS/2 doesn't support it */
#define MAX_BALLS       0       /* and 0 balls - OS/2 doesn't support it */
#define MAX_JOYNAME     128     /* Joystick name may have 128 characters */
/* Calc Button Flag for buttons A to D */
#define JOY_BUTTON_FLAG(n) (1<<n)

/* Joystick data... hold information about detected devices */
typedef struct SYS_JoyData_s
{
    Sint8 id;                           /* Device ID          */
    char  szDeviceName[MAX_JOYNAME];    /* Device Name        */
    char  axes;                         /* Number of axes     */
    char  buttons;                      /* Number of buttons  */
    char  hats;                         /* Number of buttons  */
    char  balls;                        /* Number of buttons  */
    int   axes_min[MAX_AXES];           /* minimum callibration value for axes  */
    int   axes_med[MAX_AXES];           /* medium callibration value for axes   */
    int   axes_max[MAX_AXES];           /* maximum callibration value for axes  */
    int   buttoncalc[4];                /* Used for buttons 5, 6, 7 and 8.      */
} SYS_JoyData_t, *SYS_JoyData_p;

static SYS_JoyData_t SYS_JoyData[MAX_JOYSTICKS];

/* Structure used to convert data from OS/2 driver format to SDL format */
struct _transaxes
{
    int   offset; /* Center Offset */
    float scale1; /* Center to left/up Scale */
    float scale2; /* Center to right/down Scale */
};

struct joystick_hwdata
{
    Sint8 id;
    struct _transaxes transaxes[MAX_AXES];
};

/* Structure used to get values from Joystick Environment Variable */
struct _joycfg
{
    char  name[MAX_JOYNAME];
    unsigned int  axes;
    unsigned int  buttons;
    unsigned int  hats;
    unsigned int  balls;
};

/* OS/2 Implementation Function Prototypes */
static int joyPortOpen(HFILE * hGame);
static void joyPortClose(HFILE * hGame);
static int joyGetData(const char *joyenv, char *name, char stopchar, size_t maxchars);
static int joyGetEnv(struct _joycfg * joydata);


static int numjoysticks = 0;

/************************************************************************/
/* Function to scan the system for joysticks.                           */
/* Joystick 0 should be the system default joystick.                    */
/* It should return 0, or -1 on an unrecoverable fatal error.           */
/************************************************************************/
static int OS2_JoystickInit(void)
{
    APIRET rc;  /* Generic OS/2 return code */
    GAME_PORT_STRUCT  stJoyStatus;  /* Joystick Status Structure */
    GAME_PARM_STRUCT  stGameParms;  /* Joystick Parameter Structure */
    GAME_CALIB_STRUCT stGameCalib;  /* Calibration Struct */
    ULONG ulDataLen;  /* Size of data */
    ULONG ulLastTick; /* Tick Counter for timing operations */
    Uint8 maxdevs;    /* Maximum number of devices */
    Uint8 numdevs;    /* Number of present devices */
    Uint8 maxbut;     /* Maximum number of buttons... */
    Uint8 i;          /* Temporary Count Vars */
    Uint8 ucNewJoystickMask;  /* Mask for Joystick Detection */
    struct _joycfg joycfg;    /* Joy Configuration from envvar */

    /* Open GAME$ port */
    if (joyPortOpen(&hJoyPort) < 0) {
        return 0; /* Cannot open... report no joystick */
    }
    /* Get Max Number of Devices */
    ulDataLen = sizeof(stGameParms);
    rc = DosDevIOCtl(hJoyPort, IOCTL_CAT_USER, GAME_GET_PARMS,
                      NULL, 0, NULL, &stGameParms, ulDataLen, &ulDataLen); /* Ask device info */
    if (rc != 0) {
        joyPortClose(&hJoyPort);
        return SDL_SetError("Could not read joystick port.");
    }
    maxdevs = 0;
    if (stGameParms.useA != 0) {
        maxdevs++;
    }
    if (stGameParms.useB != 0) {
        maxdevs++;
    }
    if (maxdevs > MAX_JOYSTICKS) {
        maxdevs = MAX_JOYSTICKS;
    }

    /* Defines min/max axes values (callibration) */
    ulDataLen = sizeof(stGameCalib);
    rc = DosDevIOCtl(hJoyPort, IOCTL_CAT_USER, GAME_GET_CALIB,
                     NULL, 0, NULL, &stGameCalib, ulDataLen, &ulDataLen);
    if (rc != 0) {
        joyPortClose(&hJoyPort);
        return SDL_SetError("Could not read callibration data.");
    }

    /* Determine how many joysticks are active */
    numdevs = 0;  /* Points no device */
    ucNewJoystickMask = 0x0F; /* read all 4 joystick axis */
    ulDataLen = sizeof(ucNewJoystickMask);
    rc = DosDevIOCtl(hJoyPort, IOCTL_CAT_USER, GAME_PORT_RESET,
                     &ucNewJoystickMask, ulDataLen, &ulDataLen, NULL, 0, NULL);
    if (rc == 0) {
        ulDataLen = sizeof(stJoyStatus);
        rc = DosDevIOCtl(hJoyPort, IOCTL_CAT_USER, GAME_PORT_GET,
                         NULL, 0, NULL, &stJoyStatus, ulDataLen, &ulDataLen);
        if (rc != 0) {
            joyPortClose(&hJoyPort);
            return SDL_SetError("Could not call joystick port.");
        }
        ulLastTick = stJoyStatus.ulJs_Ticks;
        while (stJoyStatus.ulJs_Ticks == ulLastTick) {
               rc = DosDevIOCtl(hJoyPort, IOCTL_CAT_USER, GAME_PORT_GET,
                                NULL, 0, NULL, &stJoyStatus, ulDataLen, &ulDataLen);
        }
        if ((stJoyStatus.ucJs_JoyStickMask & 0x03) > 0) {
            numdevs++;
        }
        if (((stJoyStatus.ucJs_JoyStickMask >> 2) & 0x03) > 0) {
            numdevs++;
        }
    }

    if (numdevs > maxdevs) {
        numdevs = maxdevs;
    }

    /* If *any* joystick was detected... Let's configure SDL for them */
    if (numdevs > 0) {
        /* Verify if it is a "user defined" joystick */
        if (joyGetEnv(&joycfg)) {
            GAME_3POS_STRUCT * axis[4];
            axis[0] = &stGameCalib.Ax;
            axis[1] = &stGameCalib.Ay;
            axis[2] = &stGameCalib.Bx;
            axis[3] = &stGameCalib.By;

            /* Say it has one device only (user defined is always one device only) */
            numdevs = 1;

            /* Define Device 0 as... */
            SYS_JoyData[0].id = 0;

            /* Define Number of Axes... up to 4 */
            if (joycfg.axes>MAX_AXES) {
                joycfg.axes = MAX_AXES;
            }
            SYS_JoyData[0].axes = joycfg.axes;

            /* Define number of buttons... 8 if 2 axes, 6 if 3 axes and 4 if 4 axes */
            maxbut = MAX_BUTTONS;
            if (joycfg.axes > 2) {
                maxbut -= ((joycfg.axes - 2) << 1);     /* MAX_BUTTONS - 2*(axes-2) */
            }
            if (joycfg.buttons > maxbut) {
                joycfg.buttons = maxbut;
            }
            SYS_JoyData[0].buttons = joycfg.buttons;

            /* Define number of hats */
            if (joycfg.hats > MAX_HATS) {
                joycfg.hats = MAX_HATS;
            }
            SYS_JoyData[0].hats = joycfg.hats;

            /* Define number of balls */
            if (joycfg.balls > MAX_BALLS) {
                joycfg.balls = MAX_BALLS;
            }
            SYS_JoyData[0].balls = joycfg.balls;

            /* Initialize Axes Callibration Values */
            for (i = 0; i < joycfg.axes; i++) {
                 SYS_JoyData[0].axes_min[i] = axis[i]->lower;
                 SYS_JoyData[0].axes_med[i] = axis[i]->centre;
                 SYS_JoyData[0].axes_max[i] = axis[i]->upper;
            }
            /* Initialize Buttons 5 to 8 structures */
            if (joycfg.buttons >=5 ) {
                SYS_JoyData[0].buttoncalc[0] = ((axis[2]->lower + axis[3]->centre) >> 1);
            }
            if (joycfg.buttons >=6 ) {
                SYS_JoyData[0].buttoncalc[1] = ((axis[3]->lower + axis[3]->centre) >> 1);
            }
            if (joycfg.buttons >=7 ) {
                SYS_JoyData[0].buttoncalc[2] = ((axis[2]->upper + axis[3]->centre) >> 1);
            }
            if (joycfg.buttons >=8 ) {
                SYS_JoyData[0].buttoncalc[3] = ((axis[3]->upper + axis[3]->centre) >> 1);
            }
            /* Intialize Joystick Name */
            SDL_strlcpy (SYS_JoyData[0].szDeviceName,joycfg.name, SDL_arraysize(SYS_JoyData[0].szDeviceName));
        }
        /* Default Init ... autoconfig */
        else {
            /* if two devices were detected... configure as Joy1 4 axis and Joy2 2 axis */
            if (numdevs == 2) {
                /* Define Device 0 as 4 axes, 4 buttons */
                SYS_JoyData[0].id = 0;
                SYS_JoyData[0].axes = 4;
                SYS_JoyData[0].buttons = 4;
                SYS_JoyData[0].hats = 0;
                SYS_JoyData[0].balls = 0;
                SYS_JoyData[0].axes_min[0] = stGameCalib.Ax.lower;
                SYS_JoyData[0].axes_med[0] = stGameCalib.Ax.centre;
                SYS_JoyData[0].axes_max[0] = stGameCalib.Ax.upper;
                SYS_JoyData[0].axes_min[1] = stGameCalib.Ay.lower;
                SYS_JoyData[0].axes_med[1] = stGameCalib.Ay.centre;
                SYS_JoyData[0].axes_max[1] = stGameCalib.Ay.upper;
                SYS_JoyData[0].axes_min[2] = stGameCalib.Bx.lower;
                SYS_JoyData[0].axes_med[2] = stGameCalib.Bx.centre;
                SYS_JoyData[0].axes_max[2] = stGameCalib.Bx.upper;
                SYS_JoyData[0].axes_min[3] = stGameCalib.By.lower;
                SYS_JoyData[0].axes_med[3] = stGameCalib.By.centre;
                SYS_JoyData[0].axes_max[3] = stGameCalib.By.upper;
                /* Define Device 1 as 2 axes, 2 buttons */
                SYS_JoyData[1].id = 1;
                SYS_JoyData[1].axes = 2;
                SYS_JoyData[1].buttons = 2;
                SYS_JoyData[1].hats = 0;
                SYS_JoyData[1].balls = 0;
                SYS_JoyData[1].axes_min[0] = stGameCalib.Bx.lower;
                SYS_JoyData[1].axes_med[0] = stGameCalib.Bx.centre;
                SYS_JoyData[1].axes_max[0] = stGameCalib.Bx.upper;
                SYS_JoyData[1].axes_min[1] = stGameCalib.By.lower;
                SYS_JoyData[1].axes_med[1] = stGameCalib.By.centre;
                SYS_JoyData[1].axes_max[1] = stGameCalib.By.upper;
            }
            /* One joystick only? */
            else {
                /* If it is joystick A... */
                if ((stJoyStatus.ucJs_JoyStickMask & 0x03) > 0) {
                    /* Define Device 0 as 2 axes, 4 buttons */
                    SYS_JoyData[0].id = 0;
                    SYS_JoyData[0].axes = 2;
                    SYS_JoyData[0].buttons = 4;
                    SYS_JoyData[0].hats = 0;
                    SYS_JoyData[0].balls = 0;
                    SYS_JoyData[0].axes_min[0] = stGameCalib.Ax.lower;
                    SYS_JoyData[0].axes_med[0] = stGameCalib.Ax.centre;
                    SYS_JoyData[0].axes_max[0] = stGameCalib.Ax.upper;
                    SYS_JoyData[0].axes_min[1] = stGameCalib.Ay.lower;
                    SYS_JoyData[0].axes_med[1] = stGameCalib.Ay.centre;
                    SYS_JoyData[0].axes_max[1] = stGameCalib.Ay.upper;
                }
                /* If not, it is joystick B */
                else {
                    /* Define Device 1 as 2 axes, 2 buttons */
                    SYS_JoyData[0].id = 1;
                    SYS_JoyData[0].axes = 2;
                    SYS_JoyData[0].buttons = 2;
                    SYS_JoyData[0].hats = 0;
                    SYS_JoyData[0].balls = 0;
                    SYS_JoyData[0].axes_min[0] = stGameCalib.Bx.lower;
                    SYS_JoyData[0].axes_med[0] = stGameCalib.Bx.centre;
                    SYS_JoyData[0].axes_max[0] = stGameCalib.Bx.upper;
                    SYS_JoyData[0].axes_min[1] = stGameCalib.By.lower;
                    SYS_JoyData[0].axes_med[1] = stGameCalib.By.centre;
                    SYS_JoyData[0].axes_max[1] = stGameCalib.By.upper;
                }
            }

            /* Hack to define Joystick Port Names */
            if (numdevs > maxdevs) {
                numdevs = maxdevs;
            }

            for (i = 0; i < numdevs; i++) {
                SDL_snprintf(SYS_JoyData[i].szDeviceName, SDL_arraysize(SYS_JoyData[i].szDeviceName),
                            "Default Joystick %c", 'A'+SYS_JoyData[i].id);
            }
        }
    }

    /* Return the number of devices found */
    numjoysticks = numdevs;
    return numdevs;
}

static int OS2_NumJoysticks(void)
{
    return numjoysticks;
}

static void OS2_JoystickDetect(void)
{
}

static const char *OS2_JoystickGetDeviceName(int device_index)
{
    /* No need to verify if device exists, already done in upper layer */
    return SYS_JoyData[device_index].szDeviceName;
}

static const char *OS2_JoystickGetDevicePath(int device_index)
{
    return NULL;
}

static int OS2_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return -1;
}

static int OS2_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

static void OS2_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

static SDL_JoystickGUID OS2_JoystickGetDeviceGUID(int device_index)
{
    /* the GUID is just the name for now */
    const char *name = OS2_JoystickGetDeviceName(device_index);
    return SDL_CreateJoystickGUIDForName(name);
}

static SDL_JoystickID OS2_JoystickGetDeviceInstanceID(int device_index)
{
    return device_index;
}

/******************************************************************************/
/* Function to open a joystick for use.                                       */
/* The joystick to open is specified by the device index.                     */
/* This should fill the nbuttons and naxes fields of the joystick structure.  */
/* It returns 0, or -1 if there is an error.                                  */
/******************************************************************************/
static int OS2_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    int index;  /* Index shortcut for index in joystick structure */
    int i;      /* Generic Counter */

    /* allocate memory for system specific hardware data */
    joystick->hwdata = (struct joystick_hwdata *) SDL_calloc(1, sizeof(*joystick->hwdata));
    if (!joystick->hwdata) {
        return SDL_OutOfMemory();
    }

    /* ShortCut Pointer */
    index = device_index;
    joystick->instance_id = device_index;

    /* Define offsets and scales for all axes */
    joystick->hwdata->id = SYS_JoyData[index].id;
    for (i = 0; i < MAX_AXES; ++i) {
        if ((i < 2) || i < SYS_JoyData[index].axes) {
            joystick->hwdata->transaxes[i].offset = ((SDL_JOYSTICK_AXIS_MAX + SDL_JOYSTICK_AXIS_MIN)>>1) - SYS_JoyData[index].axes_med[i];
            joystick->hwdata->transaxes[i].scale1 = (float)SDL_abs((SDL_JOYSTICK_AXIS_MIN/SYS_JoyData[index].axes_min[i]));
            joystick->hwdata->transaxes[i].scale2 = (float)SDL_abs((SDL_JOYSTICK_AXIS_MAX/SYS_JoyData[index].axes_max[i]));
        } else {
            joystick->hwdata->transaxes[i].offset = 0;
            joystick->hwdata->transaxes[i].scale1 = 1.0f; /* Just in case */
            joystick->hwdata->transaxes[i].scale2 = 1.0f; /* Just in case */
        }
    }

    /* fill nbuttons, naxes, and nhats fields */
    joystick->nbuttons = SYS_JoyData[index].buttons;
    joystick->naxes = SYS_JoyData[index].axes;

    /* joystick->nhats = SYS_JoyData[index].hats; */
    joystick->nhats = 0; /* No support for hats at this time */

    /* joystick->nballs = SYS_JoyData[index].balls; */
    joystick->nballs = 0; /* No support for balls at this time */

    return 0;
}

static int OS2_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    return SDL_Unsupported();
}

static int OS2_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 OS2_JoystickGetCapabilities(SDL_Joystick *joystick)
{
    return 0;
}

static int OS2_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static int OS2_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static int OS2_JoystickSetSensorsEnabled(SDL_Joystick *joystick, SDL_bool enabled)
{
    return SDL_Unsupported();
}

/***************************************************************************/
/* Function to update the state of a joystick - called as a device poll.   */
/* This function shouldn't update the joystick structure directly,         */
/* but instead should call SDL_PrivateJoystick*() to deliver events        */
/* and update joystick device state.                                       */
/***************************************************************************/
static void OS2_JoystickUpdate(SDL_Joystick *joystick)
{
    APIRET rc;  /* Generic OS/2 return code */
    int index;  /* index shortcurt to joystick index */
    int i;      /* Generic counter */
    int normbut;    /* Number of buttons reported by joystick */
    int corr;       /* Correction for button names */
    Sint16 value;   /* Values used to update axis values */
    struct _transaxes *transaxes; /* Shortcut for Correction structure */
    Uint32 pos[MAX_AXES];         /* Vector to inform the Axis status  */
    ULONG ulDataLen;              /* Size of data */
    GAME_STATUS_STRUCT stGameStatus;    /* Joystick Status Structure   */

    ulDataLen = sizeof(stGameStatus);
    rc = DosDevIOCtl(hJoyPort, IOCTL_CAT_USER, GAME_GET_STATUS,
                      NULL, 0, NULL, &stGameStatus, ulDataLen, &ulDataLen);
    if (rc != 0) {
        SDL_SetError("Could not read joystick status.");
        return; /* Could not read data */
    }

    /* Shortcut pointer */
    index = joystick->instance_id;

    /* joystick motion events */

    if (SYS_JoyData[index].id == 0) {
        pos[0] = stGameStatus.curdata.A.x;
        pos[1] = stGameStatus.curdata.A.y;
        if (SYS_JoyData[index].axes >= 3) {
            pos[2] = stGameStatus.curdata.B.x;
        } else {
            pos[2] = 0;
        }
        if (SYS_JoyData[index].axes >= 4) {
            pos[3] = stGameStatus.curdata.B.y;
        } else {
            pos[3] = 0;
        }
        /* OS/2 basic drivers do not support more than 4 axes joysticks */
    }
    else if (SYS_JoyData[index].id == 1) {
        pos[0] = stGameStatus.curdata.B.x;
        pos[1] = stGameStatus.curdata.B.y;
        pos[2] = 0;
        pos[3] = 0;
    }

    /* Corrects the movements using the callibration */
    transaxes = joystick->hwdata->transaxes;
    for (i = 0; i < joystick->naxes; i++) {
        value = pos[i] + transaxes[i].offset;
        if (value < 0) {
            value *= transaxes[i].scale1;
            if (value > 0) {
                value = SDL_JOYSTICK_AXIS_MIN;
            }
        } else {
            value *= transaxes[i].scale2;
            if (value < 0) {
                value = SDL_JOYSTICK_AXIS_MAX;
            }
        }
        SDL_PrivateJoystickAxis(joystick, (Uint8)i, (Sint16)value);
    }

    /* joystick button A to D events */
    if (SYS_JoyData[index].id == 1) {
        corr = 2;
    } else {
        corr = 0;
    }
    normbut = 4;    /* Number of normal buttons */
    if (joystick->nbuttons < normbut) {
        normbut = joystick->nbuttons;
    }
    for (i = corr; (i-corr) < normbut; ++i) {
        /*
        Button A: 1110 0000
        Button B: 1101 0000
        Button C: 1011 0000
        Button D: 0111 0000
        */
        if ((~stGameStatus.curdata.butMask)>>4 & JOY_BUTTON_FLAG(i)) {
            SDL_PrivateJoystickButton(joystick, (Uint8)(i-corr), SDL_PRESSED);
        } else {
            SDL_PrivateJoystickButton(joystick, (Uint8)(i-corr), SDL_RELEASED);
        }
    }

    /* Joystick button E to H buttons */
    /*
    Button E: Axis 2 X Left
    Button F: Axis 2 Y Up
    Button G: Axis 2 X Right
    Button H: Axis 2 Y Down
    */
    if (joystick->nbuttons >= 5) {
        if (stGameStatus.curdata.B.x < SYS_JoyData[index].buttoncalc[0]) {
            SDL_PrivateJoystickButton(joystick, (Uint8)4, SDL_PRESSED);
        } else {
            SDL_PrivateJoystickButton(joystick, (Uint8)4, SDL_RELEASED);
        }
    }
    if (joystick->nbuttons >= 6) {
        if (stGameStatus.curdata.B.y < SYS_JoyData[index].buttoncalc[1]) {
            SDL_PrivateJoystickButton(joystick, (Uint8)5, SDL_PRESSED);
        } else {
            SDL_PrivateJoystickButton(joystick, (Uint8)5, SDL_RELEASED);
        }
    }
    if (joystick->nbuttons >= 7) {
        if (stGameStatus.curdata.B.x > SYS_JoyData[index].buttoncalc[2]) {
            SDL_PrivateJoystickButton(joystick, (Uint8)6, SDL_PRESSED);
        } else {
            SDL_PrivateJoystickButton(joystick, (Uint8)6, SDL_RELEASED);
        }
    }
    if (joystick->nbuttons >= 8) {
        if (stGameStatus.curdata.B.y > SYS_JoyData[index].buttoncalc[3]) {
            SDL_PrivateJoystickButton(joystick, (Uint8)7, SDL_PRESSED);
        } else {
           SDL_PrivateJoystickButton(joystick, (Uint8)7, SDL_RELEASED);
        }
    }

    /* joystick hat events */
    /* Not Supported under OS/2 */
    /* joystick ball events */
    /* Not Supported under OS/2 */
}

/******************************************/
/* Function to close a joystick after use */
/******************************************/
static void OS2_JoystickClose(SDL_Joystick *joystick)
{
    /* free system specific hardware data */
    SDL_free(joystick->hwdata);
}

/********************************************************************/
/* Function to perform any system-specific joystick related cleanup */
/********************************************************************/
static void OS2_JoystickQuit(void)
{
    joyPortClose(&hJoyPort);
}

static SDL_bool OS2_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return SDL_FALSE;
}


/************************/
/* OS/2 Implementations */
/************************/

/*****************************************/
/* Open Joystick Port, if not opened yet */
/*****************************************/
static int joyPortOpen(HFILE * hGame)
{
    APIRET  rc; /* Generic Return Code */
    ULONG ulAction;   /* ? */
    ULONG ulVersion;  /* Version of joystick driver */
    ULONG ulDataLen;  /* Size of version data */

    /* Verifies if joyport is not already open... */
    if (*hGame != NULLHANDLE) {
        return 0;
    }

    /* Open GAME$ for read */
    rc = DosOpen("GAME$   ", hGame, &ulAction, 0, FILE_READONLY,
                 FILE_OPEN, OPEN_ACCESS_READONLY | OPEN_SHARE_DENYNONE, NULL);
    if (rc != 0) {
        return SDL_SetError("Could not open Joystick Port.");
    }

    /* Get Joystick Driver Version... must be 2.0 or higher */
    ulVersion = 0;
    ulDataLen = sizeof(ulVersion);
    rc = DosDevIOCtl(*hGame, IOCTL_CAT_USER, GAME_GET_VERSION,
                     NULL, 0, NULL, &ulVersion, ulDataLen, &ulDataLen);
    if (rc != 0) {
        joyPortClose(hGame);
        return SDL_SetError("Could not get Joystick Driver version.");
    }
    if (ulVersion < 0x20) {
        joyPortClose(hGame);
        return SDL_SetError("Driver too old. At least IBM driver version 2.0 required.");
    }

    return 0;
}

/****************************/
/* Close JoyPort, if opened */
/****************************/
static void joyPortClose(HFILE * hGame)
{
    if (*hGame != NULLHANDLE) {
        DosClose(*hGame);
    }
    *hGame = NULLHANDLE;
}

/***************************/
/* Get SDL Joystick EnvVar */
/***************************/
static int joyGetEnv(struct _joycfg * joydata)
{
    const char *joyenv; /* Pointer to tested character */
    char tempnumber[5]; /* Temporary place to put numeric texts */

    joyenv = SDL_getenv("SDL_OS2_JOYSTICK");
    if (!joyenv) {
        return 0;
    }

    /* Joystick Environment is defined! */
    while (*joyenv == ' ' && *joyenv != 0) {
        joyenv++; /* jump spaces... */
    }

    /* If the string name starts with '... get if fully */
    if (*joyenv == '\'') {
        joyenv++;
        joyenv += joyGetData(joyenv,joydata->name,'\'',sizeof(joydata->name));
    }
    /* If not, get it until the next space */
    else if (*joyenv == '\"') {
        joyenv++;
        joyenv += joyGetData(joyenv,joydata->name,'\"',sizeof(joydata->name));
    }
    else {
        joyenv += joyGetData(joyenv,joydata->name, ' ',sizeof(joydata->name));
    }

    /* Now get the number of axes */
    while (*joyenv == ' ' && *joyenv != 0) {
        joyenv++; /* jump spaces... */
    }
    joyenv += joyGetData(joyenv,tempnumber,' ',sizeof(tempnumber));
    joydata->axes = SDL_atoi(tempnumber);

    /* Now get the number of buttons */
    while (*joyenv == ' ' && *joyenv != 0) {
        joyenv++; /* jump spaces... */
    }
    joyenv += joyGetData(joyenv,tempnumber,' ',sizeof(tempnumber));
    joydata->buttons = SDL_atoi(tempnumber);

    /* Now get the number of hats */
    while (*joyenv == ' ' && *joyenv != 0) {
        joyenv++; /* jump spaces... */
    }
    joyenv += joyGetData(joyenv,tempnumber,' ',sizeof(tempnumber));
    joydata->hats = SDL_atoi(tempnumber);

    /* Now get the number of balls */
    while (*joyenv==' ' && *joyenv != 0) {
        joyenv++; /* jump spaces... */
    }
    joyenv += joyGetData(joyenv,tempnumber,' ',sizeof(tempnumber));
    joydata->balls = SDL_atoi(tempnumber);

    return 1;
}

/************************************************************************/
/* Get a text from in the string starting in joyenv until it finds      */
/* the stopchar or maxchars is reached. The result is placed in name.   */
/************************************************************************/
static int joyGetData(const char *joyenv, char *name, char stopchar, size_t maxchars)
{
    char *nameptr;  /* Pointer to the selected character */
    int chcnt = 0;  /* Count how many characters where copied */

    nameptr = name;
    while (*joyenv!=stopchar && *joyenv != 0) {
        if (nameptr < (name + (maxchars-1))) {
            *nameptr = *joyenv; /* Only copy if smaller than maximum */
            nameptr++;
        }
        chcnt++;
        joyenv++;
    }
    if (*joyenv == stopchar) {
        joyenv++; /* Jump stopchar */
        chcnt++;
    }
    *nameptr = 0; /* Mark last byte */
    return chcnt;
}

SDL_JoystickDriver SDL_OS2_JoystickDriver = {
    OS2_JoystickInit,
    OS2_NumJoysticks,
    OS2_JoystickDetect,
    OS2_JoystickGetDeviceName,
    OS2_JoystickGetDevicePath,
    OS2_JoystickGetDeviceSteamVirtualGamepadSlot,
    OS2_JoystickGetDevicePlayerIndex,
    OS2_JoystickSetDevicePlayerIndex,
    OS2_JoystickGetDeviceGUID,
    OS2_JoystickGetDeviceInstanceID,
    OS2_JoystickOpen,
    OS2_JoystickRumble,
    OS2_JoystickRumbleTriggers,
    OS2_JoystickGetCapabilities,
    OS2_JoystickSetLED,
    OS2_JoystickSendEffect,
    OS2_JoystickSetSensorsEnabled,
    OS2_JoystickUpdate,
    OS2_JoystickClose,
    OS2_JoystickQuit,
    OS2_JoystickGetGamepadMapping
};

#endif /* SDL_JOYSTICK_OS2 */
