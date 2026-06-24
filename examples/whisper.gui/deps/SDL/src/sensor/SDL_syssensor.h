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

#ifndef SDL_syssensor_c_h_
#define SDL_syssensor_c_h_

#include "SDL_config.h"

/* This is the system specific header for the SDL sensor API */

#include "SDL_sensor.h"
#include "SDL_sensor_c.h"

/* The SDL sensor structure */
struct _SDL_Sensor
{
    SDL_SensorID instance_id; /* Device instance, monotonically increasing from 0 */
    char *name;               /* Sensor name - system dependent */
    SDL_SensorType type;      /* Type of the sensor */
    int non_portable_type;    /* Platform dependent type of the sensor */

    Uint64 timestamp_us; /* The timestamp of the last sensor update */
    float data[16];      /* The current state of the sensor */

    struct _SDL_SensorDriver *driver;

    struct sensor_hwdata *hwdata; /* Driver dependent information */

    int ref_count; /* Reference count for multiple opens */

    struct _SDL_Sensor *next; /* pointer to next sensor we have allocated */
};

typedef struct _SDL_SensorDriver
{
    /* Function to scan the system for sensors.
     * sensor 0 should be the system default sensor.
     * This function should return 0, or -1 on an unrecoverable fatal error.
     */
    int (*Init)(void);

    /* Function to return the number of sensors available right now */
    int (*GetCount)(void);

    /* Function to check to see if the available sensors have changed */
    void (*Detect)(void);

    /* Function to get the device-dependent name of a sensor */
    const char *(*GetDeviceName)(int device_index);

    /* Function to get the type of a sensor */
    SDL_SensorType (*GetDeviceType)(int device_index);

    /* Function to get the platform dependent type of a sensor */
    int (*GetDeviceNonPortableType)(int device_index);

    /* Function to get the current instance id of the sensor located at device_index */
    SDL_SensorID (*GetDeviceInstanceID)(int device_index);

    /* Function to open a sensor for use.
       The sensor to open is specified by the device index.
       It returns 0, or -1 if there is an error.
     */
    int (*Open)(SDL_Sensor *sensor, int device_index);

    /* Function to update the state of a sensor - called as a device poll.
     * This function shouldn't update the sensor structure directly,
     * but instead should call SDL_PrivateSensorUpdate() to deliver events
     * and update sensor device state.
     */
    void (*Update)(SDL_Sensor *sensor);

    /* Function to close a sensor after use */
    void (*Close)(SDL_Sensor *sensor);

    /* Function to perform any system-specific sensor related cleanup */
    void (*Quit)(void);

} SDL_SensorDriver;

/* The available sensor drivers */
extern SDL_SensorDriver SDL_ANDROID_SensorDriver;
extern SDL_SensorDriver SDL_COREMOTION_SensorDriver;
extern SDL_SensorDriver SDL_WINDOWS_SensorDriver;
extern SDL_SensorDriver SDL_DUMMY_SensorDriver;
extern SDL_SensorDriver SDL_VITA_SensorDriver;
extern SDL_SensorDriver SDL_N3DS_SensorDriver;

#endif /* SDL_syssensor_h_ */

/* vi: set ts=4 sw=4 expandtab: */
