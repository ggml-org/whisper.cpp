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

#include "SDL_config.h"

#ifdef SDL_SENSOR_ANDROID

/* This is the system specific header for the SDL sensor API */
#include <android/sensor.h>

#include "SDL_error.h"
#include "SDL_sensor.h"
#include "SDL_androidsensor.h"
#include "../SDL_syssensor.h"
#include "../SDL_sensor_c.h"

#ifndef LOOPER_ID_USER
#define LOOPER_ID_USER 3
#endif

typedef struct
{
    ASensorRef asensor;
    SDL_SensorID instance_id;
} SDL_AndroidSensor;

static ASensorManager *SDL_sensor_manager;
static ALooper *SDL_sensor_looper;
static SDL_AndroidSensor *SDL_sensors;
static int SDL_sensors_count;

static int SDL_ANDROID_SensorInit(void)
{
    int i, sensors_count;
    ASensorList sensors;

    SDL_sensor_manager = ASensorManager_getInstance();
    if (!SDL_sensor_manager) {
        return SDL_SetError("Couldn't create sensor manager");
    }

    SDL_sensor_looper = ALooper_forThread();
    if (!SDL_sensor_looper) {
        SDL_sensor_looper = ALooper_prepare(ALOOPER_PREPARE_ALLOW_NON_CALLBACKS);
        if (!SDL_sensor_looper) {
            return SDL_SetError("Couldn't create sensor event loop");
        }
    }

    /* FIXME: Is the sensor list dynamic? */
    sensors_count = ASensorManager_getSensorList(SDL_sensor_manager, &sensors);
    if (sensors_count > 0) {
        SDL_sensors = (SDL_AndroidSensor *)SDL_calloc(sensors_count, sizeof(*SDL_sensors));
        if (!SDL_sensors) {
            return SDL_OutOfMemory();
        }

        for (i = 0; i < sensors_count; ++i) {
            SDL_sensors[i].asensor = sensors[i];
            SDL_sensors[i].instance_id = SDL_GetNextSensorInstanceID();
        }
        SDL_sensors_count = sensors_count;
    }
    return 0;
}

static int SDL_ANDROID_SensorGetCount(void)
{
    return SDL_sensors_count;
}

static void SDL_ANDROID_SensorDetect(void)
{
}

static const char *SDL_ANDROID_SensorGetDeviceName(int device_index)
{
    return ASensor_getName(SDL_sensors[device_index].asensor);
}

static SDL_SensorType SDL_ANDROID_SensorGetDeviceType(int device_index)
{
    switch (ASensor_getType(SDL_sensors[device_index].asensor)) {
    case 0x00000001:
        return SDL_SENSOR_ACCEL;
    case 0x00000004:
        return SDL_SENSOR_GYRO;
    default:
        return SDL_SENSOR_UNKNOWN;
    }
}

static int SDL_ANDROID_SensorGetDeviceNonPortableType(int device_index)
{
    return ASensor_getType(SDL_sensors[device_index].asensor);
}

static SDL_SensorID SDL_ANDROID_SensorGetDeviceInstanceID(int device_index)
{
    return SDL_sensors[device_index].instance_id;
}

static int SDL_ANDROID_SensorOpen(SDL_Sensor *sensor, int device_index)
{
    struct sensor_hwdata *hwdata;
    int delay_us, min_delay_us;

    hwdata = (struct sensor_hwdata *)SDL_calloc(1, sizeof(*hwdata));
    if (!hwdata) {
        return SDL_OutOfMemory();
    }

    hwdata->asensor = SDL_sensors[device_index].asensor;
    hwdata->eventqueue = ASensorManager_createEventQueue(SDL_sensor_manager, SDL_sensor_looper, LOOPER_ID_USER, NULL, NULL);
    if (!hwdata->eventqueue) {
        SDL_free(hwdata);
        return SDL_SetError("Couldn't create sensor event queue");
    }

    if (ASensorEventQueue_enableSensor(hwdata->eventqueue, hwdata->asensor) < 0) {
        ASensorManager_destroyEventQueue(SDL_sensor_manager, hwdata->eventqueue);
        SDL_free(hwdata);
        return SDL_SetError("Couldn't enable sensor");
    }

    /* Use 60 Hz update rate if possible */
    /* FIXME: Maybe add a hint for this? */
    delay_us = 1000000 / 60;
    min_delay_us = ASensor_getMinDelay(hwdata->asensor);
    if (delay_us < min_delay_us) {
        delay_us = min_delay_us;
    }
    ASensorEventQueue_setEventRate(hwdata->eventqueue, hwdata->asensor, delay_us);

    sensor->hwdata = hwdata;
    return 0;
}

static void SDL_ANDROID_SensorUpdate(SDL_Sensor *sensor)
{
    int events;
    ASensorEvent event;
    struct android_poll_source *source;

    if (ALooper_pollOnce(0, NULL, &events, (void **)&source) == LOOPER_ID_USER) {
        SDL_zero(event);
        while (ASensorEventQueue_getEvents(sensor->hwdata->eventqueue, &event, 1) > 0) {
            SDL_PrivateSensorUpdate(sensor, 0, event.data, SDL_arraysize(event.data));
        }
    }
}

static void SDL_ANDROID_SensorClose(SDL_Sensor *sensor)
{
    if (sensor->hwdata) {
        ASensorEventQueue_disableSensor(sensor->hwdata->eventqueue, sensor->hwdata->asensor);
        ASensorManager_destroyEventQueue(SDL_sensor_manager, sensor->hwdata->eventqueue);
        SDL_free(sensor->hwdata);
        sensor->hwdata = NULL;
    }
}

static void SDL_ANDROID_SensorQuit(void)
{
    if (SDL_sensors) {
        SDL_free(SDL_sensors);
        SDL_sensors = NULL;
        SDL_sensors_count = 0;
    }
}

SDL_SensorDriver SDL_ANDROID_SensorDriver = {
    SDL_ANDROID_SensorInit,
    SDL_ANDROID_SensorGetCount,
    SDL_ANDROID_SensorDetect,
    SDL_ANDROID_SensorGetDeviceName,
    SDL_ANDROID_SensorGetDeviceType,
    SDL_ANDROID_SensorGetDeviceNonPortableType,
    SDL_ANDROID_SensorGetDeviceInstanceID,
    SDL_ANDROID_SensorOpen,
    SDL_ANDROID_SensorUpdate,
    SDL_ANDROID_SensorClose,
    SDL_ANDROID_SensorQuit,
};

#endif /* SDL_SENSOR_ANDROID */

/* vi: set ts=4 sw=4 expandtab: */
