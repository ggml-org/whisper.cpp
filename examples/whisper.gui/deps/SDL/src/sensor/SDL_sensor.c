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
#include "../SDL_internal.h"

/* This is the sensor API for Simple DirectMedia Layer */

#include "SDL.h"
#include "SDL_atomic.h"
#include "SDL_events.h"
#include "SDL_syssensor.h"

#ifndef SDL_EVENTS_DISABLED
#include "../events/SDL_events_c.h"
#endif

static SDL_SensorDriver *SDL_sensor_drivers[] = {
#ifdef SDL_SENSOR_ANDROID
    &SDL_ANDROID_SensorDriver,
#endif
#ifdef SDL_SENSOR_COREMOTION
    &SDL_COREMOTION_SensorDriver,
#endif
#ifdef SDL_SENSOR_WINDOWS
    &SDL_WINDOWS_SensorDriver,
#endif
#ifdef SDL_SENSOR_VITA
    &SDL_VITA_SensorDriver,
#endif
#ifdef SDL_SENSOR_N3DS
    &SDL_N3DS_SensorDriver,
#endif
#if defined(SDL_SENSOR_DUMMY) || defined(SDL_SENSOR_DISABLED)
    &SDL_DUMMY_SensorDriver
#endif
};
static SDL_mutex *SDL_sensor_lock = NULL; /* This needs to support recursive locks */
static SDL_Sensor *SDL_sensors SDL_GUARDED_BY(SDL_sensor_lock) = NULL;
static SDL_atomic_t SDL_next_sensor_instance_id SDL_GUARDED_BY(SDL_sensor_lock);
static SDL_bool SDL_updating_sensor SDL_GUARDED_BY(SDL_sensor_lock) = SDL_FALSE;

void SDL_LockSensors(void) SDL_ACQUIRE(SDL_sensor_lock)
{
    SDL_LockMutex(SDL_sensor_lock);
}

void SDL_UnlockSensors(void) SDL_RELEASE(SDL_sensor_lock)
{
    SDL_UnlockMutex(SDL_sensor_lock);
}

int SDL_SensorInit(void)
{
    int i, status;

    /* Create the sensor list lock */
    if (SDL_sensor_lock == NULL) {
        SDL_sensor_lock = SDL_CreateMutex();
    }

#ifndef SDL_EVENTS_DISABLED
    if (SDL_InitSubSystem(SDL_INIT_EVENTS) < 0) {
        return -1;
    }
#endif /* !SDL_EVENTS_DISABLED */

    status = -1;
    for (i = 0; i < SDL_arraysize(SDL_sensor_drivers); ++i) {
        if (SDL_sensor_drivers[i]->Init() >= 0) {
            status = 0;
        }
    }
    return status;
}

/*
 * Count the number of sensors attached to the system
 */
int SDL_NumSensors(void)
{
    int i, total_sensors = 0;
    SDL_LockSensors();
    for (i = 0; i < SDL_arraysize(SDL_sensor_drivers); ++i) {
        total_sensors += SDL_sensor_drivers[i]->GetCount();
    }
    SDL_UnlockSensors();
    return total_sensors;
}

/*
 * Return the next available sensor instance ID
 * This may be called by drivers from multiple threads, unprotected by any locks
 */
SDL_SensorID SDL_GetNextSensorInstanceID(void)
{
    return SDL_AtomicIncRef(&SDL_next_sensor_instance_id);
}

/*
 * Get the driver and device index for an API device index
 * This should be called while the sensor lock is held, to prevent another thread from updating the list
 */
static SDL_bool SDL_GetDriverAndSensorIndex(int device_index, SDL_SensorDriver **driver, int *driver_index)
{
    int i, num_sensors, total_sensors = 0;

    if (device_index >= 0) {
        for (i = 0; i < SDL_arraysize(SDL_sensor_drivers); ++i) {
            num_sensors = SDL_sensor_drivers[i]->GetCount();
            if (device_index < num_sensors) {
                *driver = SDL_sensor_drivers[i];
                *driver_index = device_index;
                return SDL_TRUE;
            }
            device_index -= num_sensors;
            total_sensors += num_sensors;
        }
    }

    SDL_SetError("There are %d sensors available", total_sensors);
    return SDL_FALSE;
}

/*
 * Get the implementation dependent name of a sensor
 */
const char *SDL_SensorGetDeviceName(int device_index)
{
    SDL_SensorDriver *driver;
    const char *name = NULL;

    SDL_LockSensors();
    if (SDL_GetDriverAndSensorIndex(device_index, &driver, &device_index)) {
        name = driver->GetDeviceName(device_index);
    }
    SDL_UnlockSensors();

    /* FIXME: Really we should reference count this name so it doesn't go away after unlock */
    return name;
}

SDL_SensorType SDL_SensorGetDeviceType(int device_index)
{
    SDL_SensorDriver *driver;
    SDL_SensorType type = SDL_SENSOR_INVALID;

    SDL_LockSensors();
    if (SDL_GetDriverAndSensorIndex(device_index, &driver, &device_index)) {
        type = driver->GetDeviceType(device_index);
    }
    SDL_UnlockSensors();

    return type;
}

int SDL_SensorGetDeviceNonPortableType(int device_index)
{
    SDL_SensorDriver *driver;
    int type = -1;

    SDL_LockSensors();
    if (SDL_GetDriverAndSensorIndex(device_index, &driver, &device_index)) {
        type = driver->GetDeviceNonPortableType(device_index);
    }
    SDL_UnlockSensors();

    return type;
}

SDL_SensorID SDL_SensorGetDeviceInstanceID(int device_index)
{
    SDL_SensorDriver *driver;
    SDL_SensorID instance_id = -1;

    SDL_LockSensors();
    if (SDL_GetDriverAndSensorIndex(device_index, &driver, &device_index)) {
        instance_id = driver->GetDeviceInstanceID(device_index);
    }
    SDL_UnlockSensors();

    return instance_id;
}

/*
 * Open a sensor for use - the index passed as an argument refers to
 * the N'th sensor on the system.  This index is the value which will
 * identify this sensor in future sensor events.
 *
 * This function returns a sensor identifier, or NULL if an error occurred.
 */
SDL_Sensor *SDL_SensorOpen(int device_index)
{
    SDL_SensorDriver *driver;
    SDL_SensorID instance_id;
    SDL_Sensor *sensor;
    SDL_Sensor *sensorlist;
    const char *sensorname = NULL;

    SDL_LockSensors();

    if (!SDL_GetDriverAndSensorIndex(device_index, &driver, &device_index)) {
        SDL_UnlockSensors();
        return NULL;
    }

    sensorlist = SDL_sensors;
    /* If the sensor is already open, return it
     * it is important that we have a single sensor * for each instance id
     */
    instance_id = driver->GetDeviceInstanceID(device_index);
    while (sensorlist) {
        if (instance_id == sensorlist->instance_id) {
            sensor = sensorlist;
            ++sensor->ref_count;
            SDL_UnlockSensors();
            return sensor;
        }
        sensorlist = sensorlist->next;
    }

    /* Create and initialize the sensor */
    sensor = (SDL_Sensor *)SDL_calloc(sizeof(*sensor), 1);
    if (!sensor) {
        SDL_OutOfMemory();
        SDL_UnlockSensors();
        return NULL;
    }
    sensor->driver = driver;
    sensor->instance_id = instance_id;
    sensor->type = driver->GetDeviceType(device_index);
    sensor->non_portable_type = driver->GetDeviceNonPortableType(device_index);

    if (driver->Open(sensor, device_index) < 0) {
        SDL_free(sensor);
        SDL_UnlockSensors();
        return NULL;
    }

    sensorname = driver->GetDeviceName(device_index);
    if (sensorname) {
        sensor->name = SDL_strdup(sensorname);
    } else {
        sensor->name = NULL;
    }

    /* Add sensor to list */
    ++sensor->ref_count;
    /* Link the sensor in the list */
    sensor->next = SDL_sensors;
    SDL_sensors = sensor;

    SDL_UnlockSensors();

    driver->Update(sensor);

    return sensor;
}

/*
 * Find the SDL_Sensor that owns this instance id
 */
SDL_Sensor *SDL_SensorFromInstanceID(SDL_SensorID instance_id)
{
    SDL_Sensor *sensor;

    SDL_LockSensors();
    for (sensor = SDL_sensors; sensor; sensor = sensor->next) {
        if (sensor->instance_id == instance_id) {
            break;
        }
    }
    SDL_UnlockSensors();
    return sensor;
}

/*
 * Checks to make sure the sensor is valid.
 */
static int SDL_PrivateSensorValid(SDL_Sensor *sensor)
{
    int valid;

    if (!sensor) {
        SDL_SetError("Sensor hasn't been opened yet");
        valid = 0;
    } else {
        valid = 1;
    }

    return valid;
}

/*
 * Get the friendly name of this sensor
 */
const char *SDL_SensorGetName(SDL_Sensor *sensor)
{
    if (!SDL_PrivateSensorValid(sensor)) {
        return NULL;
    }

    return sensor->name;
}

/*
 * Get the type of this sensor
 */
SDL_SensorType SDL_SensorGetType(SDL_Sensor *sensor)
{
    if (!SDL_PrivateSensorValid(sensor)) {
        return SDL_SENSOR_INVALID;
    }

    return sensor->type;
}

/*
 * Get the platform dependent type of this sensor
 */
int SDL_SensorGetNonPortableType(SDL_Sensor *sensor)
{
    if (!SDL_PrivateSensorValid(sensor)) {
        return -1;
    }

    return sensor->non_portable_type;
}

/*
 * Get the instance id for this opened sensor
 */
SDL_SensorID SDL_SensorGetInstanceID(SDL_Sensor *sensor)
{
    if (!SDL_PrivateSensorValid(sensor)) {
        return -1;
    }

    return sensor->instance_id;
}

/*
 * Get the current state of this sensor
 */
int SDL_SensorGetData(SDL_Sensor *sensor, float *data, int num_values)
{
    return SDL_SensorGetDataWithTimestamp(sensor, NULL, data, num_values);
}

/*
 * Get the current state of this sensor
 */
int SDL_SensorGetDataWithTimestamp(SDL_Sensor *sensor, Uint64 *timestamp, float *data, int num_values)
{
    if (!SDL_PrivateSensorValid(sensor)) {
        return -1;
    }

    num_values = SDL_min(num_values, SDL_arraysize(sensor->data));
    SDL_memcpy(data, sensor->data, num_values * sizeof(*data));
    if (timestamp) {
        *timestamp = sensor->timestamp_us;
    }
    return 0;
}

/*
 * Close a sensor previously opened with SDL_SensorOpen()
 */
void SDL_SensorClose(SDL_Sensor *sensor)
{
    SDL_Sensor *sensorlist;
    SDL_Sensor *sensorlistprev;

    if (!SDL_PrivateSensorValid(sensor)) {
        return;
    }

    SDL_LockSensors();

    /* First decrement ref count */
    if (--sensor->ref_count > 0) {
        SDL_UnlockSensors();
        return;
    }

    if (SDL_updating_sensor) {
        SDL_UnlockSensors();
        return;
    }

    sensor->driver->Close(sensor);
    sensor->hwdata = NULL;

    sensorlist = SDL_sensors;
    sensorlistprev = NULL;
    while (sensorlist) {
        if (sensor == sensorlist) {
            if (sensorlistprev) {
                /* unlink this entry */
                sensorlistprev->next = sensorlist->next;
            } else {
                SDL_sensors = sensor->next;
            }
            break;
        }
        sensorlistprev = sensorlist;
        sensorlist = sensorlist->next;
    }

    SDL_free(sensor->name);

    /* Free the data associated with this sensor */
    SDL_free(sensor);

    SDL_UnlockSensors();
}

void SDL_SensorQuit(void)
{
    int i;

    SDL_LockSensors();

    /* Make sure we're not getting called in the middle of updating sensors */
    SDL_assert(!SDL_updating_sensor);

    /* Stop the event polling */
    while (SDL_sensors) {
        SDL_sensors->ref_count = 1;
        SDL_SensorClose(SDL_sensors);
    }

    /* Quit the sensor setup */
    for (i = 0; i < SDL_arraysize(SDL_sensor_drivers); ++i) {
        SDL_sensor_drivers[i]->Quit();
    }

    SDL_UnlockSensors();

#ifndef SDL_EVENTS_DISABLED
    SDL_QuitSubSystem(SDL_INIT_EVENTS);
#endif

    if (SDL_sensor_lock) {
        SDL_DestroyMutex(SDL_sensor_lock);
        SDL_sensor_lock = NULL;
    }
}

/* These are global for SDL_syssensor.c and SDL_events.c */

int SDL_PrivateSensorUpdate(SDL_Sensor *sensor, Uint64 timestamp_us, float *data, int num_values)
{
    int posted;

    /* Allow duplicate events, for things like steps and heartbeats */

    /* Update internal sensor state */
    num_values = SDL_min(num_values, SDL_arraysize(sensor->data));
    SDL_memcpy(sensor->data, data, num_values * sizeof(*data));
    sensor->timestamp_us = timestamp_us;

    /* Post the event, if desired */
    posted = 0;
#ifndef SDL_EVENTS_DISABLED
    if (SDL_GetEventState(SDL_SENSORUPDATE) == SDL_ENABLE) {
        SDL_Event event;
        event.type = SDL_SENSORUPDATE;
        event.sensor.which = sensor->instance_id;
        num_values = SDL_min(num_values, SDL_arraysize(event.sensor.data));
        SDL_memset(event.sensor.data, 0, sizeof(event.sensor.data));
        SDL_memcpy(event.sensor.data, data, num_values * sizeof(*data));
        event.sensor.timestamp_us = timestamp_us;
        posted = SDL_PushEvent(&event) == 1;
    }
#endif /* !SDL_EVENTS_DISABLED */
    return posted;
}

void SDL_SensorUpdate(void)
{
    int i;
    SDL_Sensor *sensor, *next;

    if (!SDL_WasInit(SDL_INIT_SENSOR)) {
        return;
    }

    SDL_LockSensors();

    if (SDL_updating_sensor) {
        /* The sensors are already being updated */
        SDL_UnlockSensors();
        return;
    }

    SDL_updating_sensor = SDL_TRUE;

    for (sensor = SDL_sensors; sensor; sensor = sensor->next) {
        sensor->driver->Update(sensor);
    }

    SDL_updating_sensor = SDL_FALSE;

    /* If any sensors were closed while updating, free them here */
    for (sensor = SDL_sensors; sensor; sensor = next) {
        next = sensor->next;
        if (sensor->ref_count <= 0) {
            SDL_SensorClose(sensor);
        }
    }

    /* this needs to happen AFTER walking the sensor list above, so that any
       dangling hardware data from removed devices can be free'd
     */
    for (i = 0; i < SDL_arraysize(SDL_sensor_drivers); ++i) {
        SDL_sensor_drivers[i]->Detect();
    }

    SDL_UnlockSensors();
}

/* vi: set ts=4 sw=4 expandtab: */
