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

#ifndef SDL_sensor_c_h_
#define SDL_sensor_c_h_

#include "SDL_config.h"

struct _SDL_SensorDriver;

/* Useful functions and variables from SDL_sensor.c */
#include "SDL_sensor.h"

/* Function to get the next available sensor instance ID */
extern SDL_SensorID SDL_GetNextSensorInstanceID(void);

/* Initialization and shutdown functions */
extern int SDL_SensorInit(void);
extern void SDL_SensorQuit(void);

/* Internal event queueing functions */
extern int SDL_PrivateSensorUpdate(SDL_Sensor *sensor, Uint64 timestamp_us, float *data, int num_values);

#endif /* SDL_sensor_c_h_ */

/* vi: set ts=4 sw=4 expandtab: */
