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

#include "SDL_haptic.h"
#include "SDL_windowshaptic_c.h"

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

extern int SDL_XINPUT_HapticInit(void);
extern int SDL_XINPUT_HapticMaybeAddDevice(const DWORD dwUserid);
extern int SDL_XINPUT_HapticMaybeRemoveDevice(const DWORD dwUserid);
extern int SDL_XINPUT_HapticOpen(SDL_Haptic *haptic, SDL_hapticlist_item *item);
extern int SDL_XINPUT_JoystickSameHaptic(SDL_Haptic *haptic, SDL_Joystick *joystick);
extern int SDL_XINPUT_HapticOpenFromJoystick(SDL_Haptic *haptic, SDL_Joystick *joystick);
extern void SDL_XINPUT_HapticClose(SDL_Haptic *haptic);
extern void SDL_XINPUT_HapticQuit(void);
extern int SDL_XINPUT_HapticNewEffect(SDL_Haptic *haptic, struct haptic_effect *effect, SDL_HapticEffect *base);
extern int SDL_XINPUT_HapticUpdateEffect(SDL_Haptic *haptic, struct haptic_effect *effect, SDL_HapticEffect *data);
extern int SDL_XINPUT_HapticRunEffect(SDL_Haptic *haptic, struct haptic_effect *effect, Uint32 iterations);
extern int SDL_XINPUT_HapticStopEffect(SDL_Haptic *haptic, struct haptic_effect *effect);
extern void SDL_XINPUT_HapticDestroyEffect(SDL_Haptic *haptic, struct haptic_effect *effect);
extern int SDL_XINPUT_HapticGetEffectStatus(SDL_Haptic *haptic, struct haptic_effect *effect);
extern int SDL_XINPUT_HapticSetGain(SDL_Haptic *haptic, int gain);
extern int SDL_XINPUT_HapticSetAutocenter(SDL_Haptic *haptic, int autocenter);
extern int SDL_XINPUT_HapticPause(SDL_Haptic *haptic);
extern int SDL_XINPUT_HapticUnpause(SDL_Haptic *haptic);
extern int SDL_XINPUT_HapticStopAll(SDL_Haptic *haptic);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif

/* vi: set ts=4 sw=4 expandtab: */
