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
#include "./SDL_internal.h"

#include "SDL_hints.h"
#include "SDL_error.h"
#include "SDL_hints_c.h"

/* Assuming there aren't many hints set and they aren't being queried in
   critical performance paths, we'll just use linked lists here.
 */
typedef struct SDL_HintWatch
{
    SDL_HintCallback callback;
    void *userdata;
    struct SDL_HintWatch *next;
} SDL_HintWatch;

typedef struct SDL_Hint
{
    char *name;
    char *value;
    SDL_HintPriority priority;
    SDL_HintWatch *callbacks;
    struct SDL_Hint *next;
} SDL_Hint;

static SDL_Hint *SDL_hints;

SDL_bool SDL_SetHintWithPriority(const char *name, const char *value, SDL_HintPriority priority)
{
    const char *env;
    SDL_Hint *hint;
    SDL_HintWatch *entry;

    if (!name) {
        return SDL_FALSE;
    }

    env = SDL_getenv(name);
    if (env && priority < SDL_HINT_OVERRIDE) {
        return SDL_FALSE;
    }

    for (hint = SDL_hints; hint; hint = hint->next) {
        if (SDL_strcmp(name, hint->name) == 0) {
            if (priority < hint->priority) {
                return SDL_FALSE;
            }
            if (hint->value != value &&
                (!value || !hint->value || SDL_strcmp(hint->value, value) != 0)) {
                for (entry = hint->callbacks; entry;) {
                    /* Save the next entry in case this one is deleted */
                    SDL_HintWatch *next = entry->next;
                    entry->callback(entry->userdata, name, hint->value, value);
                    entry = next;
                }
                SDL_free(hint->value);
                hint->value = value ? SDL_strdup(value) : NULL;
            }
            hint->priority = priority;
            return SDL_TRUE;
        }
    }

    /* Couldn't find the hint, add a new one */
    hint = (SDL_Hint *)SDL_malloc(sizeof(*hint));
    if (!hint) {
        return SDL_FALSE;
    }
    hint->name = SDL_strdup(name);
    hint->value = value ? SDL_strdup(value) : NULL;
    hint->priority = priority;
    hint->callbacks = NULL;
    hint->next = SDL_hints;
    SDL_hints = hint;
    return SDL_TRUE;
}

SDL_bool SDL_ResetHint(const char *name)
{
    const char *env;
    SDL_Hint *hint;
    SDL_HintWatch *entry;

    if (!name) {
        return SDL_FALSE;
    }

    env = SDL_getenv(name);
    for (hint = SDL_hints; hint; hint = hint->next) {
        if (SDL_strcmp(name, hint->name) == 0) {
            if ((!env && hint->value) ||
                (env && !hint->value) ||
                (env && SDL_strcmp(env, hint->value) != 0)) {
                for (entry = hint->callbacks; entry;) {
                    /* Save the next entry in case this one is deleted */
                    SDL_HintWatch *next = entry->next;
                    entry->callback(entry->userdata, name, hint->value, env);
                    entry = next;
                }
            }
            SDL_free(hint->value);
            hint->value = NULL;
            hint->priority = SDL_HINT_DEFAULT;
            return SDL_TRUE;
        }
    }
    return SDL_FALSE;
}

void SDL_ResetHints(void)
{
    const char *env;
    SDL_Hint *hint;
    SDL_HintWatch *entry;

    for (hint = SDL_hints; hint; hint = hint->next) {
        env = SDL_getenv(hint->name);
        if ((!env && hint->value) ||
            (env && !hint->value) ||
            (env && SDL_strcmp(env, hint->value) != 0)) {
            for (entry = hint->callbacks; entry;) {
                /* Save the next entry in case this one is deleted */
                SDL_HintWatch *next = entry->next;
                entry->callback(entry->userdata, hint->name, hint->value, env);
                entry = next;
            }
        }
        SDL_free(hint->value);
        hint->value = NULL;
        hint->priority = SDL_HINT_DEFAULT;
    }
}

SDL_bool SDL_SetHint(const char *name, const char *value)
{
    return SDL_SetHintWithPriority(name, value, SDL_HINT_NORMAL);
}

const char *SDL_GetHint(const char *name)
{
    const char *env;
    SDL_Hint *hint;

    if (!name) {
        return NULL;
    }

    env = SDL_getenv(name);
    for (hint = SDL_hints; hint; hint = hint->next) {
        if (SDL_strcmp(name, hint->name) == 0) {
            if (!env || hint->priority == SDL_HINT_OVERRIDE) {
                return hint->value;
            }
            break;
        }
    }
    return env;
}

SDL_bool SDL_GetStringBoolean(const char *value, SDL_bool default_value)
{
    if (!value || !*value) {
        return default_value;
    }
    if (*value == '0' || SDL_strcasecmp(value, "false") == 0) {
        return SDL_FALSE;
    }
    return SDL_TRUE;
}

SDL_bool SDL_GetHintBoolean(const char *name, SDL_bool default_value)
{
    const char *hint = SDL_GetHint(name);
    return SDL_GetStringBoolean(hint, default_value);
}

void SDL_AddHintCallback(const char *name, SDL_HintCallback callback, void *userdata)
{
    SDL_Hint *hint;
    SDL_HintWatch *entry;
    const char *value;

    if (!name || !*name) {
        SDL_InvalidParamError("name");
        return;
    }
    if (!callback) {
        SDL_InvalidParamError("callback");
        return;
    }

    SDL_DelHintCallback(name, callback, userdata);

    entry = (SDL_HintWatch *)SDL_malloc(sizeof(*entry));
    if (!entry) {
        SDL_OutOfMemory();
        return;
    }
    entry->callback = callback;
    entry->userdata = userdata;

    for (hint = SDL_hints; hint; hint = hint->next) {
        if (SDL_strcmp(name, hint->name) == 0) {
            break;
        }
    }
    if (!hint) {
        /* Need to add a hint entry for this watcher */
        hint = (SDL_Hint *)SDL_malloc(sizeof(*hint));
        if (!hint) {
            SDL_OutOfMemory();
            SDL_free(entry);
            return;
        }
        hint->name = SDL_strdup(name);
        if (!hint->name) {
            SDL_free(entry);
            SDL_free(hint);
            SDL_OutOfMemory();
            return;
        }
        hint->value = NULL;
        hint->priority = SDL_HINT_DEFAULT;
        hint->callbacks = NULL;
        hint->next = SDL_hints;
        SDL_hints = hint;
    }

    /* Add it to the callbacks for this hint */
    entry->next = hint->callbacks;
    hint->callbacks = entry;

    /* Now call it with the current value */
    value = SDL_GetHint(name);
    callback(userdata, name, value, value);
}

void SDL_DelHintCallback(const char *name, SDL_HintCallback callback, void *userdata)
{
    SDL_Hint *hint;
    SDL_HintWatch *entry, *prev;

    for (hint = SDL_hints; hint; hint = hint->next) {
        if (SDL_strcmp(name, hint->name) == 0) {
            prev = NULL;
            for (entry = hint->callbacks; entry; entry = entry->next) {
                if (callback == entry->callback && userdata == entry->userdata) {
                    if (prev) {
                        prev->next = entry->next;
                    } else {
                        hint->callbacks = entry->next;
                    }
                    SDL_free(entry);
                    break;
                }
                prev = entry;
            }
            return;
        }
    }
}

void SDL_ClearHints(void)
{
    SDL_Hint *hint;
    SDL_HintWatch *entry;

    while (SDL_hints) {
        hint = SDL_hints;
        SDL_hints = hint->next;

        SDL_free(hint->name);
        SDL_free(hint->value);
        for (entry = hint->callbacks; entry;) {
            SDL_HintWatch *freeable = entry;
            entry = entry->next;
            SDL_free(freeable);
        }
        SDL_free(hint);
    }
}

/* vi: set ts=4 sw=4 expandtab: */
