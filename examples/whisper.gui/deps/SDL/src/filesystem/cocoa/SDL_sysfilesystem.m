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

#ifdef SDL_FILESYSTEM_COCOA

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* System dependent filesystem routines                                */

#include <Foundation/Foundation.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "SDL_error.h"
#include "SDL_stdinc.h"
#include "SDL_filesystem.h"

char *SDL_GetBasePath(void)
{
    @autoreleasepool {
        NSBundle *bundle = [NSBundle mainBundle];
        const char *baseType = [[[bundle infoDictionary] objectForKey:@"SDL_FILESYSTEM_BASE_DIR_TYPE"] UTF8String];
        const char *base = NULL;
        char *retval = NULL;

        if (baseType == NULL) {
            baseType = "resource";
        }
        if (SDL_strcasecmp(baseType, "bundle") == 0) {
            base = [[bundle bundlePath] fileSystemRepresentation];
        } else if (SDL_strcasecmp(baseType, "parent") == 0) {
            base = [[[bundle bundlePath] stringByDeletingLastPathComponent] fileSystemRepresentation];
        } else {
            /* this returns the exedir for non-bundled  and the resourceDir for bundled apps */
            base = [[bundle resourcePath] fileSystemRepresentation];
        }

        if (base) {
            const size_t len = SDL_strlen(base) + 2;
            retval = (char *)SDL_malloc(len);
            if (retval == NULL) {
                SDL_OutOfMemory();
            } else {
                SDL_snprintf(retval, len, "%s/", base);
            }
        }

        return retval;
    }
}

char *SDL_GetPrefPath(const char *org, const char *app)
{
    @autoreleasepool {
        char *retval = NULL;
        NSArray *array;

        if (!app) {
            SDL_InvalidParamError("app");
            return NULL;
        }
        if (!org) {
            org = "";
        }

#if !TARGET_OS_TV
        array = NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory, NSUserDomainMask, YES);
#else
        /* tvOS does not have persistent local storage!
         * The only place on-device where we can store data is
         * a cache directory that the OS can empty at any time.
         *
         * It's therefore very likely that save data will be erased
         * between sessions. If you want your app's save data to
         * actually stick around, you'll need to use iCloud storage.
         */
        {
            static SDL_bool shown = SDL_FALSE;
            if (!shown) {
                shown = SDL_TRUE;
                SDL_LogCritical(SDL_LOG_CATEGORY_SYSTEM, "tvOS does not have persistent local storage! Use iCloud storage if you want your data to persist between sessions.\n");
            }
        }

        array = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
#endif /* !TARGET_OS_TV */

        if ([array count] > 0) { /* we only want the first item in the list. */
            NSString *str = [array objectAtIndex:0];
            const char *base = [str fileSystemRepresentation];
            if (base) {
                const size_t len = SDL_strlen(base) + SDL_strlen(org) + SDL_strlen(app) + 4;
                retval = (char *)SDL_malloc(len);
                if (retval == NULL) {
                    SDL_OutOfMemory();
                } else {
                    char *ptr;
                    if (*org) {
                        SDL_snprintf(retval, len, "%s/%s/%s/", base, org, app);
                    } else {
                        SDL_snprintf(retval, len, "%s/%s/", base, app);
                    }
                    for (ptr = retval + 1; *ptr; ptr++) {
                        if (*ptr == '/') {
                            *ptr = '\0';
                            mkdir(retval, 0700);
                            *ptr = '/';
                        }
                    }
                    mkdir(retval, 0700);
                }
            }
        }

        return retval;
    }
}

#endif /* SDL_FILESYSTEM_COCOA */

/* vi: set ts=4 sw=4 expandtab: */
