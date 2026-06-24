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

#ifdef __APPLE__
#import <Foundation/Foundation.h>

#include "SDL_rwopsbundlesupport.h"

/* For proper OS X applications, the resources are contained inside the application bundle.
 So the strategy is to first check the application bundle for the file, then fallback to the current working directory.
 Note: One additional corner-case is if the resource is in a framework's resource bundle instead of the app.
 We might want to use bundle identifiers, e.g. org.libsdl.sdl to get the bundle for the framework,
 but we would somehow need to know what the bundle identifiers we need to search are.
 Also, note the bundle layouts are different for iPhone and Mac.
*/
FILE *SDL_OpenFPFromBundleOrFallback(const char *file, const char *mode)
{
    @autoreleasepool {
        FILE *fp = NULL;
        NSFileManager *file_manager;
        NSString *resource_path;
        NSString *ns_string_file_component;
        NSString *full_path_with_file_to_try;

        /* If the file mode is writable, skip all the bundle stuff because generally the bundle is read-only. */
        if (SDL_strchr(mode, 'r') == NULL) {
            return fopen(file, mode);
        }

        file_manager = [NSFileManager defaultManager];
        resource_path = [[NSBundle mainBundle] resourcePath];

        ns_string_file_component = [file_manager stringWithFileSystemRepresentation:file length:strlen(file)];

        full_path_with_file_to_try = [resource_path stringByAppendingPathComponent:ns_string_file_component];
        if ([file_manager fileExistsAtPath:full_path_with_file_to_try]) {
            fp = fopen([full_path_with_file_to_try fileSystemRepresentation], mode);
        } else {
            fp = fopen(file, mode);
        }

        return fp;
    }
}

#endif /* __APPLE__ */

/* vi: set ts=4 sw=4 expandtab: */
