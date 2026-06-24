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

#include "../SDL_sysurl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>

int SDL_SYS_OpenURL(const char *url)
{
    const pid_t pid1 = fork();
    if (pid1 == 0) { /* child process */
#ifdef USE_POSIX_SPAWN
        pid_t pid2;
        const char *args[] = { "xdg-open", url, NULL };
        /* Clear LD_PRELOAD so Chrome opens correctly when this application is launched by Steam */
        unsetenv("LD_PRELOAD");
        if (posix_spawnp(&pid2, args[0], NULL, NULL, (char **)args, environ) == 0) {
            /* Child process doesn't wait for possibly-blocking grandchild. */
            _exit(EXIT_SUCCESS);
        } else {
            _exit(EXIT_FAILURE);
        }
#else
        pid_t pid2;
        /* Clear LD_PRELOAD so Chrome opens correctly when this application is launched by Steam */
        unsetenv("LD_PRELOAD");
        /* Notice this is vfork and not fork! */
        pid2 = vfork();
        if (pid2 == 0) { /* Grandchild process will try to launch the url */
            execlp("xdg-open", "xdg-open", url, NULL);
            _exit(EXIT_FAILURE);
        } else if (pid2 < 0) { /* There was an error forking */
            _exit(EXIT_FAILURE);
        } else {
            /* Child process doesn't wait for possibly-blocking grandchild. */
            _exit(EXIT_SUCCESS);
        }
#endif /* USE_POSIX_SPAWN */
    } else if (pid1 < 0) {
        return SDL_SetError("fork() failed: %s", strerror(errno));
    } else {
        int status;
        if (waitpid(pid1, &status, 0) == pid1) {
            if (WIFEXITED(status)) {
                if (WEXITSTATUS(status) == 0) {
                    return 0; /* success! */
                } else {
                    return SDL_SetError("xdg-open reported error or failed to launch: %d", WEXITSTATUS(status));
                }
            } else {
                return SDL_SetError("xdg-open failed for some reason");
            }
        } else {
            return SDL_SetError("Waiting on xdg-open failed: %s", strerror(errno));
        }
    }
}

/* vi: set ts=4 sw=4 expandtab: */
