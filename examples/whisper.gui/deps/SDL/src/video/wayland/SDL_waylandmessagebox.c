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

#ifdef SDL_VIDEO_DRIVER_WAYLAND

#include "SDL.h"
#include <stdlib.h> /* fgets */
#include <stdio.h> /* FILE, STDOUT_FILENO, fdopen, fclose */
#include <unistd.h> /* pid_t, pipe, fork, close, dup2, execvp, _exit */
#include <sys/wait.h> /* waitpid, WIFEXITED, WEXITSTATUS */
#include <string.h>   /* strerr */
#include <errno.h>

#include "SDL_waylandmessagebox.h"

#define ZENITY_VERSION_LEN 32 /* Number of bytes to read from zenity --version (including NUL)*/

#define MAX_BUTTONS 8 /* Maximum number of buttons supported */

static int run_zenity(const char **args, int fd_pipe[2]) {
    int status;
    pid_t pid1;

    pid1 = fork();
    if (pid1 == 0) { /* child process */
        close(fd_pipe[0]); /* no reading from pipe */
        /* write stdout in pipe */
        if (dup2(fd_pipe[1], STDOUT_FILENO) == -1) {
            _exit(128);
        }

        /* const casting argv is fine:
         * https://pubs.opengroup.org/onlinepubs/9699919799/functions/fexecve.html -> rational
         */
        execvp("zenity", (char **)args);
        _exit(129);
    } else if (pid1 < 0) { /* fork() failed */
        return SDL_SetError("fork() failed: %s", strerror(errno));
    } else { /* parent process */
        close(fd_pipe[1]); /* no writing to the pipe */
        if (waitpid(pid1, &status, 0) != pid1) {
            return SDL_SetError("Waiting on zenity failed: %s", strerror(errno));
        }

        if (!WIFEXITED(status)) {
            return SDL_SetError("zenity failed for some reason");
        }

        if (WEXITSTATUS(status) >= 128) {
            return SDL_SetError("zenity reported error or failed to launch: %d", WEXITSTATUS(status));
        }

        return 0; /* success! */
    }
}

static int get_zenity_version(int *major, int *minor) {
    int fd_pipe[2]; /* fd_pipe[0]: read end of pipe, fd_pipe[1]: write end of pipe */
    const char *argv[] = { "zenity", "--version", NULL };

    if (pipe(fd_pipe) != 0) { /* create a pipe */
        return SDL_SetError("pipe() failed: %s", strerror(errno));
    }

    if (run_zenity(argv, fd_pipe) == 0) {
        FILE *outputfp = NULL;
        char version_str[ZENITY_VERSION_LEN];
        char *version_ptr = NULL, *end_ptr = NULL;
        int tmp;

        outputfp = fdopen(fd_pipe[0], "r");
        if (!outputfp) {
            close(fd_pipe[0]);
            return SDL_SetError("failed to open pipe for reading: %s", strerror(errno));
        }

        version_ptr = fgets(version_str, ZENITY_VERSION_LEN, outputfp);
        (void)fclose(outputfp); /* will close underlying fd */

        /* we expect the version string is in the form of MAJOR.MINOR.MICRO
         * as described in meson.build. We'll ignore everything after that.
         */
        tmp = (int) SDL_strtol(version_ptr, &end_ptr, 10);
        if (tmp == 0 && end_ptr == version_ptr) {
            return SDL_SetError("failed to get zenity major version number");
        }
        *major = tmp;

        version_ptr = end_ptr + 1; /* skip the dot */
        tmp = (int) SDL_strtol(version_ptr, &end_ptr, 10);
        if (tmp == 0 && end_ptr == version_ptr) {
            return SDL_SetError("failed to get zenity minor version number");
        }
        *minor = tmp;

        return 0; /* success */
    }

    close(fd_pipe[0]);
    close(fd_pipe[1]);
    return -1; /* run_zenity should've called SDL_SetError() */
}

int Wayland_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonid) {
    int fd_pipe[2]; /* fd_pipe[0]: read end of pipe, fd_pipe[1]: write end of pipe */
    int zenity_major = 0, zenity_minor = 0, output_len = 0;
    int argc = 5, i;
    const char *argv[5 + 2 /* icon name */ + 2 /* title */ + 2 /* message */ + 2 * MAX_BUTTONS + 1 /* NULL */] = {
        "zenity", "--question", "--switch", "--no-wrap", "--no-markup"
    };

    if (messageboxdata->numbuttons > MAX_BUTTONS) {
        return SDL_SetError("Too many buttons (%d max allowed)", MAX_BUTTONS);
    }

    /* get zenity version so we know which arg to use */
    if (get_zenity_version(&zenity_major, &zenity_minor) != 0) {
        return -1; /* get_zenity_version() calls SDL_SetError(), so message is already set */
    }

    if (pipe(fd_pipe) != 0) { /* create a pipe */
        return SDL_SetError("pipe() failed: %s", strerror(errno));
    }

    /* https://gitlab.gnome.org/GNOME/zenity/-/commit/c686bdb1b45e95acf010efd9ca0c75527fbb4dea
     * This commit removed --icon-name without adding a deprecation notice.
     * We need to handle it gracefully, otherwise no message box will be shown.
     */
    argv[argc++] = zenity_major > 3 || (zenity_major == 3 && zenity_minor >= 90) ? "--icon" : "--icon-name";
    switch (messageboxdata->flags) {
    case SDL_MESSAGEBOX_ERROR:
        argv[argc++] = "dialog-error";
        break;
    case SDL_MESSAGEBOX_WARNING:
        argv[argc++] = "dialog-warning";
        break;
    case SDL_MESSAGEBOX_INFORMATION:
    default:
        argv[argc++] = "dialog-information";
        break;
    }

    if (messageboxdata->title && messageboxdata->title[0]) {
        argv[argc++] = "--title";
        argv[argc++] = messageboxdata->title;
    } else {
        argv[argc++] = "--title=\"\"";
    }

    if (messageboxdata->message && messageboxdata->message[0]) {
        argv[argc++] = "--text";
        argv[argc++] = messageboxdata->message;
    } else {
        argv[argc++] = "--text=\"\"";
    }

    for (i = 0; i < messageboxdata->numbuttons; ++i) {
        if (messageboxdata->buttons[i].text && messageboxdata->buttons[i].text[0]) {
            int len = SDL_strlen(messageboxdata->buttons[i].text);
            if (len > output_len) {
                output_len = len;
            }

            argv[argc++] = "--extra-button";
            argv[argc++] = messageboxdata->buttons[i].text;
        } else {
            argv[argc++] = "--extra-button=\"\"";
        }
    }
    argv[argc] = NULL;

    if (run_zenity(argv, fd_pipe) == 0) {
        FILE *outputfp = NULL;
        char *output = NULL;
        char *tmp = NULL;

        if (!buttonid) {
            /* if we don't need buttonid, we can return immediately */
            close(fd_pipe[0]);
            return 0; /* success */
        }
        *buttonid = -1;

        output = SDL_malloc(output_len + 1);
        if (!output) {
            close(fd_pipe[0]);
            return SDL_OutOfMemory();
        }
        output[0] = '\0';

        outputfp = fdopen(fd_pipe[0], "r");
        if (!outputfp) {
            SDL_free(output);
            close(fd_pipe[0]);
            return SDL_SetError("Couldn't open pipe for reading: %s", strerror(errno));
        }
        tmp = fgets(output, output_len + 1, outputfp);
        (void)fclose(outputfp);

        if ((!tmp) || (*tmp == '\0') || (*tmp == '\n')) {
            SDL_free(output);
            return 0; /* User simply closed the dialog */
        }

        /* It likes to add a newline... */
        tmp = SDL_strrchr(output, '\n');
        if (tmp) {
            *tmp = '\0';
        }

        /* Check which button got pressed */
        for (i = 0; i < messageboxdata->numbuttons; i += 1) {
            if (messageboxdata->buttons[i].text) {
                if (SDL_strcmp(output, messageboxdata->buttons[i].text) == 0) {
                    *buttonid = messageboxdata->buttons[i].buttonid;
                    break;
                }
            }
        }

        SDL_free(output);
        return 0; /* success! */
    }

    close(fd_pipe[0]);
    close(fd_pipe[1]);
    return -1; /* run_zenity() calls SDL_SetError(), so message is already set */
}

#endif /* SDL_VIDEO_DRIVER_WAYLAND */

/* vi: set ts=4 sw=4 expandtab: */
