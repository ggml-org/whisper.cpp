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

#ifdef SDL_FILESYSTEM_UNIX

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* System dependent filesystem routines                                */

#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <limits.h>
#include <fcntl.h>

#if defined(__FREEBSD__) || defined(__OPENBSD__)
#include <sys/sysctl.h>
#endif

#include "SDL_error.h"
#include "SDL_stdinc.h"
#include "SDL_filesystem.h"
#include "SDL_rwops.h"

/* QNX's /proc/self/exefile is a text file and not a symlink. */
#if !defined(__QNXNTO__)
static char *readSymLink(const char *path)
{
    char *retval = NULL;
    ssize_t len = 64;
    ssize_t rc = -1;

    while (1) {
        char *ptr = (char *)SDL_realloc(retval, (size_t)len);
        if (!ptr) {
            SDL_OutOfMemory();
            break;
        }

        retval = ptr;

        rc = readlink(path, retval, len);
        if (rc == -1) {
            break; /* not a symlink, i/o error, etc. */
        } else if (rc < len) {
            retval[rc] = '\0'; /* readlink doesn't null-terminate. */
            return retval;     /* we're good to go. */
        }

        len *= 2; /* grow buffer, try again. */
    }

    SDL_free(retval);
    return NULL;
}
#endif


#if defined(__OPENBSD__)
static char *search_path_for_binary(const char *bin)
{
    char *envr = SDL_getenv("PATH");
    size_t alloc_size;
    char *exe = NULL;
    char *start = envr;
    char *ptr;

    if (!envr) {
        SDL_SetError("No $PATH set");
        return NULL;
    }

    envr = SDL_strdup(envr);
    if (!envr) {
        SDL_OutOfMemory();
        return NULL;
    }

    SDL_assert(bin != NULL);

    alloc_size = SDL_strlen(bin) + SDL_strlen(envr) + 2;
    exe = (char *)SDL_malloc(alloc_size);

    do {
        ptr = SDL_strchr(start, ':'); /* find next $PATH separator. */
        if (ptr != start) {
            if (ptr) {
                *ptr = '\0';
            }

            /* build full binary path... */
            SDL_snprintf(exe, alloc_size, "%s%s%s", start, (ptr && (ptr[-1] == '/')) ? "" : "/", bin);

            if (access(exe, X_OK) == 0) { /* Exists as executable? We're done. */
                SDL_free(envr);
                return exe;
            }
        }
        start = ptr + 1; /* start points to beginning of next element. */
    } while (ptr);

    SDL_free(envr);
    SDL_free(exe);

    SDL_SetError("Process not found in $PATH");
    return NULL; /* doesn't exist in path. */
}
#endif

char *SDL_GetBasePath(void)
{
    char *retval = NULL;

#if defined(__FREEBSD__)
    char fullpath[PATH_MAX];
    size_t buflen = sizeof(fullpath);
    const int mib[] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1 };
    if (sysctl(mib, SDL_arraysize(mib), fullpath, &buflen, NULL, 0) != -1) {
        retval = SDL_strdup(fullpath);
        if (!retval) {
            SDL_OutOfMemory();
            return NULL;
        }
    }
#endif
#if defined(__OPENBSD__)
    /* Please note that this will fail if the process was launched with a relative path and $PWD + the cwd have changed, or argv is altered. So don't do that. Or add a new sysctl to OpenBSD. */
    char **cmdline;
    size_t len;
    const int mib[] = { CTL_KERN, KERN_PROC_ARGS, getpid(), KERN_PROC_ARGV };
    if (sysctl(mib, 4, NULL, &len, NULL, 0) != -1) {
        char *exe, *pwddst;
        char *realpathbuf = (char *)SDL_malloc(PATH_MAX + 1);
        if (!realpathbuf) {
            SDL_OutOfMemory();
            return NULL;
        }

        cmdline = SDL_malloc(len);
        if (!cmdline) {
            SDL_free(realpathbuf);
            SDL_OutOfMemory();
            return NULL;
        }

        sysctl(mib, 4, cmdline, &len, NULL, 0);

        exe = cmdline[0];
        pwddst = NULL;
        if (SDL_strchr(exe, '/') == NULL) { /* not a relative or absolute path, check $PATH for it */
            exe = search_path_for_binary(cmdline[0]);
        } else {
            if (exe && *exe == '.') {
                const char *pwd = SDL_getenv("PWD");
                if (pwd && *pwd) {
                    SDL_asprintf(&pwddst, "%s/%s", pwd, exe);
                }
            }
        }

        if (exe) {
            if (!pwddst) {
                if (realpath(exe, realpathbuf) != NULL) {
                    retval = realpathbuf;
                }
            } else {
                if (realpath(pwddst, realpathbuf) != NULL) {
                    retval = realpathbuf;
                }
                SDL_free(pwddst);
            }

            if (exe != cmdline[0]) {
                SDL_free(exe);
            }
        }

        if (!retval) {
            SDL_free(realpathbuf);
        }

        SDL_free(cmdline);
    }
#endif

    /* is a Linux-style /proc filesystem available? */
    if (!retval && (access("/proc", F_OK) == 0)) {
        /* !!! FIXME: after 2.0.6 ships, let's delete this code and just
                      use the /proc/%llu version. There's no reason to have
                      two copies of this plus all the #ifdefs. --ryan. */
#if defined(__FREEBSD__)
        retval = readSymLink("/proc/curproc/file");
#elif defined(__NETBSD__)
        retval = readSymLink("/proc/curproc/exe");
#elif defined(__SOLARIS__)
        retval = readSymLink("/proc/self/path/a.out");
#elif defined(__QNXNTO__)
        retval = SDL_LoadFile("/proc/self/exefile", NULL);
#else
        retval = readSymLink("/proc/self/exe"); /* linux. */
        if (!retval) {
            /* older kernels don't have /proc/self ... try PID version... */
            char path[64];
            const int rc = SDL_snprintf(path, sizeof(path),
                                        "/proc/%llu/exe",
                                        (unsigned long long)getpid());
            if ((rc > 0) && (rc < sizeof(path))) {
                retval = readSymLink(path);
            }
        }
#endif
    }

#if defined(__SOLARIS__)  /* try this as a fallback if /proc didn't pan out */
    if (!retval) {
        const char *path = getexecname();
        if ((path != NULL) && (path[0] == '/')) { /* must be absolute path... */
            retval = SDL_strdup(path);
            if (!retval) {
                SDL_OutOfMemory();
                return NULL;
            }
        }
    }
#endif

    /* If we had access to argv[0] here, we could check it for a path,
        or troll through $PATH looking for it, too. */

    if (retval) { /* chop off filename. */
        char *ptr = SDL_strrchr(retval, '/');
        if (ptr) {
            *(ptr + 1) = '\0';
        } else { /* shouldn't happen, but just in case... */
            SDL_free(retval);
            retval = NULL;
        }
    }

    if (retval) {
        /* try to shrink buffer... */
        char *ptr = (char *)SDL_realloc(retval, SDL_strlen(retval) + 1);
        if (ptr) {
            retval = ptr; /* oh well if it failed. */
        }
    }

    return retval;
}

char *SDL_GetPrefPath(const char *org, const char *app)
{
    /*
     * We use XDG's base directory spec, even if you're not on Linux.
     *  This isn't strictly correct, but the results are relatively sane
     *  in any case.
     *
     * http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
     */
    const char *envr = SDL_getenv("XDG_DATA_HOME");
    const char *append;
    char *retval = NULL;
    char *ptr = NULL;
    size_t len = 0;

    if (!app) {
        SDL_InvalidParamError("app");
        return NULL;
    }
    if (!org) {
        org = "";
    }

    if (!envr) {
        /* You end up with "$HOME/.local/share/Game Name 2" */
        envr = SDL_getenv("HOME");
        if (!envr) {
            /* we could take heroic measures with /etc/passwd, but oh well. */
            SDL_SetError("neither XDG_DATA_HOME nor HOME environment is set");
            return NULL;
        }
        append = "/.local/share/";
    } else {
        append = "/";
    }

    len = SDL_strlen(envr);
    if (envr[len - 1] == '/') {
        append += 1;
    }

    len += SDL_strlen(append) + SDL_strlen(org) + SDL_strlen(app) + 3;
    retval = (char *)SDL_malloc(len);
    if (!retval) {
        SDL_OutOfMemory();
        return NULL;
    }

    if (*org) {
        (void)SDL_snprintf(retval, len, "%s%s%s/%s/", envr, append, org, app);
    } else {
        (void)SDL_snprintf(retval, len, "%s%s%s/", envr, append, app);
    }

    for (ptr = retval + 1; *ptr; ptr++) {
        if (*ptr == '/') {
            *ptr = '\0';
            if (mkdir(retval, 0700) != 0 && errno != EEXIST) {
                goto error;
            }
            *ptr = '/';
        }
    }
    if (mkdir(retval, 0700) != 0 && errno != EEXIST) {
    error:
        SDL_SetError("Couldn't create directory '%s': '%s'", retval, strerror(errno));
        SDL_free(retval);
        return NULL;
    }

    return retval;
}

#endif /* SDL_FILESYSTEM_UNIX */

/* vi: set ts=4 sw=4 expandtab: */
