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

/* Simple error handling in SDL */

#include "SDL_error.h"
#include "SDL_error_c.h"

int SDL_SetError(SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    /* Ignore call if invalid format pointer was passed */
    if (fmt) {
        va_list ap;
        int result;
        SDL_error *error = SDL_GetErrBuf();

        error->error = 1; /* mark error as valid */

        va_start(ap, fmt);
        result = SDL_vsnprintf(error->str, error->len, fmt, ap);
        va_end(ap);

        if (result >= 0 && (size_t)result >= error->len && error->realloc_func) {
            size_t len = (size_t)result + 1;
            char *str = (char *)error->realloc_func(error->str, len);
            if (str) {
                error->str = str;
                error->len = len;
                va_start(ap, fmt);
                (void)SDL_vsnprintf(error->str, error->len, fmt, ap);
                va_end(ap);
            }
        }

        if (SDL_LogGetPriority(SDL_LOG_CATEGORY_ERROR) <= SDL_LOG_PRIORITY_DEBUG) {
            /* If we are in debug mode, print out the error message */
            SDL_LogDebug(SDL_LOG_CATEGORY_ERROR, "%s", error->str);
        }
    }

    return -1;
}

/* Available for backwards compatibility */
const char *SDL_GetError(void)
{
    const SDL_error *error = SDL_GetErrBuf();
    return error->error ? error->str : "";
}

void SDL_ClearError(void)
{
    SDL_GetErrBuf()->error = 0;
}

/* Very common errors go here */
int SDL_Error(SDL_errorcode code)
{
    switch (code) {
    case SDL_ENOMEM:
        return SDL_SetError("Out of memory");
    case SDL_EFREAD:
        return SDL_SetError("Error reading from datastream");
    case SDL_EFWRITE:
        return SDL_SetError("Error writing to datastream");
    case SDL_EFSEEK:
        return SDL_SetError("Error seeking in datastream");
    case SDL_UNSUPPORTED:
        return SDL_SetError("That operation is not supported");
    default:
        return SDL_SetError("Unknown SDL error");
    }
}

#ifdef TEST_ERROR
int main(int argc, char *argv[])
{
    char buffer[BUFSIZ + 1];

    SDL_SetError("Hi there!");
    printf("Error 1: %s\n", SDL_GetError());
    SDL_ClearError();
    SDL_memset(buffer, '1', BUFSIZ);
    buffer[BUFSIZ] = 0;
    SDL_SetError("This is the error: %s (%f)", buffer, 1.0);
    printf("Error 2: %s\n", SDL_GetError());
    exit(0);
}
#endif

char *SDL_GetErrorMsg(char *errstr, int maxlen)
{
    const SDL_error *error = SDL_GetErrBuf();

    if (error->error) {
        SDL_strlcpy(errstr, error->str, maxlen);
    } else {
        *errstr = '\0';
    }

    return errstr;
}

/* vi: set ts=4 sw=4 expandtab: */
