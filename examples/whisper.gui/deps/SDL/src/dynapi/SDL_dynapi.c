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

#include "SDL_config.h"
#include "SDL_dynapi.h"

#if SDL_DYNAMIC_API

#define SDL_DYNAMIC_API_ENVVAR "SDL_DYNAMIC_API"

#if defined(__OS2__)
#define INCL_DOS
#define INCL_DOSERRORS
#include <os2.h>
#include <dos.h>
#endif

#include "SDL.h"

/* These headers have system specific definitions, so aren't included above */
#include "SDL_syswm.h"
#include "SDL_vulkan.h"

/* This is the version of the dynamic API. This doesn't match the SDL version
   and should not change until there's been a major revamp in API/ABI.
   So 2.0.5 adds functions over 2.0.4? This number doesn't change;
   the sizeof(jump_table) changes instead. But 2.1.0 changes how a function
   works in an incompatible way or removes a function? This number changes,
   since sizeof(jump_table) isn't sufficient anymore. It's likely
   we'll forget to bump every time we add a function, so this is the
   failsafe switch for major API change decisions. Respect it and use it
   sparingly. */
#define SDL_DYNAPI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

static void SDL_InitDynamicAPI(void);

/* BE CAREFUL CALLING ANY SDL CODE IN HERE, IT WILL BLOW UP.
   Even self-contained stuff might call SDL_Error and break everything. */

/* behold, the macro salsa! */

/* !!! FIXME: ...disabled...until we write it.  :) */
#define DISABLE_JUMP_MAGIC 1

#if DISABLE_JUMP_MAGIC
/* Can't use the macro for varargs nonsense. This is atrocious. */
#define SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, logname, prio)                                     \
    _static void SDLCALL SDL_Log##logname##name(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) \
    {                                                                                                        \
        va_list ap;                                                                                          \
        initcall;                                                                                            \
        va_start(ap, fmt);                                                                                   \
        jump_table.SDL_LogMessageV(category, SDL_LOG_PRIORITY_##prio, fmt, ap);                              \
        va_end(ap);                                                                                          \
    }

#define SDL_DYNAPI_VARARGS(_static, name, initcall)                                                                                       \
    _static int SDLCALL SDL_SetError##name(SDL_PRINTF_FORMAT_STRING const char *fmt, ...)                                                 \
    {                                                                                                                                     \
        char buf[128], *str = buf;                                                                                                        \
        int result;                                                                                                                       \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        result = jump_table.SDL_vsnprintf(buf, sizeof(buf), fmt, ap);                                                                     \
        va_end(ap);                                                                                                                       \
        if (result >= 0 && (size_t)result >= sizeof(buf)) {                                                                               \
            size_t len = (size_t)result + 1;                                                                                              \
            str = (char *)jump_table.SDL_malloc(len);                                                                                     \
            if (str) {                                                                                                                    \
                va_start(ap, fmt);                                                                                                        \
                result = jump_table.SDL_vsnprintf(str, len, fmt, ap);                                                                     \
                va_end(ap);                                                                                                               \
            }                                                                                                                             \
        }                                                                                                                                 \
        if (result >= 0) {                                                                                                                \
            result = jump_table.SDL_SetError("%s", str);                                                                                  \
        }                                                                                                                                 \
        if (str != buf) {                                                                                                                 \
            jump_table.SDL_free(str);                                                                                                     \
        }                                                                                                                                 \
        return result;                                                                                                                    \
    }                                                                                                                                     \
    _static int SDLCALL SDL_sscanf##name(const char *buf, SDL_SCANF_FORMAT_STRING const char *fmt, ...)                                   \
    {                                                                                                                                     \
        int retval;                                                                                                                       \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        retval = jump_table.SDL_vsscanf(buf, fmt, ap);                                                                                    \
        va_end(ap);                                                                                                                       \
        return retval;                                                                                                                    \
    }                                                                                                                                     \
    _static int SDLCALL SDL_snprintf##name(SDL_OUT_Z_CAP(maxlen) char *buf, size_t maxlen, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) \
    {                                                                                                                                     \
        int retval;                                                                                                                       \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        retval = jump_table.SDL_vsnprintf(buf, maxlen, fmt, ap);                                                                          \
        va_end(ap);                                                                                                                       \
        return retval;                                                                                                                    \
    }                                                                                                                                     \
    _static int SDLCALL SDL_asprintf##name(char **strp, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)                                    \
    {                                                                                                                                     \
        int retval;                                                                                                                       \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        retval = jump_table.SDL_vasprintf(strp, fmt, ap);                                                                                 \
        va_end(ap);                                                                                                                       \
        return retval;                                                                                                                    \
    }                                                                                                                                     \
    _static void SDLCALL SDL_Log##name(SDL_PRINTF_FORMAT_STRING const char *fmt, ...)                                                     \
    {                                                                                                                                     \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        jump_table.SDL_LogMessageV(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO, fmt, ap);                                         \
        va_end(ap);                                                                                                                       \
    }                                                                                                                                     \
    _static void SDLCALL SDL_LogMessage##name(int category, SDL_LogPriority priority, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)      \
    {                                                                                                                                     \
        va_list ap;                                                                                                                       \
        initcall;                                                                                                                         \
        va_start(ap, fmt);                                                                                                                \
        jump_table.SDL_LogMessageV(category, priority, fmt, ap);                                                                          \
        va_end(ap);                                                                                                                       \
    }                                                                                                                                     \
    SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, Verbose, VERBOSE)                                                                   \
    SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, Debug, DEBUG)                                                                       \
    SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, Info, INFO)                                                                         \
    SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, Warn, WARN)                                                                         \
    SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, Error, ERROR)                                                                       \
    SDL_DYNAPI_VARARGS_LOGFN(_static, name, initcall, Critical, CRITICAL)
#endif

/* Typedefs for function pointers for jump table, and predeclare funcs */
/* The DEFAULT funcs will init jump table and then call real function. */
/* The REAL funcs are the actual functions, name-mangled to not clash. */
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) \
    typedef rc (SDLCALL *SDL_DYNAPIFN_##fn) params;\
    static rc SDLCALL fn##_DEFAULT params;         \
    extern rc SDLCALL fn##_REAL params;
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC

/* The jump table! */
typedef struct
{
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) SDL_DYNAPIFN_##fn fn;
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
} SDL_DYNAPI_jump_table;

/* Predeclare the default functions for initializing the jump table. */
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) static rc SDLCALL fn##_DEFAULT params;
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC

/* The actual jump table. */
static SDL_DYNAPI_jump_table jump_table = {
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) fn##_DEFAULT,
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
};

/* Default functions init the function table then call right thing. */
#if DISABLE_JUMP_MAGIC
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) \
    static rc SDLCALL fn##_DEFAULT params          \
    {                                              \
        SDL_InitDynamicAPI();                      \
        ret jump_table.fn args;                    \
    }
#define SDL_DYNAPI_PROC_NO_VARARGS
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
#undef SDL_DYNAPI_PROC_NO_VARARGS
SDL_DYNAPI_VARARGS(static, _DEFAULT, SDL_InitDynamicAPI())
#else
/* !!! FIXME: need the jump magic. */
#error Write me.
#endif

/* Public API functions to jump into the jump table. */
#if DISABLE_JUMP_MAGIC
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) \
    rc SDLCALL fn params                           \
    {                                              \
        ret jump_table.fn args;                    \
    }
#define SDL_DYNAPI_PROC_NO_VARARGS
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
#undef SDL_DYNAPI_PROC_NO_VARARGS
SDL_DYNAPI_VARARGS(, , )
#else
/* !!! FIXME: need the jump magic. */
#error Write me.
#endif

#define ENABLE_SDL_CALL_LOGGING 0
#if ENABLE_SDL_CALL_LOGGING
static int SDLCALL SDL_SetError_LOGSDLCALLS(SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    char buf[512]; /* !!! FIXME: dynamic allocation */
    va_list ap;
    SDL_Log_REAL("SDL2CALL SDL_SetError");
    va_start(ap, fmt);
    SDL_vsnprintf_REAL(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return SDL_SetError_REAL("%s", buf);
}
static int SDLCALL SDL_sscanf_LOGSDLCALLS(const char *buf, SDL_SCANF_FORMAT_STRING const char *fmt, ...)
{
    int retval;
    va_list ap;
    SDL_Log_REAL("SDL2CALL SDL_sscanf");
    va_start(ap, fmt);
    retval = SDL_vsscanf_REAL(buf, fmt, ap);
    va_end(ap);
    return retval;
}
static int SDLCALL SDL_snprintf_LOGSDLCALLS(SDL_OUT_Z_CAP(maxlen) char *buf, size_t maxlen, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    int retval;
    va_list ap;
    SDL_Log_REAL("SDL2CALL SDL_snprintf");
    va_start(ap, fmt);
    retval = SDL_vsnprintf_REAL(buf, maxlen, fmt, ap);
    va_end(ap);
    return retval;
}
static int SDLCALL SDL_asprintf_LOGSDLCALLS(char **strp, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    int retval;
    va_list ap;
    SDL_Log_REAL("SDL2CALL SDL_asprintf");
    va_start(ap, fmt);
    retval = SDL_vasprintf_REAL(strp, fmt, ap);
    va_end(ap);
    return retval;
}
static void SDLCALL SDL_Log_LOGSDLCALLS(SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;
    SDL_Log_REAL("SDL2CALL SDL_Log");
    va_start(ap, fmt);
    SDL_LogMessageV_REAL(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO, fmt, ap);
    va_end(ap);
}
static void SDLCALL SDL_LogMessage_LOGSDLCALLS(int category, SDL_LogPriority priority, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    va_list ap;
    SDL_Log_REAL("SDL2CALL SDL_LogMessage");
    va_start(ap, fmt);
    SDL_LogMessageV_REAL(category, priority, fmt, ap);
    va_end(ap);
}
#define SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(logname, prio) \
    static void SDLCALL SDL_Log##logname##_LOGSDLCALLS(int category, SDL_PRINTF_FORMAT_STRING const char *fmt, ...) { \
        va_list ap; va_start(ap, fmt); \
        SDL_Log_REAL("SDL2CALL SDL_Log%s", #logname); \
        SDL_LogMessageV_REAL(category, SDL_LOG_PRIORITY_##prio, fmt, ap); \
        va_end(ap); \
    }
SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(Verbose, VERBOSE)
SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(Debug, DEBUG)
SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(Info, INFO)
SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(Warn, WARN)
SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(Error, ERROR)
SDL_DYNAPI_VARARGS_LOGFN_LOGSDLCALLS(Critical, CRITICAL)
#define SDL_DYNAPI_PROC(rc,fn,params,args,ret) \
    rc SDLCALL fn##_LOGSDLCALLS params { SDL_Log_REAL("SDL2CALL %s", #fn); ret fn##_REAL args; }
#define SDL_DYNAPI_PROC_NO_VARARGS
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
#undef SDL_DYNAPI_PROC_NO_VARARGS
#endif

/* we make this a static function so we can call the correct one without the
   system's dynamic linker resolving to the wrong version of this. */
static Sint32 initialize_jumptable(Uint32 apiver, void *table, Uint32 tablesize)
{
    SDL_DYNAPI_jump_table *output_jump_table = (SDL_DYNAPI_jump_table *)table;

    if (apiver != SDL_DYNAPI_VERSION) {
        /* !!! FIXME: can maybe handle older versions? */
        return -1; /* not compatible. */
    } else if (tablesize > sizeof(jump_table)) {
        return -1; /* newer version of SDL with functions we can't provide. */
    }

/* Init our jump table first. */
#if ENABLE_SDL_CALL_LOGGING
    {
        const char *env = SDL_getenv_REAL("SDL_DYNAPI_LOG_CALLS");
        const SDL_bool log_calls = (env && SDL_atoi_REAL(env));
        if (log_calls) {
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) jump_table.fn = fn##_LOGSDLCALLS;
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
        } else {
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) jump_table.fn = fn##_REAL;
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
        }
    }
#else
#define SDL_DYNAPI_PROC(rc, fn, params, args, ret) jump_table.fn = fn##_REAL;
#include "SDL_dynapi_procs.h"
#undef SDL_DYNAPI_PROC
#endif

    /* Then the external table... */
    if (output_jump_table != &jump_table) {
        jump_table.SDL_memcpy(output_jump_table, &jump_table, tablesize);
    }

    /* Safe to call SDL functions now; jump table is initialized! */

    return 0; /* success! */
}

/* Here's the exported entry point that fills in the jump table. */
/*  Use specific types when an "int" might suffice to keep this sane. */
typedef Sint32 (SDLCALL *SDL_DYNAPI_ENTRYFN)(Uint32 apiver, void *table, Uint32 tablesize);
extern DECLSPEC Sint32 SDLCALL SDL_DYNAPI_entry(Uint32, void *, Uint32);

Sint32 SDL_DYNAPI_entry(Uint32 apiver, void *table, Uint32 tablesize)
{
    return initialize_jumptable(apiver, table, tablesize);
}

#ifdef __cplusplus
}
#endif

/* Obviously we can't use SDL_LoadObject() to load SDL.  :)  */
/* Also obviously, we never close the loaded library. */
#if defined(WIN32) || defined(_WIN32) || defined(__CYGWIN__)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
static SDL_INLINE void *get_sdlapi_entry(const char *fname, const char *sym)
{
    HMODULE lib = LoadLibraryA(fname);
    void *retval = NULL;
    if (lib) {
        retval = (void *) GetProcAddress(lib, sym);
        if (!retval) {
            FreeLibrary(lib);
        }
    }
    return retval;
}

#elif defined(unix) || defined(__unix__) || defined(__APPLE__) || defined(__HAIKU__) || defined(__QNX__)
#include <dlfcn.h>
static SDL_INLINE void *get_sdlapi_entry(const char *fname, const char *sym)
{
    void *lib = dlopen(fname, RTLD_NOW | RTLD_LOCAL);
    void *retval = NULL;
    if (lib) {
        retval = dlsym(lib, sym);
        if (!retval) {
            dlclose(lib);
        }
    }
    return retval;
}

#elif defined(__OS2__)
static SDL_INLINE void *get_sdlapi_entry(const char *fname, const char *sym)
{
    HMODULE hmodule;
    PFN retval = NULL;
    char error[256];
    if (DosLoadModule(error, sizeof(error), fname, &hmodule) == NO_ERROR) {
        if (DosQueryProcAddr(hmodule, 0, sym, &retval) != NO_ERROR) {
            DosFreeModule(hmodule);
        }
    }
    return (void *)retval;
}

#else
#error Please define your platform.
#endif

static void dynapi_warn(const char *msg)
{
    const char *caption = "SDL Dynamic API Failure!";
/* SDL_ShowSimpleMessageBox() is a too heavy for here. */
#if (defined(WIN32) || defined(_WIN32) || defined(__CYGWIN__)) && !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    MessageBoxA(NULL, msg, caption, MB_OK | MB_ICONERROR);
#elif defined(HAVE_STDIO_H)
    fprintf(stderr, "\n\n%s\n%s\n\n", caption, msg);
    fflush(stderr);
#endif
}

/* This is not declared in any header, although it is shared between some
    parts of SDL, because we don't want anything calling it without an
    extremely good reason. */
#ifdef __cplusplus
extern "C" {
#endif
extern SDL_NORETURN void SDL_ExitProcess(int exitcode);
#if defined(__WATCOMC__)
#pragma aux SDL_ExitProcess aborts;
#endif
#ifdef __cplusplus
}
#endif

static void SDL_InitDynamicAPILocked(void)
{
    char *libname = SDL_getenv_REAL(SDL_DYNAMIC_API_ENVVAR);
    SDL_DYNAPI_ENTRYFN entry = NULL; /* funcs from here by default. */
    SDL_bool use_internal = SDL_TRUE;

    if (libname) {
        while (*libname && !entry) {
            char *ptr = libname;
            while (SDL_TRUE) {
                const char ch = *ptr;
                if ((ch == ',') || (ch == '\0')) {
                    *ptr = '\0';
                    entry = (SDL_DYNAPI_ENTRYFN)get_sdlapi_entry(libname, "SDL_DYNAPI_entry");
                    *ptr = ch;
                    libname = (ch == '\0') ? ptr : (ptr + 1);
                    break;
                } else {
                    ptr++;
                }
            }
        }
        if (!entry) {
            dynapi_warn("Couldn't load an overriding SDL library. Please fix or remove the " SDL_DYNAMIC_API_ENVVAR " environment variable. Using the default SDL.");
            /* Just fill in the function pointers from this library, later. */
        }
    }

    if (entry) {
        if (entry(SDL_DYNAPI_VERSION, &jump_table, sizeof(jump_table)) < 0) {
            dynapi_warn("Couldn't override SDL library. Using a newer SDL build might help. Please fix or remove the " SDL_DYNAMIC_API_ENVVAR " environment variable. Using the default SDL.");
            /* Just fill in the function pointers from this library, later. */
        } else {
            use_internal = SDL_FALSE; /* We overrode SDL! Don't use the internal version! */
        }
    }

    /* Just fill in the function pointers from this library. */
    if (use_internal) {
        if (initialize_jumptable(SDL_DYNAPI_VERSION, &jump_table, sizeof(jump_table)) < 0) {
            /* Now we're screwed. Should definitely abort now. */
            dynapi_warn("Failed to initialize internal SDL dynapi. As this would otherwise crash, we have to abort now.");
            SDL_ExitProcess(86);
        }
    }

    /* we intentionally never close the newly-loaded lib, of course. */
}

static void SDL_InitDynamicAPI(void)
{
    /* So the theory is that every function in the jump table defaults to
     *  calling this function, and then replaces itself with a version that
     *  doesn't call this function anymore. But it's possible that, in an
     *  extreme corner case, you can have a second thread hit this function
     *  while the jump table is being initialized by the first.
     * In this case, a spinlock is really painful compared to what spinlocks
     *  _should_ be used for, but this would only happen once, and should be
     *  insanely rare, as you would have to spin a thread outside of SDL (as
     *  SDL_CreateThread() would also call this function before building the
     *  new thread).
     */
    static SDL_bool already_initialized = SDL_FALSE;

    /* SDL_AtomicLock calls SDL mutex functions to emulate if
       SDL_ATOMIC_DISABLED, which we can't do here, so in such a
       configuration, you're on your own. */
    #ifndef SDL_ATOMIC_DISABLED
    static SDL_SpinLock lock = 0;
    SDL_AtomicLock_REAL(&lock);
    #endif

    if (!already_initialized) {
        SDL_InitDynamicAPILocked();
        already_initialized = SDL_TRUE;
    }

    #ifndef SDL_ATOMIC_DISABLED
    SDL_AtomicUnlock_REAL(&lock);
    #endif
}

#endif /* SDL_DYNAMIC_API */

/* vi: set ts=4 sw=4 expandtab: */
