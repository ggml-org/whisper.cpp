/*
    SDL_windows_main.c, placed in the public domain by Sam Lantinga  4/13/98

    The WinMain function -- calls your program's main() function
*/
#include "SDL_config.h"

#ifdef __WIN32__

/* Include this so we define UNICODE properly */
#include "../../core/windows/SDL_windows.h"
#include <shellapi.h> /* CommandLineToArgvW() */

/* Include the SDL main definition header */
#include "SDL.h"
#include "SDL_main.h"

#ifdef main
#undef main
#endif /* main */

/* Pop up an out of memory message, returns to Windows */
static BOOL OutOfMemory(void)
{
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fatal Error", "Out of memory - aborting", NULL);
    return FALSE;
}

#if defined(_MSC_VER)
/* The VC++ compiler needs main/wmain defined */
#define console_ansi_main main
#if UNICODE
#define console_wmain wmain
#endif
#endif

/* Gets the arguments with GetCommandLine, converts them to argc and argv
   and calls SDL_main */
static int main_getcmdline(void)
{
    LPWSTR *argvw;
    char **argv;
    int i, argc, result;

    argvw = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argvw) {
        return OutOfMemory();
    }

    /* Note that we need to be careful about how we allocate/free memory here.
     * If the application calls SDL_SetMemoryFunctions(), we can't rely on
     * SDL_free() to use the same allocator after SDL_main() returns.
     */

    /* Parse it into argv and argc */
    argv = (char **)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, (argc + 1) * sizeof(*argv));
    if (!argv) {
        return OutOfMemory();
    }
    for (i = 0; i < argc; ++i) {
        DWORD len;
        char *arg = WIN_StringToUTF8W(argvw[i]);
        if (!arg) {
            return OutOfMemory();
        }
        len = (DWORD)SDL_strlen(arg);
        argv[i] = (char *)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, (size_t)len + 1);
        if (!argv[i]) {
            return OutOfMemory();
        }
        SDL_memcpy(argv[i], arg, len);
        SDL_free(arg);
    }
    argv[i] = NULL;
    LocalFree(argvw);

    SDL_SetMainReady();

    /* Run the application main() code */
    result = SDL_main(argc, argv);

    /* Free argv, to avoid memory leak */
    for (i = 0; i < argc; ++i) {
        HeapFree(GetProcessHeap(), 0, argv[i]);
    }
    HeapFree(GetProcessHeap(), 0, argv);

    return result;
}

/* This is where execution begins [console apps, ansi] */
int console_ansi_main(int argc, char *argv[])
{
    return main_getcmdline();
}

#if UNICODE
/* This is where execution begins [console apps, unicode] */
int console_wmain(int argc, wchar_t *wargv[], wchar_t *wenvp)
{
    return main_getcmdline();
}
#endif

/* This is where execution begins [windowed apps] */
int WINAPI MINGW32_FORCEALIGN
WinMain(HINSTANCE hInst, HINSTANCE hPrev, LPSTR szCmdLine, int sw) /* NOLINT(readability-inconsistent-declaration-parameter-name) */
{
    return main_getcmdline();
}

#endif /* __WIN32__ */

/* vi: set ts=4 sw=4 expandtab: */
