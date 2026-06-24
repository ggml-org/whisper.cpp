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

extern "C" {
#include "SDL_system.h"
#include "../windows/SDL_windows.h"
#include "SDL_messagebox.h"
#include "SDL_main.h"
#include "SDL_events.h"
#include "../../events/SDL_events_c.h"
}
#include <XGameRuntime.h>
#include <xsapi-c/services_c.h>
#include <shellapi.h> /* CommandLineToArgvW() */
#include <appnotify.h>

static XTaskQueueHandle GDK_GlobalTaskQueue;

PAPPSTATE_REGISTRATION hPLM = {};
PAPPCONSTRAIN_REGISTRATION hCPLM = {};
HANDLE plmSuspendComplete = nullptr;

extern "C" DECLSPEC int
SDL_GDKGetTaskQueue(XTaskQueueHandle *outTaskQueue)
{
    /* If this is the first call, first create the global task queue. */
    if (!GDK_GlobalTaskQueue) {
        HRESULT hr;

        hr = XTaskQueueCreate(XTaskQueueDispatchMode::ThreadPool,
                              XTaskQueueDispatchMode::Manual,
                              &GDK_GlobalTaskQueue);
        if (FAILED(hr)) {
            return SDL_SetError("[GDK] Could not create global task queue");
        }

        /* The initial call gets the non-duplicated handle so they can clean it up */
        *outTaskQueue = GDK_GlobalTaskQueue;
    } else {
        /* Duplicate the global task queue handle into outTaskQueue */
        if (FAILED(XTaskQueueDuplicateHandle(GDK_GlobalTaskQueue, outTaskQueue))) {
            return SDL_SetError("[GDK] Unable to acquire global task queue");
        }
    }

    return 0;
}

extern "C" void
GDK_DispatchTaskQueue(void)
{
    /* If there is no global task queue, don't do anything.
     * This gives the option to opt-out for those who want to handle everything themselves.
     */
    if (GDK_GlobalTaskQueue) {
        /* Dispatch any callbacks which are ready. */
        while (XTaskQueueDispatch(GDK_GlobalTaskQueue, XTaskQueuePort::Completion, 0))
            ;
    }
}

/* Pop up an out of memory message, returns to Windows */
extern "C" static BOOL
OutOfMemory(void)
{
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fatal Error", "Out of memory - aborting", NULL);
    return FALSE;
}

/* Gets the arguments with GetCommandLine, converts them to argc and argv
   and calls SDL_main */
extern "C" DECLSPEC int
SDL_GDKRunApp(SDL_main_func mainFunction, void *reserved)
{
    LPWSTR *argvw;
    char **argv;
    int i, argc, result;
    HRESULT hr;
    XTaskQueueHandle taskQueue;

    argvw = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (argvw == NULL) {
        return OutOfMemory();
    }

    /* Note that we need to be careful about how we allocate/free memory here.
     * If the application calls SDL_SetMemoryFunctions(), we can't rely on
     * SDL_free() to use the same allocator after SDL_main() returns.
     */

    /* Parse it into argv and argc */
    argv = (char **)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, (argc + 1) * sizeof(*argv));
    if (argv == NULL) {
        return OutOfMemory();
    }
    for (i = 0; i < argc; ++i) {
        DWORD len;
        char *arg = WIN_StringToUTF8W(argvw[i]);
        if (arg == NULL) {
            return OutOfMemory();
        }
        len = (DWORD)SDL_strlen(arg);
        argv[i] = (char *)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, len + 1);
        if (!argv[i]) {
            return OutOfMemory();
        }
        SDL_memcpy(argv[i], arg, len);
        SDL_free(arg);
    }
    argv[i] = NULL;
    LocalFree(argvw);

    hr = XGameRuntimeInitialize();

    if (SUCCEEDED(hr) && SDL_GDKGetTaskQueue(&taskQueue) == 0) {
        Uint32 titleid = 0;
        char scidBuffer[64];
        XblInitArgs xblArgs;

        XTaskQueueSetCurrentProcessTaskQueue(taskQueue);

        /* Try to get the title ID and initialize Xbox Live */
        hr = XGameGetXboxTitleId(&titleid);
        if (SUCCEEDED(hr)) {
            SDL_zero(xblArgs);
            xblArgs.queue = taskQueue;
            SDL_snprintf(scidBuffer, 64, "00000000-0000-0000-0000-0000%08X", titleid);
            xblArgs.scid = scidBuffer;
            hr = XblInitialize(&xblArgs);
        } else {
            SDL_SetError("[GDK] Unable to get titleid. Will not call XblInitialize. Check MicrosoftGame.config!");
        }

        SDL_SetMainReady();

        /* Register suspend/resume handling */
        plmSuspendComplete = CreateEventEx(nullptr, nullptr, 0, EVENT_MODIFY_STATE | SYNCHRONIZE);
        if (!plmSuspendComplete) {
            SDL_SetError("[GDK] Unable to create plmSuspendComplete event");
            return -1;
        }
        auto rascn = [](BOOLEAN quiesced, PVOID context) {
            SDL_LogDebug(SDL_LOG_CATEGORY_APPLICATION, "[GDK] in RegisterAppStateChangeNotification handler");
            if (quiesced) {
                ResetEvent(plmSuspendComplete);
                SDL_SendAppEvent(SDL_APP_DIDENTERBACKGROUND);

                // To defer suspension, we must wait to exit this callback.
                // IMPORTANT: The app must call SDL_GDKSuspendComplete() to release this lock.
                (void)WaitForSingleObject(plmSuspendComplete, INFINITE);

                SDL_LogDebug(SDL_LOG_CATEGORY_APPLICATION, "[GDK] in RegisterAppStateChangeNotification handler: plmSuspendComplete event signaled.");
            } else {
                SDL_SendAppEvent(SDL_APP_WILLENTERFOREGROUND);
            }
        };
        if (RegisterAppStateChangeNotification(rascn, NULL, &hPLM)) {
            SDL_SetError("[GDK] Unable to call RegisterAppStateChangeNotification");
            return -1;
        }

        /* Register constrain/unconstrain handling */
        auto raccn = [](BOOLEAN constrained, PVOID context) {
            SDL_LogDebug(SDL_LOG_CATEGORY_APPLICATION, "[GDK] in RegisterAppConstrainedChangeNotification handler");
            SDL_VideoDevice *_this = SDL_GetVideoDevice();
            if (_this) {
                if (constrained) {
                    SDL_SetKeyboardFocus(NULL);
                } else {
                    SDL_SetKeyboardFocus(_this->windows);
                }
            }
        };
        if (RegisterAppConstrainedChangeNotification(raccn, NULL, &hCPLM)) {
            SDL_SetError("[GDK] Unable to call RegisterAppConstrainedChangeNotification");
            return -1;
        }

        /* Run the application main() code */
        result = mainFunction(argc, argv);

        /* Unregister suspend/resume handling */
        UnregisterAppStateChangeNotification(hPLM);
        CloseHandle(plmSuspendComplete);

        /* Unregister constrain/unconstrain handling */
        UnregisterAppConstrainedChangeNotification(hCPLM);

        /* !!! FIXME: This follows the docs exactly, but for some reason still leaks handles on exit? */
        /* Terminate the task queue and dispatch any pending tasks */
        XTaskQueueTerminate(taskQueue, false, nullptr, nullptr);
        while (XTaskQueueDispatch(taskQueue, XTaskQueuePort::Completion, 0))
            ;

        XTaskQueueCloseHandle(taskQueue);

        XGameRuntimeUninitialize();
    } else {
#ifdef __WINGDK__
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fatal Error", "[GDK] Could not initialize - aborting", NULL);
#else
        SDL_assert_always(0 && "[GDK] Could not initialize - aborting");
#endif
        result = -1;
    }

    /* Free argv, to avoid memory leak */
    for (i = 0; i < argc; ++i) {
        HeapFree(GetProcessHeap(), 0, argv[i]);
    }
    HeapFree(GetProcessHeap(), 0, argv);

    return result;
}

extern "C" DECLSPEC void
SDL_GDKSuspendComplete()
{
    if (plmSuspendComplete) {
        SetEvent(plmSuspendComplete);
    }
}

extern "C" DECLSPEC int
SDL_GDKGetDefaultUser(XUserHandle *outUserHandle)
{
    XAsyncBlock block = { 0 };
    HRESULT result;

    if (FAILED(result = XUserAddAsync(XUserAddOptions::AddDefaultUserAllowingUI, &block))) {
        return WIN_SetErrorFromHRESULT("XUserAddAsync", result);
    }

    do {
        result = XUserAddResult(&block, outUserHandle);
    } while (result == E_PENDING);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT("XUserAddResult", result);
    }

    return 0;
}

