#include "SDL.h"
#include <stdio.h>

#include EXPORT_HEADER

#if defined(_WIN32)
#include <windows.h>
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    return TRUE;
}
#endif

int MYLIBRARY_EXPORT mylibrary_init(void);
void MYLIBRARY_EXPORT mylibrary_quit(void);
int MYLIBRARY_EXPORT mylibrary_work(void);

int mylibrary_init(void) {
    SDL_SetMainReady();
    if (SDL_Init(0) < 0) {
        fprintf(stderr, "could not initialize sdl2: %s\n", SDL_GetError());
        return 1;
    }
    return 0;
}

void mylibrary_quit(void) {
    SDL_Quit();
}

int mylibrary_work(void) {
    SDL_Delay(100);
    return 0;
}
