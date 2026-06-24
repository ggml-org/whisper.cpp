/*
    SDL_dummy_main.c, placed in the public domain by Sam Lantinga  3/13/14
*/
#include "../../SDL_internal.h"

/* Include the SDL main definition header */
#include "SDL_main.h"

#ifdef main
#undef main
int main(int argc, char *argv[])
{
    return SDL_main(argc, argv);
}
#else
/* Nothing to do on this platform */
int SDL_main_stub_symbol(void)
{
    return 0;
}
#endif

/* vi: set ts=4 sw=4 expandtab: */
