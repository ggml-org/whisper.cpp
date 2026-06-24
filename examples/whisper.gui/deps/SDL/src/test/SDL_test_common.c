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

/* Ported from original test/common.c file. */

#include "SDL_config.h"
#include "SDL_test.h"

#include <stdio.h>

static const char *video_usage[] = {
    "[--video driver]", "[--renderer driver]", "[--gldebug]",
    "[--info all|video|modes|render|event|event_motion]",
    "[--log all|error|system|audio|video|render|input]", "[--display N]",
    "[--metal-window | --opengl-window | --vulkan-window]",
    "[--fullscreen | --fullscreen-desktop | --windows N]", "[--title title]",
    "[--icon icon.bmp]", "[--center | --position X,Y]", "[--geometry WxH]",
    "[--min-geometry WxH]", "[--max-geometry WxH]", "[--logical WxH]",
    "[--scale N]", "[--depth N]", "[--refresh R]", "[--vsync]", "[--noframe]",
    "[--resizable]", "[--minimize]", "[--maximize]", "[--grab]", "[--keyboard-grab]",
    "[--shown]", "[--hidden]", "[--input-focus]", "[--mouse-focus]",
    "[--flash-on-focus-loss]", "[--allow-highdpi]", "[--confine-cursor X,Y,W,H]",
    "[--usable-bounds]"
};

static const char *audio_usage[] = {
    "[--rate N]", "[--format U8|S8|U16|U16LE|U16BE|S16|S16LE|S16BE]",
    "[--channels N]", "[--samples N]"
};

static void SDL_snprintfcat(SDL_OUT_Z_CAP(maxlen) char *text, size_t maxlen, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    size_t length = SDL_strlen(text);
    va_list ap;

    va_start(ap, fmt);
    text += length;
    maxlen -= length;
    (void)SDL_vsnprintf(text, maxlen, fmt, ap);
    va_end(ap);
}

SDLTest_CommonState *SDLTest_CommonCreateState(char **argv, Uint32 flags)
{
    int i;
    SDLTest_CommonState *state;

    /* Do this first so we catch all allocations */
    for (i = 1; argv[i]; ++i) {
        if (SDL_strcasecmp(argv[i], "--trackmem") == 0) {
            SDLTest_TrackAllocations();
            break;
        }
    }

    state = (SDLTest_CommonState *)SDL_calloc(1, sizeof(*state));
    if (!state) {
        SDL_OutOfMemory();
        return NULL;
    }

    /* Initialize some defaults */
    state->argv = argv;
    state->flags = flags;
    state->window_title = argv[0];
    state->window_flags = 0;
    state->window_x = SDL_WINDOWPOS_UNDEFINED;
    state->window_y = SDL_WINDOWPOS_UNDEFINED;
    state->window_w = DEFAULT_WINDOW_WIDTH;
    state->window_h = DEFAULT_WINDOW_HEIGHT;
    state->num_windows = 1;
    state->audiospec.freq = 22050;
    state->audiospec.format = AUDIO_S16;
    state->audiospec.channels = 2;
    state->audiospec.samples = 2048;

    /* Set some very sane GL defaults */
    state->gl_red_size = 3;
    state->gl_green_size = 3;
    state->gl_blue_size = 2;
    state->gl_alpha_size = 0;
    state->gl_buffer_size = 0;
    state->gl_depth_size = 16;
    state->gl_stencil_size = 0;
    state->gl_double_buffer = 1;
    state->gl_accum_red_size = 0;
    state->gl_accum_green_size = 0;
    state->gl_accum_blue_size = 0;
    state->gl_accum_alpha_size = 0;
    state->gl_stereo = 0;
    state->gl_multisamplebuffers = 0;
    state->gl_multisamplesamples = 0;
    state->gl_retained_backing = 1;
    state->gl_accelerated = -1;
    state->gl_debug = 0;

    return state;
}

#define SEARCHARG(dim)            \
    while (*dim && *dim != ',') { \
        ++dim;                    \
    }                             \
    if (!*dim) {                  \
        return -1;                \
    }                             \
    *dim++ = '\0';

int SDLTest_CommonArg(SDLTest_CommonState *state, int index)
{
    char **argv = state->argv;

    if (SDL_strcasecmp(argv[index], "--video") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->videodriver = argv[index];
        SDL_SetHint(SDL_HINT_VIDEODRIVER, state->videodriver);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--renderer") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->renderdriver = argv[index];
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--gldebug") == 0) {
        state->gl_debug = 1;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--info") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        if (SDL_strcasecmp(argv[index], "all") == 0) {
            state->verbose |=
                (VERBOSE_VIDEO | VERBOSE_MODES | VERBOSE_RENDER |
                 VERBOSE_EVENT);
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "video") == 0) {
            state->verbose |= VERBOSE_VIDEO;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "modes") == 0) {
            state->verbose |= VERBOSE_MODES;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "render") == 0) {
            state->verbose |= VERBOSE_RENDER;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "event") == 0) {
            state->verbose |= VERBOSE_EVENT;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "event_motion") == 0) {
            state->verbose |= (VERBOSE_EVENT | VERBOSE_MOTION);
            return 2;
        }
        return -1;
    }
    if (SDL_strcasecmp(argv[index], "--log") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        if (SDL_strcasecmp(argv[index], "all") == 0) {
            SDL_LogSetAllPriority(SDL_LOG_PRIORITY_VERBOSE);
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "error") == 0) {
            SDL_LogSetPriority(SDL_LOG_CATEGORY_ERROR, SDL_LOG_PRIORITY_VERBOSE);
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "system") == 0) {
            SDL_LogSetPriority(SDL_LOG_CATEGORY_SYSTEM, SDL_LOG_PRIORITY_VERBOSE);
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "audio") == 0) {
            SDL_LogSetPriority(SDL_LOG_CATEGORY_AUDIO, SDL_LOG_PRIORITY_VERBOSE);
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "video") == 0) {
            SDL_LogSetPriority(SDL_LOG_CATEGORY_VIDEO, SDL_LOG_PRIORITY_VERBOSE);
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "render") == 0) {
            SDL_LogSetPriority(SDL_LOG_CATEGORY_RENDER, SDL_LOG_PRIORITY_VERBOSE);
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "input") == 0) {
            SDL_LogSetPriority(SDL_LOG_CATEGORY_INPUT, SDL_LOG_PRIORITY_VERBOSE);
            return 2;
        }
        return -1;
    }
    if (SDL_strcasecmp(argv[index], "--display") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->display = SDL_atoi(argv[index]);
        if (SDL_WINDOWPOS_ISUNDEFINED(state->window_x)) {
            state->window_x = SDL_WINDOWPOS_UNDEFINED_DISPLAY(state->display);
            state->window_y = SDL_WINDOWPOS_UNDEFINED_DISPLAY(state->display);
        }
        if (SDL_WINDOWPOS_ISCENTERED(state->window_x)) {
            state->window_x = SDL_WINDOWPOS_CENTERED_DISPLAY(state->display);
            state->window_y = SDL_WINDOWPOS_CENTERED_DISPLAY(state->display);
        }
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--metal-window") == 0) {
        state->window_flags |= SDL_WINDOW_METAL;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--opengl-window") == 0) {
        state->window_flags |= SDL_WINDOW_OPENGL;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--vulkan-window") == 0) {
        state->window_flags |= SDL_WINDOW_VULKAN;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--fullscreen") == 0) {
        state->window_flags |= SDL_WINDOW_FULLSCREEN;
        state->num_windows = 1;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--fullscreen-desktop") == 0) {
        state->window_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
        state->num_windows = 1;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--allow-highdpi") == 0) {
        state->window_flags |= SDL_WINDOW_ALLOW_HIGHDPI;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--windows") == 0) {
        ++index;
        if (!argv[index] || !SDL_isdigit((unsigned char)*argv[index])) {
            return -1;
        }
        if (!(state->window_flags & SDL_WINDOW_FULLSCREEN)) {
            state->num_windows = SDL_atoi(argv[index]);
        }
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--title") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->window_title = argv[index];
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--icon") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->window_icon = argv[index];
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--center") == 0) {
        state->window_x = SDL_WINDOWPOS_CENTERED;
        state->window_y = SDL_WINDOWPOS_CENTERED;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--position") == 0) {
        char *x, *y;
        ++index;
        if (!argv[index]) {
            return -1;
        }
        x = argv[index];
        y = argv[index];
        while (*y && *y != ',') {
            ++y;
        }
        if (!*y) {
            return -1;
        }
        *y++ = '\0';
        state->window_x = SDL_atoi(x);
        state->window_y = SDL_atoi(y);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--confine-cursor") == 0) {
        char *x, *y, *w, *h;
        ++index;
        if (!argv[index]) {
            return -1;
        }
        x = argv[index];
        y = argv[index];
        SEARCHARG(y)
        w = y;
        SEARCHARG(w)
        h = w;
        SEARCHARG(h)
        state->confine.x = SDL_atoi(x);
        state->confine.y = SDL_atoi(y);
        state->confine.w = SDL_atoi(w);
        state->confine.h = SDL_atoi(h);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--usable-bounds") == 0) {
        /* !!! FIXME: this is a bit of a hack, but I don't want to add a
           !!! FIXME:  flag to the public structure in 2.0.x */
        state->window_x = -1;
        state->window_y = -1;
        state->window_w = -1;
        state->window_h = -1;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--geometry") == 0) {
        char *w, *h;
        ++index;
        if (!argv[index]) {
            return -1;
        }
        w = argv[index];
        h = argv[index];
        while (*h && *h != 'x') {
            ++h;
        }
        if (!*h) {
            return -1;
        }
        *h++ = '\0';
        state->window_w = SDL_atoi(w);
        state->window_h = SDL_atoi(h);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--min-geometry") == 0) {
        char *w, *h;
        ++index;
        if (!argv[index]) {
            return -1;
        }
        w = argv[index];
        h = argv[index];
        while (*h && *h != 'x') {
            ++h;
        }
        if (!*h) {
            return -1;
        }
        *h++ = '\0';
        state->window_minW = SDL_atoi(w);
        state->window_minH = SDL_atoi(h);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--max-geometry") == 0) {
        char *w, *h;
        ++index;
        if (!argv[index]) {
            return -1;
        }
        w = argv[index];
        h = argv[index];
        while (*h && *h != 'x') {
            ++h;
        }
        if (!*h) {
            return -1;
        }
        *h++ = '\0';
        state->window_maxW = SDL_atoi(w);
        state->window_maxH = SDL_atoi(h);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--logical") == 0) {
        char *w, *h;
        ++index;
        if (!argv[index]) {
            return -1;
        }
        w = argv[index];
        h = argv[index];
        while (*h && *h != 'x') {
            ++h;
        }
        if (!*h) {
            return -1;
        }
        *h++ = '\0';
        state->logical_w = SDL_atoi(w);
        state->logical_h = SDL_atoi(h);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--scale") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->scale = (float)SDL_atof(argv[index]);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--depth") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->depth = SDL_atoi(argv[index]);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--refresh") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->refresh_rate = SDL_atoi(argv[index]);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--vsync") == 0) {
        state->render_flags |= SDL_RENDERER_PRESENTVSYNC;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--noframe") == 0) {
        state->window_flags |= SDL_WINDOW_BORDERLESS;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--resizable") == 0) {
        state->window_flags |= SDL_WINDOW_RESIZABLE;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--minimize") == 0) {
        state->window_flags |= SDL_WINDOW_MINIMIZED;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--maximize") == 0) {
        state->window_flags |= SDL_WINDOW_MAXIMIZED;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--shown") == 0) {
        state->window_flags |= SDL_WINDOW_SHOWN;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--hidden") == 0) {
        state->window_flags |= SDL_WINDOW_HIDDEN;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--input-focus") == 0) {
        state->window_flags |= SDL_WINDOW_INPUT_FOCUS;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--mouse-focus") == 0) {
        state->window_flags |= SDL_WINDOW_MOUSE_FOCUS;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--flash-on-focus-loss") == 0) {
        state->flash_on_focus_loss = SDL_TRUE;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--grab") == 0) {
        state->window_flags |= SDL_WINDOW_MOUSE_GRABBED;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--keyboard-grab") == 0) {
        state->window_flags |= SDL_WINDOW_KEYBOARD_GRABBED;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--rate") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->audiospec.freq = SDL_atoi(argv[index]);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--format") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        if (SDL_strcasecmp(argv[index], "U8") == 0) {
            state->audiospec.format = AUDIO_U8;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "S8") == 0) {
            state->audiospec.format = AUDIO_S8;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "U16") == 0) {
            state->audiospec.format = AUDIO_U16;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "U16LE") == 0) {
            state->audiospec.format = AUDIO_U16LSB;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "U16BE") == 0) {
            state->audiospec.format = AUDIO_U16MSB;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "S16") == 0) {
            state->audiospec.format = AUDIO_S16;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "S16LE") == 0) {
            state->audiospec.format = AUDIO_S16LSB;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "S16BE") == 0) {
            state->audiospec.format = AUDIO_S16MSB;
            return 2;
        }
        return -1;
    }
    if (SDL_strcasecmp(argv[index], "--channels") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->audiospec.channels = (Uint8)SDL_atoi(argv[index]);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--samples") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->audiospec.samples = (Uint16)SDL_atoi(argv[index]);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--trackmem") == 0) {
        /* Already handled in SDLTest_CommonCreateState() */
        return 1;
    }
    if ((SDL_strcasecmp(argv[index], "-h") == 0) || (SDL_strcasecmp(argv[index], "--help") == 0)) {
        /* Print the usage message */
        return -1;
    }
    if (SDL_strcmp(argv[index], "-NSDocumentRevisionsDebugMode") == 0) {
        /* Debug flag sent by Xcode */
        return 2;
    }
    return 0;
}

void SDLTest_CommonLogUsage(SDLTest_CommonState *state, const char *argv0, const char **options)
{
    int i;

    SDL_Log("USAGE: %s", argv0);
    SDL_Log("    %s", "[--trackmem]");

    if (state->flags & SDL_INIT_VIDEO) {
        for (i = 0; i < SDL_arraysize(video_usage); i++) {
            SDL_Log("    %s", video_usage[i]);
        }
    }

    if (state->flags & SDL_INIT_AUDIO) {
        for (i = 0; i < SDL_arraysize(audio_usage); i++) {
            SDL_Log("    %s", audio_usage[i]);
        }
    }

    if (options) {
        for (i = 0; options[i] != NULL; i++) {
            SDL_Log("    %s", options[i]);
        }
    }
}

static const char *BuildCommonUsageString(char **pstr, const char **strlist, const int numitems, const char **strlist2, const int numitems2)
{
    char *str = *pstr;
    if (!str) {
        size_t len = SDL_strlen("[--trackmem]") + 2;
        int i;
        for (i = 0; i < numitems; i++) {
            len += SDL_strlen(strlist[i]) + 1;
        }
        if (strlist2) {
            for (i = 0; i < numitems2; i++) {
                len += SDL_strlen(strlist2[i]) + 1;
            }
        }
        str = (char *)SDL_calloc(1, len);
        if (!str) {
            return ""; /* oh well. */
        }
        SDL_strlcat(str, "[--trackmem] ", len);
        for (i = 0; i < numitems - 1; i++) {
            SDL_strlcat(str, strlist[i], len);
            SDL_strlcat(str, " ", len);
        }
        SDL_strlcat(str, strlist[i], len);
        if (strlist2) {
            SDL_strlcat(str, " ", len);
            for (i = 0; i < numitems2 - 1; i++) {
                SDL_strlcat(str, strlist2[i], len);
                SDL_strlcat(str, " ", len);
            }
            SDL_strlcat(str, strlist2[i], len);
        }
        *pstr = str;
    }
    return str;
}

static char *common_usage_video = NULL;
static char *common_usage_audio = NULL;
static char *common_usage_videoaudio = NULL;

const char *SDLTest_CommonUsage(SDLTest_CommonState *state)
{

    switch (state->flags & (SDL_INIT_VIDEO | SDL_INIT_AUDIO)) {
    case SDL_INIT_VIDEO:
        return BuildCommonUsageString(&common_usage_video, video_usage, SDL_arraysize(video_usage), NULL, 0);
    case SDL_INIT_AUDIO:
        return BuildCommonUsageString(&common_usage_audio, audio_usage, SDL_arraysize(audio_usage), NULL, 0);
    case (SDL_INIT_VIDEO | SDL_INIT_AUDIO):
        return BuildCommonUsageString(&common_usage_videoaudio, video_usage, SDL_arraysize(video_usage), audio_usage, SDL_arraysize(audio_usage));
    default:
        return "[--trackmem]";
    }
}

SDL_bool SDLTest_CommonDefaultArgs(SDLTest_CommonState *state, const int argc, char **argv)
{
    int i = 1;
    while (i < argc) {
        const int consumed = SDLTest_CommonArg(state, i);
        if (consumed == 0) {
            SDLTest_CommonLogUsage(state, argv[0], NULL);
            return SDL_FALSE;
        }
        i += consumed;
    }
    return SDL_TRUE;
}

static void SDLTest_PrintDisplayOrientation(char *text, size_t maxlen, SDL_DisplayOrientation orientation)
{
    switch (orientation) {
    case SDL_ORIENTATION_UNKNOWN:
        SDL_snprintfcat(text, maxlen, "UNKNOWN");
        break;
    case SDL_ORIENTATION_LANDSCAPE:
        SDL_snprintfcat(text, maxlen, "LANDSCAPE");
        break;
    case SDL_ORIENTATION_LANDSCAPE_FLIPPED:
        SDL_snprintfcat(text, maxlen, "LANDSCAPE_FLIPPED");
        break;
    case SDL_ORIENTATION_PORTRAIT:
        SDL_snprintfcat(text, maxlen, "PORTRAIT");
        break;
    case SDL_ORIENTATION_PORTRAIT_FLIPPED:
        SDL_snprintfcat(text, maxlen, "PORTRAIT_FLIPPED");
        break;
    default:
        SDL_snprintfcat(text, maxlen, "0x%8.8x", orientation);
        break;
    }
}

static void SDLTest_PrintWindowFlag(char *text, size_t maxlen, Uint32 flag)
{
    switch (flag) {
    case SDL_WINDOW_FULLSCREEN:
        SDL_snprintfcat(text, maxlen, "FULLSCREEN");
        break;
    case SDL_WINDOW_OPENGL:
        SDL_snprintfcat(text, maxlen, "OPENGL");
        break;
    case SDL_WINDOW_SHOWN:
        SDL_snprintfcat(text, maxlen, "SHOWN");
        break;
    case SDL_WINDOW_HIDDEN:
        SDL_snprintfcat(text, maxlen, "HIDDEN");
        break;
    case SDL_WINDOW_BORDERLESS:
        SDL_snprintfcat(text, maxlen, "BORDERLESS");
        break;
    case SDL_WINDOW_RESIZABLE:
        SDL_snprintfcat(text, maxlen, "RESIZABLE");
        break;
    case SDL_WINDOW_MINIMIZED:
        SDL_snprintfcat(text, maxlen, "MINIMIZED");
        break;
    case SDL_WINDOW_MAXIMIZED:
        SDL_snprintfcat(text, maxlen, "MAXIMIZED");
        break;
    case SDL_WINDOW_MOUSE_GRABBED:
        SDL_snprintfcat(text, maxlen, "MOUSE_GRABBED");
        break;
    case SDL_WINDOW_INPUT_FOCUS:
        SDL_snprintfcat(text, maxlen, "INPUT_FOCUS");
        break;
    case SDL_WINDOW_MOUSE_FOCUS:
        SDL_snprintfcat(text, maxlen, "MOUSE_FOCUS");
        break;
    case SDL_WINDOW_FULLSCREEN_DESKTOP:
        SDL_snprintfcat(text, maxlen, "FULLSCREEN_DESKTOP");
        break;
    case SDL_WINDOW_FOREIGN:
        SDL_snprintfcat(text, maxlen, "FOREIGN");
        break;
    case SDL_WINDOW_ALLOW_HIGHDPI:
        SDL_snprintfcat(text, maxlen, "ALLOW_HIGHDPI");
        break;
    case SDL_WINDOW_MOUSE_CAPTURE:
        SDL_snprintfcat(text, maxlen, "MOUSE_CAPTURE");
        break;
    case SDL_WINDOW_ALWAYS_ON_TOP:
        SDL_snprintfcat(text, maxlen, "ALWAYS_ON_TOP");
        break;
    case SDL_WINDOW_SKIP_TASKBAR:
        SDL_snprintfcat(text, maxlen, "SKIP_TASKBAR");
        break;
    case SDL_WINDOW_UTILITY:
        SDL_snprintfcat(text, maxlen, "UTILITY");
        break;
    case SDL_WINDOW_TOOLTIP:
        SDL_snprintfcat(text, maxlen, "TOOLTIP");
        break;
    case SDL_WINDOW_POPUP_MENU:
        SDL_snprintfcat(text, maxlen, "POPUP_MENU");
        break;
    case SDL_WINDOW_KEYBOARD_GRABBED:
        SDL_snprintfcat(text, maxlen, "KEYBOARD_GRABBED");
        break;
    case SDL_WINDOW_VULKAN:
        SDL_snprintfcat(text, maxlen, "VULKAN");
        break;
    case SDL_WINDOW_METAL:
        SDL_snprintfcat(text, maxlen, "METAL");
        break;
    default:
        SDL_snprintfcat(text, maxlen, "0x%8.8x", flag);
        break;
    }
}

static void SDLTest_PrintWindowFlags(char *text, size_t maxlen, Uint32 flags)
{
    const Uint32 window_flags[] = {
        SDL_WINDOW_FULLSCREEN,
        SDL_WINDOW_OPENGL,
        SDL_WINDOW_SHOWN,
        SDL_WINDOW_HIDDEN,
        SDL_WINDOW_BORDERLESS,
        SDL_WINDOW_RESIZABLE,
        SDL_WINDOW_MINIMIZED,
        SDL_WINDOW_MAXIMIZED,
        SDL_WINDOW_MOUSE_GRABBED,
        SDL_WINDOW_INPUT_FOCUS,
        SDL_WINDOW_MOUSE_FOCUS,
        SDL_WINDOW_FULLSCREEN_DESKTOP,
        SDL_WINDOW_FOREIGN,
        SDL_WINDOW_ALLOW_HIGHDPI,
        SDL_WINDOW_MOUSE_CAPTURE,
        SDL_WINDOW_ALWAYS_ON_TOP,
        SDL_WINDOW_SKIP_TASKBAR,
        SDL_WINDOW_UTILITY,
        SDL_WINDOW_TOOLTIP,
        SDL_WINDOW_POPUP_MENU,
        SDL_WINDOW_KEYBOARD_GRABBED,
        SDL_WINDOW_VULKAN,
        SDL_WINDOW_METAL
    };

    int i;
    int count = 0;
    for (i = 0; i < (sizeof(window_flags) / sizeof(window_flags[0])); ++i) {
        const Uint32 flag = window_flags[i];
        if ((flags & flag) == flag) {
            if (count > 0) {
                SDL_snprintfcat(text, maxlen, " | ");
            }
            SDLTest_PrintWindowFlag(text, maxlen, flag);
            ++count;
        }
    }
}

static void SDLTest_PrintModStateFlag(char *text, size_t maxlen, SDL_Keymod flag)
{
    switch (flag) {
    case KMOD_LSHIFT:
        SDL_snprintfcat(text, maxlen, "LSHIFT");
        break;
    case KMOD_RSHIFT:
        SDL_snprintfcat(text, maxlen, "RSHIFT");
        break;
    case KMOD_LCTRL:
        SDL_snprintfcat(text, maxlen, "LCTRL");
        break;
    case KMOD_RCTRL:
        SDL_snprintfcat(text, maxlen, "RCTRL");
        break;
    case KMOD_LALT:
        SDL_snprintfcat(text, maxlen, "LALT");
        break;
    case KMOD_RALT:
        SDL_snprintfcat(text, maxlen, "RALT");
        break;
    case KMOD_LGUI:
        SDL_snprintfcat(text, maxlen, "LGUI");
        break;
    case KMOD_RGUI:
        SDL_snprintfcat(text, maxlen, "RGUI");
        break;
    case KMOD_NUM:
        SDL_snprintfcat(text, maxlen, "NUM");
        break;
    case KMOD_CAPS:
        SDL_snprintfcat(text, maxlen, "CAPS");
        break;
    case KMOD_MODE:
        SDL_snprintfcat(text, maxlen, "MODE");
        break;
    case KMOD_SCROLL:
        SDL_snprintfcat(text, maxlen, "SCROLL");
        break;
    default:
        SDL_snprintfcat(text, maxlen, "0x%8.8x", (unsigned int) flag);
        break;
    }
}

static void SDLTest_PrintModState(char *text, size_t maxlen, SDL_Keymod keymod)
{
    const SDL_Keymod kmod_flags[] = {
        KMOD_LSHIFT,
        KMOD_RSHIFT,
        KMOD_LCTRL,
        KMOD_RCTRL,
        KMOD_LALT,
        KMOD_RALT,
        KMOD_LGUI,
        KMOD_RGUI,
        KMOD_NUM,
        KMOD_CAPS,
        KMOD_MODE,
        KMOD_SCROLL
    };

    int i;
    int count = 0;
    for (i = 0; i < SDL_arraysize(kmod_flags); ++i) {
        const SDL_Keymod flag = kmod_flags[i];
        if ((keymod & flag) == flag) {
            if (count > 0) {
                SDL_snprintfcat(text, maxlen, " | ");
            }
            SDLTest_PrintModStateFlag(text, maxlen, flag);
            ++count;
        }
    }
}

static void SDLTest_PrintButtonMask(char *text, size_t maxlen, Uint32 flags)
{
    int i;
    int count = 0;
    for (i = 1; i <= 32; ++i) {
        const Uint32 flag = SDL_BUTTON(i);
        if ((flags & flag) == flag) {
            if (count > 0) {
                SDL_snprintfcat(text, maxlen, " | ");
            }
            SDL_snprintfcat(text, maxlen, "SDL_BUTTON(%d)", i);
            ++count;
        }
    }
}

static void SDLTest_PrintRendererFlag(char *text, size_t maxlen, Uint32 flag)
{
    switch (flag) {
    case SDL_RENDERER_SOFTWARE:
        SDL_snprintfcat(text, maxlen, "Software");
        break;
    case SDL_RENDERER_ACCELERATED:
        SDL_snprintfcat(text, maxlen, "Accelerated");
        break;
    case SDL_RENDERER_PRESENTVSYNC:
        SDL_snprintfcat(text, maxlen, "PresentVSync");
        break;
    case SDL_RENDERER_TARGETTEXTURE:
        SDL_snprintfcat(text, maxlen, "TargetTexturesSupported");
        break;
    default:
        SDL_snprintfcat(text, maxlen, "0x%8.8x", flag);
        break;
    }
}

static void SDLTest_PrintPixelFormat(char *text, size_t maxlen, Uint32 format)
{
    switch (format) {
    case SDL_PIXELFORMAT_UNKNOWN:
        SDL_snprintfcat(text, maxlen, "Unknown");
        break;
    case SDL_PIXELFORMAT_INDEX1LSB:
        SDL_snprintfcat(text, maxlen, "Index1LSB");
        break;
    case SDL_PIXELFORMAT_INDEX1MSB:
        SDL_snprintfcat(text, maxlen, "Index1MSB");
        break;
    case SDL_PIXELFORMAT_INDEX2LSB:
        SDL_snprintfcat(text, maxlen, "Index2LSB");
        break;
    case SDL_PIXELFORMAT_INDEX2MSB:
        SDL_snprintfcat(text, maxlen, "Index2MSB");
        break;
    case SDL_PIXELFORMAT_INDEX4LSB:
        SDL_snprintfcat(text, maxlen, "Index4LSB");
        break;
    case SDL_PIXELFORMAT_INDEX4MSB:
        SDL_snprintfcat(text, maxlen, "Index4MSB");
        break;
    case SDL_PIXELFORMAT_INDEX8:
        SDL_snprintfcat(text, maxlen, "Index8");
        break;
    case SDL_PIXELFORMAT_RGB332:
        SDL_snprintfcat(text, maxlen, "RGB332");
        break;
    case SDL_PIXELFORMAT_RGB444:
        SDL_snprintfcat(text, maxlen, "RGB444");
        break;
    case SDL_PIXELFORMAT_BGR444:
        SDL_snprintfcat(text, maxlen, "BGR444");
        break;
    case SDL_PIXELFORMAT_RGB555:
        SDL_snprintfcat(text, maxlen, "RGB555");
        break;
    case SDL_PIXELFORMAT_BGR555:
        SDL_snprintfcat(text, maxlen, "BGR555");
        break;
    case SDL_PIXELFORMAT_ARGB4444:
        SDL_snprintfcat(text, maxlen, "ARGB4444");
        break;
    case SDL_PIXELFORMAT_ABGR4444:
        SDL_snprintfcat(text, maxlen, "ABGR4444");
        break;
    case SDL_PIXELFORMAT_ARGB1555:
        SDL_snprintfcat(text, maxlen, "ARGB1555");
        break;
    case SDL_PIXELFORMAT_ABGR1555:
        SDL_snprintfcat(text, maxlen, "ABGR1555");
        break;
    case SDL_PIXELFORMAT_RGB565:
        SDL_snprintfcat(text, maxlen, "RGB565");
        break;
    case SDL_PIXELFORMAT_BGR565:
        SDL_snprintfcat(text, maxlen, "BGR565");
        break;
    case SDL_PIXELFORMAT_RGB24:
        SDL_snprintfcat(text, maxlen, "RGB24");
        break;
    case SDL_PIXELFORMAT_BGR24:
        SDL_snprintfcat(text, maxlen, "BGR24");
        break;
    case SDL_PIXELFORMAT_RGB888:
        SDL_snprintfcat(text, maxlen, "RGB888");
        break;
    case SDL_PIXELFORMAT_BGR888:
        SDL_snprintfcat(text, maxlen, "BGR888");
        break;
    case SDL_PIXELFORMAT_ARGB8888:
        SDL_snprintfcat(text, maxlen, "ARGB8888");
        break;
    case SDL_PIXELFORMAT_RGBA8888:
        SDL_snprintfcat(text, maxlen, "RGBA8888");
        break;
    case SDL_PIXELFORMAT_ABGR8888:
        SDL_snprintfcat(text, maxlen, "ABGR8888");
        break;
    case SDL_PIXELFORMAT_BGRA8888:
        SDL_snprintfcat(text, maxlen, "BGRA8888");
        break;
    case SDL_PIXELFORMAT_ARGB2101010:
        SDL_snprintfcat(text, maxlen, "ARGB2101010");
        break;
    case SDL_PIXELFORMAT_YV12:
        SDL_snprintfcat(text, maxlen, "YV12");
        break;
    case SDL_PIXELFORMAT_IYUV:
        SDL_snprintfcat(text, maxlen, "IYUV");
        break;
    case SDL_PIXELFORMAT_YUY2:
        SDL_snprintfcat(text, maxlen, "YUY2");
        break;
    case SDL_PIXELFORMAT_UYVY:
        SDL_snprintfcat(text, maxlen, "UYVY");
        break;
    case SDL_PIXELFORMAT_YVYU:
        SDL_snprintfcat(text, maxlen, "YVYU");
        break;
    case SDL_PIXELFORMAT_NV12:
        SDL_snprintfcat(text, maxlen, "NV12");
        break;
    case SDL_PIXELFORMAT_NV21:
        SDL_snprintfcat(text, maxlen, "NV21");
        break;
    default:
        SDL_snprintfcat(text, maxlen, "0x%8.8x", format);
        break;
    }
}

static void SDLTest_PrintRenderer(SDL_RendererInfo *info)
{
    int i, count;
    char text[1024];

    SDL_Log("  Renderer %s:\n", info->name);

    (void)SDL_snprintf(text, sizeof(text), "    Flags: 0x%8.8" SDL_PRIX32, info->flags);
    SDL_snprintfcat(text, sizeof(text), " (");
    count = 0;
    for (i = 0; i < 8 * sizeof(info->flags); ++i) {
        Uint32 flag = (1 << i);
        if (info->flags & flag) {
            if (count > 0) {
                SDL_snprintfcat(text, sizeof(text), " | ");
            }
            SDLTest_PrintRendererFlag(text, sizeof(text), flag);
            ++count;
        }
    }
    SDL_snprintfcat(text, sizeof(text), ")");
    SDL_Log("%s\n", text);

    (void)SDL_snprintf(text, sizeof(text), "    Texture formats (%" SDL_PRIu32 "): ", info->num_texture_formats);
    for (i = 0; i < (int)info->num_texture_formats; ++i) {
        if (i > 0) {
            SDL_snprintfcat(text, sizeof(text), ", ");
        }
        SDLTest_PrintPixelFormat(text, sizeof(text), info->texture_formats[i]);
    }
    SDL_Log("%s\n", text);

    if (info->max_texture_width || info->max_texture_height) {
        SDL_Log("    Max Texture Size: %dx%d\n",
                info->max_texture_width, info->max_texture_height);
    }
}

static SDL_Surface *SDLTest_LoadIcon(const char *file)
{
    SDL_Surface *icon;

    /* Load the icon surface */
    icon = SDL_LoadBMP(file);
    if (!icon) {
        SDL_Log("Couldn't load %s: %s\n", file, SDL_GetError());
        return NULL;
    }

    if (icon->format->palette) {
        /* Set the colorkey */
        SDL_SetColorKey(icon, 1, *((Uint8 *)icon->pixels));
    }

    return icon;
}

static SDL_HitTestResult SDLCALL SDLTest_ExampleHitTestCallback(SDL_Window *win, const SDL_Point *area, void *data)
{
    int w, h;
    const int RESIZE_BORDER = 8;
    const int DRAGGABLE_TITLE = 32;

    /*SDL_Log("Hit test point %d,%d\n", area->x, area->y);*/

    SDL_GetWindowSize(win, &w, &h);

    if (area->x < RESIZE_BORDER) {
        if (area->y < RESIZE_BORDER) {
            SDL_Log("SDL_HITTEST_RESIZE_TOPLEFT\n");
            return SDL_HITTEST_RESIZE_TOPLEFT;
        } else if (area->y >= (h - RESIZE_BORDER)) {
            SDL_Log("SDL_HITTEST_RESIZE_BOTTOMLEFT\n");
            return SDL_HITTEST_RESIZE_BOTTOMLEFT;
        } else {
            SDL_Log("SDL_HITTEST_RESIZE_LEFT\n");
            return SDL_HITTEST_RESIZE_LEFT;
        }
    } else if (area->x >= (w - RESIZE_BORDER)) {
        if (area->y < RESIZE_BORDER) {
            SDL_Log("SDL_HITTEST_RESIZE_TOPRIGHT\n");
            return SDL_HITTEST_RESIZE_TOPRIGHT;
        } else if (area->y >= (h - RESIZE_BORDER)) {
            SDL_Log("SDL_HITTEST_RESIZE_BOTTOMRIGHT\n");
            return SDL_HITTEST_RESIZE_BOTTOMRIGHT;
        } else {
            SDL_Log("SDL_HITTEST_RESIZE_RIGHT\n");
            return SDL_HITTEST_RESIZE_RIGHT;
        }
    } else if (area->y >= (h - RESIZE_BORDER)) {
        SDL_Log("SDL_HITTEST_RESIZE_BOTTOM\n");
        return SDL_HITTEST_RESIZE_BOTTOM;
    } else if (area->y < RESIZE_BORDER) {
        SDL_Log("SDL_HITTEST_RESIZE_TOP\n");
        return SDL_HITTEST_RESIZE_TOP;
    } else if (area->y < DRAGGABLE_TITLE) {
        SDL_Log("SDL_HITTEST_DRAGGABLE\n");
        return SDL_HITTEST_DRAGGABLE;
    }
    return SDL_HITTEST_NORMAL;
}

SDL_bool SDLTest_CommonInit(SDLTest_CommonState *state)
{
    int i, j, m, n, w, h;
    SDL_DisplayMode fullscreen_mode;
    char text[1024];

    if (state->flags & SDL_INIT_VIDEO) {
        if (state->verbose & VERBOSE_VIDEO) {
            n = SDL_GetNumVideoDrivers();
            if (n == 0) {
                SDL_Log("No built-in video drivers\n");
            } else {
                (void)SDL_snprintf(text, sizeof(text), "Built-in video drivers:");
                for (i = 0; i < n; ++i) {
                    if (i > 0) {
                        SDL_snprintfcat(text, sizeof(text), ",");
                    }
                    SDL_snprintfcat(text, sizeof(text), " %s", SDL_GetVideoDriver(i));
                }
                SDL_Log("%s\n", text);
            }
        }
        if (SDL_VideoInit(state->videodriver) < 0) {
            SDL_Log("Couldn't initialize video driver: %s\n",
                    SDL_GetError());
            return SDL_FALSE;
        }
        if (state->verbose & VERBOSE_VIDEO) {
            SDL_Log("Video driver: %s\n",
                    SDL_GetCurrentVideoDriver());
        }

        /* Upload GL settings */
        SDL_GL_SetAttribute(SDL_GL_RED_SIZE, state->gl_red_size);
        SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, state->gl_green_size);
        SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, state->gl_blue_size);
        SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, state->gl_alpha_size);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, state->gl_double_buffer);
        SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, state->gl_buffer_size);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, state->gl_depth_size);
        SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, state->gl_stencil_size);
        SDL_GL_SetAttribute(SDL_GL_ACCUM_RED_SIZE, state->gl_accum_red_size);
        SDL_GL_SetAttribute(SDL_GL_ACCUM_GREEN_SIZE, state->gl_accum_green_size);
        SDL_GL_SetAttribute(SDL_GL_ACCUM_BLUE_SIZE, state->gl_accum_blue_size);
        SDL_GL_SetAttribute(SDL_GL_ACCUM_ALPHA_SIZE, state->gl_accum_alpha_size);
        SDL_GL_SetAttribute(SDL_GL_STEREO, state->gl_stereo);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, state->gl_multisamplebuffers);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, state->gl_multisamplesamples);
        if (state->gl_accelerated >= 0) {
            SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL,
                                state->gl_accelerated);
        }
        SDL_GL_SetAttribute(SDL_GL_RETAINED_BACKING, state->gl_retained_backing);
        if (state->gl_major_version) {
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, state->gl_major_version);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, state->gl_minor_version);
        }
        if (state->gl_debug) {
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
        }
        if (state->gl_profile_mask) {
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, state->gl_profile_mask);
        }

        if (state->verbose & VERBOSE_MODES) {
            SDL_Rect bounds, usablebounds;
            float hdpi = 0;
            float vdpi = 0;
            SDL_DisplayMode mode;
            int bpp;
            Uint32 Rmask, Gmask, Bmask, Amask;
#ifdef SDL_VIDEO_DRIVER_WINDOWS
            int adapterIndex = 0;
            int outputIndex = 0;
#endif
            n = SDL_GetNumVideoDisplays();
            SDL_Log("Number of displays: %d\n", n);
            for (i = 0; i < n; ++i) {
                SDL_Log("Display %d: %s\n", i, SDL_GetDisplayName(i));

                SDL_zero(bounds);
                SDL_GetDisplayBounds(i, &bounds);

                SDL_zero(usablebounds);
                SDL_GetDisplayUsableBounds(i, &usablebounds);

                SDL_GetDisplayDPI(i, NULL, &hdpi, &vdpi);

                SDL_Log("Bounds: %dx%d at %d,%d\n", bounds.w, bounds.h, bounds.x, bounds.y);
                SDL_Log("Usable bounds: %dx%d at %d,%d\n", usablebounds.w, usablebounds.h, usablebounds.x, usablebounds.y);
                SDL_Log("DPI: %fx%f\n", hdpi, vdpi);

                SDL_GetDesktopDisplayMode(i, &mode);
                SDL_PixelFormatEnumToMasks(mode.format, &bpp, &Rmask, &Gmask,
                                           &Bmask, &Amask);
                SDL_Log("  Current mode: %dx%d@%dHz, %d bits-per-pixel (%s)\n",
                        mode.w, mode.h, mode.refresh_rate, bpp,
                        SDL_GetPixelFormatName(mode.format));
                if (Rmask || Gmask || Bmask) {
                    SDL_Log("      Red Mask   = 0x%.8" SDL_PRIx32 "\n", Rmask);
                    SDL_Log("      Green Mask = 0x%.8" SDL_PRIx32 "\n", Gmask);
                    SDL_Log("      Blue Mask  = 0x%.8" SDL_PRIx32 "\n", Bmask);
                    if (Amask) {
                        SDL_Log("      Alpha Mask = 0x%.8" SDL_PRIx32 "\n", Amask);
                    }
                }

                /* Print available fullscreen video modes */
                m = SDL_GetNumDisplayModes(i);
                if (m == 0) {
                    SDL_Log("No available fullscreen video modes\n");
                } else {
                    SDL_Log("  Fullscreen video modes:\n");
                    for (j = 0; j < m; ++j) {
                        SDL_GetDisplayMode(i, j, &mode);
                        SDL_PixelFormatEnumToMasks(mode.format, &bpp, &Rmask,
                                                   &Gmask, &Bmask, &Amask);
                        SDL_Log("    Mode %d: %dx%d@%dHz, %d bits-per-pixel (%s)\n",
                                j, mode.w, mode.h, mode.refresh_rate, bpp,
                                SDL_GetPixelFormatName(mode.format));
                        if (Rmask || Gmask || Bmask) {
                            SDL_Log("        Red Mask   = 0x%.8" SDL_PRIx32 "\n",
                                    Rmask);
                            SDL_Log("        Green Mask = 0x%.8" SDL_PRIx32 "\n",
                                    Gmask);
                            SDL_Log("        Blue Mask  = 0x%.8" SDL_PRIx32 "\n",
                                    Bmask);
                            if (Amask) {
                                SDL_Log("        Alpha Mask = 0x%.8" SDL_PRIx32 "\n", Amask);
                            }
                        }
                    }
                }

#if defined(SDL_VIDEO_DRIVER_WINDOWS) && !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
                /* Print the D3D9 adapter index */
                adapterIndex = SDL_Direct3D9GetAdapterIndex(i);
                SDL_Log("D3D9 Adapter Index: %d", adapterIndex);

                /* Print the DXGI adapter and output indices */
                SDL_DXGIGetOutputInfo(i, &adapterIndex, &outputIndex);
                SDL_Log("DXGI Adapter Index: %d  Output Index: %d", adapterIndex, outputIndex);
#endif
            }
        }

        if (state->verbose & VERBOSE_RENDER) {
            SDL_RendererInfo info;

            n = SDL_GetNumRenderDrivers();
            if (n == 0) {
                SDL_Log("No built-in render drivers\n");
            } else {
                SDL_Log("Built-in render drivers:\n");
                for (i = 0; i < n; ++i) {
                    SDL_GetRenderDriverInfo(i, &info);
                    SDLTest_PrintRenderer(&info);
                }
            }
        }

        SDL_zero(fullscreen_mode);
        switch (state->depth) {
        case 8:
            fullscreen_mode.format = SDL_PIXELFORMAT_INDEX8;
            break;
        case 15:
            fullscreen_mode.format = SDL_PIXELFORMAT_RGB555;
            break;
        case 16:
            fullscreen_mode.format = SDL_PIXELFORMAT_RGB565;
            break;
        case 24:
            fullscreen_mode.format = SDL_PIXELFORMAT_RGB24;
            break;
        default:
            fullscreen_mode.format = SDL_PIXELFORMAT_RGB888;
            break;
        }
        fullscreen_mode.refresh_rate = state->refresh_rate;

        state->windows =
            (SDL_Window **)SDL_calloc(state->num_windows,
                                      sizeof(*state->windows));
        state->renderers =
            (SDL_Renderer **)SDL_calloc(state->num_windows,
                                        sizeof(*state->renderers));
        state->targets =
            (SDL_Texture **)SDL_calloc(state->num_windows,
                                       sizeof(*state->targets));
        if (!state->windows || !state->renderers) {
            SDL_Log("Out of memory!\n");
            return SDL_FALSE;
        }
        for (i = 0; i < state->num_windows; ++i) {
            char title[1024];
            SDL_Rect r;

            r.x = state->window_x;
            r.y = state->window_y;
            r.w = state->window_w;
            r.h = state->window_h;

            /* !!! FIXME: hack to make --usable-bounds work for now. */
            if ((r.x == -1) && (r.y == -1) && (r.w == -1) && (r.h == -1)) {
                SDL_GetDisplayUsableBounds(state->display, &r);
            }

            if (state->num_windows > 1) {
                (void)SDL_snprintf(title, SDL_arraysize(title), "%s %d",
                                   state->window_title, i + 1);
            } else {
                SDL_strlcpy(title, state->window_title, SDL_arraysize(title));
            }
            state->windows[i] =
                SDL_CreateWindow(title, r.x, r.y, r.w, r.h, state->window_flags);
            if (!state->windows[i]) {
                SDL_Log("Couldn't create window: %s\n",
                        SDL_GetError());
                return SDL_FALSE;
            }
            if (state->window_minW || state->window_minH) {
                SDL_SetWindowMinimumSize(state->windows[i], state->window_minW, state->window_minH);
            }
            if (state->window_maxW || state->window_maxH) {
                SDL_SetWindowMaximumSize(state->windows[i], state->window_maxW, state->window_maxH);
            }
            SDL_GetWindowSize(state->windows[i], &w, &h);
            if (!(state->window_flags & SDL_WINDOW_RESIZABLE) &&
                (w != state->window_w || h != state->window_h)) {
                SDL_Log("Window requested size %dx%d, got %dx%d\n", state->window_w, state->window_h, w, h);
                state->window_w = w;
                state->window_h = h;
            }
            fullscreen_mode.w = state->window_w;
            fullscreen_mode.h = state->window_h;
            if (SDL_SetWindowDisplayMode(state->windows[i], &fullscreen_mode) < 0) {
                SDL_Log("Can't set up fullscreen display mode: %s\n",
                        SDL_GetError());
                return SDL_FALSE;
            }

            /* Add resize/drag areas for windows that are borderless and resizable */
            if ((state->window_flags & (SDL_WINDOW_RESIZABLE | SDL_WINDOW_BORDERLESS)) ==
                (SDL_WINDOW_RESIZABLE | SDL_WINDOW_BORDERLESS)) {
                SDL_SetWindowHitTest(state->windows[i], SDLTest_ExampleHitTestCallback, NULL);
            }

            if (state->window_icon) {
                SDL_Surface *icon = SDLTest_LoadIcon(state->window_icon);
                if (icon) {
                    SDL_SetWindowIcon(state->windows[i], icon);
                    SDL_FreeSurface(icon);
                }
            }

            SDL_ShowWindow(state->windows[i]);

            if (!SDL_RectEmpty(&state->confine)) {
                SDL_SetWindowMouseRect(state->windows[i], &state->confine);
            }

            if (!state->skip_renderer && (state->renderdriver || !(state->window_flags & (SDL_WINDOW_OPENGL | SDL_WINDOW_VULKAN | SDL_WINDOW_METAL)))) {
                m = -1;
                if (state->renderdriver) {
                    SDL_RendererInfo info;
                    n = SDL_GetNumRenderDrivers();
                    for (j = 0; j < n; ++j) {
                        SDL_GetRenderDriverInfo(j, &info);
                        if (SDL_strcasecmp(info.name, state->renderdriver) == 0) {
                            m = j;
                            break;
                        }
                    }
                    if (m == -1) {
                        SDL_Log("Couldn't find render driver named %s",
                                state->renderdriver);
                        return SDL_FALSE;
                    }
                }
                state->renderers[i] = SDL_CreateRenderer(state->windows[i],
                                                         m, state->render_flags);
                if (!state->renderers[i]) {
                    SDL_Log("Couldn't create renderer: %s\n",
                            SDL_GetError());
                    return SDL_FALSE;
                }
                if (state->logical_w && state->logical_h) {
                    SDL_RenderSetLogicalSize(state->renderers[i], state->logical_w, state->logical_h);
                } else if (state->scale != 0.) {
                    SDL_RenderSetScale(state->renderers[i], state->scale, state->scale);
                }
                if (state->verbose & VERBOSE_RENDER) {
                    SDL_RendererInfo info;

                    SDL_Log("Current renderer:\n");
                    SDL_GetRendererInfo(state->renderers[i], &info);
                    SDLTest_PrintRenderer(&info);
                }
            }
        }
    }

    if (state->flags & SDL_INIT_AUDIO) {
        if (state->verbose & VERBOSE_AUDIO) {
            n = SDL_GetNumAudioDrivers();
            if (n == 0) {
                SDL_Log("No built-in audio drivers\n");
            } else {
                (void)SDL_snprintf(text, sizeof(text), "Built-in audio drivers:");
                for (i = 0; i < n; ++i) {
                    if (i > 0) {
                        SDL_snprintfcat(text, sizeof(text), ",");
                    }
                    SDL_snprintfcat(text, sizeof(text), " %s", SDL_GetAudioDriver(i));
                }
                SDL_Log("%s\n", text);
            }
        }
        if (SDL_AudioInit(state->audiodriver) < 0) {
            SDL_Log("Couldn't initialize audio driver: %s\n",
                    SDL_GetError());
            return SDL_FALSE;
        }
        if (state->verbose & VERBOSE_AUDIO) {
            SDL_Log("Audio driver: %s\n",
                    SDL_GetCurrentAudioDriver());
        }

        if (SDL_OpenAudio(&state->audiospec, NULL) < 0) {
            SDL_Log("Couldn't open audio: %s\n", SDL_GetError());
            return SDL_FALSE;
        }
    }

    return SDL_TRUE;
}

static const char *DisplayOrientationName(int orientation)
{
    switch (orientation) {
#define CASE(X)               \
    case SDL_ORIENTATION_##X: \
        return #X
        CASE(UNKNOWN);
        CASE(LANDSCAPE);
        CASE(LANDSCAPE_FLIPPED);
        CASE(PORTRAIT);
        CASE(PORTRAIT_FLIPPED);
#undef CASE
    default:
        return "???";
    }
}

static const char *ControllerAxisName(const SDL_GameControllerAxis axis)
{
    switch (axis) {
#define AXIS_CASE(ax)              \
    case SDL_CONTROLLER_AXIS_##ax: \
        return #ax
        AXIS_CASE(INVALID);
        AXIS_CASE(LEFTX);
        AXIS_CASE(LEFTY);
        AXIS_CASE(RIGHTX);
        AXIS_CASE(RIGHTY);
        AXIS_CASE(TRIGGERLEFT);
        AXIS_CASE(TRIGGERRIGHT);
#undef AXIS_CASE
    default:
        return "???";
    }
}

static const char *ControllerButtonName(const SDL_GameControllerButton button)
{
    switch (button) {
#define BUTTON_CASE(btn)              \
    case SDL_CONTROLLER_BUTTON_##btn: \
        return #btn
        BUTTON_CASE(INVALID);
        BUTTON_CASE(A);
        BUTTON_CASE(B);
        BUTTON_CASE(X);
        BUTTON_CASE(Y);
        BUTTON_CASE(BACK);
        BUTTON_CASE(GUIDE);
        BUTTON_CASE(START);
        BUTTON_CASE(LEFTSTICK);
        BUTTON_CASE(RIGHTSTICK);
        BUTTON_CASE(LEFTSHOULDER);
        BUTTON_CASE(RIGHTSHOULDER);
        BUTTON_CASE(DPAD_UP);
        BUTTON_CASE(DPAD_DOWN);
        BUTTON_CASE(DPAD_LEFT);
        BUTTON_CASE(DPAD_RIGHT);
#undef BUTTON_CASE
    default:
        return "???";
    }
}

static void SDLTest_PrintEvent(SDL_Event *event)
{
    switch (event->type) {
    case SDL_DISPLAYEVENT:
        switch (event->display.event) {
        case SDL_DISPLAYEVENT_CONNECTED:
            SDL_Log("SDL EVENT: Display %" SDL_PRIu32 " connected",
                    event->display.display);
            break;
        case SDL_DISPLAYEVENT_MOVED:
            SDL_Log("SDL EVENT: Display %" SDL_PRIu32 " changed position",
                    event->display.display);
            break;
        case SDL_DISPLAYEVENT_ORIENTATION:
            SDL_Log("SDL EVENT: Display %" SDL_PRIu32 " changed orientation to %s",
                    event->display.display, DisplayOrientationName(event->display.data1));
            break;
        case SDL_DISPLAYEVENT_DISCONNECTED:
            SDL_Log("SDL EVENT: Display %" SDL_PRIu32 " disconnected",
                    event->display.display);
            break;
        default:
            SDL_Log("SDL EVENT: Display %" SDL_PRIu32 " got unknown event 0x%4.4x",
                    event->display.display, event->display.event);
            break;
        }
        break;
    case SDL_WINDOWEVENT:
        switch (event->window.event) {
        case SDL_WINDOWEVENT_SHOWN:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " shown", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_HIDDEN:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " hidden", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_EXPOSED:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " exposed", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_MOVED:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " moved to %" SDL_PRIs32 ",%" SDL_PRIs32,
                    event->window.windowID, event->window.data1, event->window.data2);
            break;
        case SDL_WINDOWEVENT_RESIZED:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " resized to %" SDL_PRIs32 "x%" SDL_PRIs32,
                    event->window.windowID, event->window.data1, event->window.data2);
            break;
        case SDL_WINDOWEVENT_SIZE_CHANGED:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " changed size to %" SDL_PRIs32 "x%" SDL_PRIs32,
                    event->window.windowID, event->window.data1, event->window.data2);
            break;
        case SDL_WINDOWEVENT_MINIMIZED:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " minimized", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_MAXIMIZED:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " maximized", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_RESTORED:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " restored", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_ENTER:
            SDL_Log("SDL EVENT: Mouse entered window %" SDL_PRIu32, event->window.windowID);
            break;
        case SDL_WINDOWEVENT_LEAVE:
            SDL_Log("SDL EVENT: Mouse left window %" SDL_PRIu32, event->window.windowID);
            break;
        case SDL_WINDOWEVENT_FOCUS_GAINED:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " gained keyboard focus",
                    event->window.windowID);
            break;
        case SDL_WINDOWEVENT_FOCUS_LOST:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " lost keyboard focus",
                    event->window.windowID);
            break;
        case SDL_WINDOWEVENT_CLOSE:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " closed", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_TAKE_FOCUS:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " take focus", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_HIT_TEST:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " hit test", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_ICCPROF_CHANGED:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " ICC profile changed", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_DISPLAY_CHANGED:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " display changed to %" SDL_PRIs32, event->window.windowID, event->window.data1);
            break;
        default:
            SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " got unknown event 0x%4.4x",
                    event->window.windowID, event->window.event);
            break;
        }
        break;
    case SDL_KEYDOWN:
    case SDL_KEYUP: {
        char modstr[64];
        if (event->key.keysym.mod) {
            modstr[0] = '\0';
            SDLTest_PrintModState(modstr, sizeof (modstr), event->key.keysym.mod);
        } else {
            SDL_strlcpy(modstr, "NONE", sizeof (modstr));
        }
        SDL_Log("SDL EVENT: Keyboard: key %s in window %" SDL_PRIu32 ": scancode 0x%08X = %s, keycode 0x%08" SDL_PRIX32 " = %s, mods = %s",
                (event->type == SDL_KEYDOWN) ? "pressed" : "released",
                event->key.windowID,
                event->key.keysym.scancode,
                SDL_GetScancodeName(event->key.keysym.scancode),
                event->key.keysym.sym, SDL_GetKeyName(event->key.keysym.sym),
                modstr);
        break;
    }
    case SDL_TEXTEDITING:
        SDL_Log("SDL EVENT: Keyboard: text editing \"%s\" in window %" SDL_PRIu32,
                event->edit.text, event->edit.windowID);
        break;
    case SDL_TEXTINPUT:
        SDL_Log("SDL EVENT: Keyboard: text input \"%s\" in window %" SDL_PRIu32,
                event->text.text, event->text.windowID);
        break;
    case SDL_KEYMAPCHANGED:
        SDL_Log("SDL EVENT: Keymap changed");
        break;
    case SDL_MOUSEMOTION:
        SDL_Log("SDL EVENT: Mouse: moved to %" SDL_PRIs32 ",%" SDL_PRIs32 " (%" SDL_PRIs32 ",%" SDL_PRIs32 ") in window %" SDL_PRIu32,
                event->motion.x, event->motion.y,
                event->motion.xrel, event->motion.yrel,
                event->motion.windowID);
        break;
    case SDL_MOUSEBUTTONDOWN:
        SDL_Log("SDL EVENT: Mouse: button %d pressed at %" SDL_PRIs32 ",%" SDL_PRIs32 " with click count %d in window %" SDL_PRIu32,
                event->button.button, event->button.x, event->button.y, event->button.clicks,
                event->button.windowID);
        break;
    case SDL_MOUSEBUTTONUP:
        SDL_Log("SDL EVENT: Mouse: button %d released at %" SDL_PRIs32 ",%" SDL_PRIs32 " with click count %d in window %" SDL_PRIu32,
                event->button.button, event->button.x, event->button.y, event->button.clicks,
                event->button.windowID);
        break;
    case SDL_MOUSEWHEEL:
        SDL_Log("SDL EVENT: Mouse: wheel scrolled %" SDL_PRIs32 " in x and %" SDL_PRIs32 " in y (reversed: %" SDL_PRIu32 ") in window %" SDL_PRIu32,
                event->wheel.x, event->wheel.y, event->wheel.direction, event->wheel.windowID);
        break;
    case SDL_JOYDEVICEADDED:
        SDL_Log("SDL EVENT: Joystick index %" SDL_PRIs32 " attached",
                event->jdevice.which);
        break;
    case SDL_JOYDEVICEREMOVED:
        SDL_Log("SDL EVENT: Joystick %" SDL_PRIs32 " removed",
                event->jdevice.which);
        break;
    case SDL_JOYBALLMOTION:
        SDL_Log("SDL EVENT: Joystick %" SDL_PRIs32 ": ball %d moved by %d,%d",
                event->jball.which, event->jball.ball, event->jball.xrel,
                event->jball.yrel);
        break;
    case SDL_JOYHATMOTION:
    {
        const char *position = "UNKNOWN";
        switch (event->jhat.value) {
        case SDL_HAT_CENTERED:
            position = "CENTER";
            break;
        case SDL_HAT_UP:
            position = "UP";
            break;
        case SDL_HAT_RIGHTUP:
            position = "RIGHTUP";
            break;
        case SDL_HAT_RIGHT:
            position = "RIGHT";
            break;
        case SDL_HAT_RIGHTDOWN:
            position = "RIGHTDOWN";
            break;
        case SDL_HAT_DOWN:
            position = "DOWN";
            break;
        case SDL_HAT_LEFTDOWN:
            position = "LEFTDOWN";
            break;
        case SDL_HAT_LEFT:
            position = "LEFT";
            break;
        case SDL_HAT_LEFTUP:
            position = "LEFTUP";
            break;
        }
        SDL_Log("SDL EVENT: Joystick %" SDL_PRIs32 ": hat %d moved to %s",
                event->jhat.which, event->jhat.hat, position);
    } break;
    case SDL_JOYBUTTONDOWN:
        SDL_Log("SDL EVENT: Joystick %" SDL_PRIs32 ": button %d pressed",
                event->jbutton.which, event->jbutton.button);
        break;
    case SDL_JOYBUTTONUP:
        SDL_Log("SDL EVENT: Joystick %" SDL_PRIs32 ": button %d released",
                event->jbutton.which, event->jbutton.button);
        break;
    case SDL_CONTROLLERDEVICEADDED:
        SDL_Log("SDL EVENT: Controller index %" SDL_PRIs32 " attached",
                event->cdevice.which);
        break;
    case SDL_CONTROLLERDEVICEREMOVED:
        SDL_Log("SDL EVENT: Controller %" SDL_PRIs32 " removed",
                event->cdevice.which);
        break;
    case SDL_CONTROLLERAXISMOTION:
        SDL_Log("SDL EVENT: Controller %" SDL_PRIs32 " axis %d ('%s') value: %d",
                event->caxis.which,
                event->caxis.axis,
                ControllerAxisName((SDL_GameControllerAxis)event->caxis.axis),
                event->caxis.value);
        break;
    case SDL_CONTROLLERBUTTONDOWN:
        SDL_Log("SDL EVENT: Controller %" SDL_PRIs32 "button %d ('%s') down",
                event->cbutton.which, event->cbutton.button,
                ControllerButtonName((SDL_GameControllerButton)event->cbutton.button));
        break;
    case SDL_CONTROLLERBUTTONUP:
        SDL_Log("SDL EVENT: Controller %" SDL_PRIs32 " button %d ('%s') up",
                event->cbutton.which, event->cbutton.button,
                ControllerButtonName((SDL_GameControllerButton)event->cbutton.button));
        break;
    case SDL_CLIPBOARDUPDATE:
        SDL_Log("SDL EVENT: Clipboard updated");
        break;

    case SDL_FINGERMOTION:
        SDL_Log("SDL EVENT: Finger: motion touch=%ld, finger=%ld, x=%f, y=%f, dx=%f, dy=%f, pressure=%f",
                (long)event->tfinger.touchId,
                (long)event->tfinger.fingerId,
                event->tfinger.x, event->tfinger.y,
                event->tfinger.dx, event->tfinger.dy, event->tfinger.pressure);
        break;
    case SDL_FINGERDOWN:
    case SDL_FINGERUP:
        SDL_Log("SDL EVENT: Finger: %s touch=%ld, finger=%ld, x=%f, y=%f, dx=%f, dy=%f, pressure=%f",
                (event->type == SDL_FINGERDOWN) ? "down" : "up",
                (long)event->tfinger.touchId,
                (long)event->tfinger.fingerId,
                event->tfinger.x, event->tfinger.y,
                event->tfinger.dx, event->tfinger.dy, event->tfinger.pressure);
        break;
    case SDL_DOLLARGESTURE:
        SDL_Log("SDL_EVENT: Dollar gesture detect: %ld", (long)event->dgesture.gestureId);
        break;
    case SDL_DOLLARRECORD:
        SDL_Log("SDL_EVENT: Dollar gesture record: %ld", (long)event->dgesture.gestureId);
        break;
    case SDL_MULTIGESTURE:
        SDL_Log("SDL_EVENT: Multi gesture fingers: %d", event->mgesture.numFingers);
        break;

    case SDL_RENDER_DEVICE_RESET:
        SDL_Log("SDL EVENT: render device reset");
        break;
    case SDL_RENDER_TARGETS_RESET:
        SDL_Log("SDL EVENT: render targets reset");
        break;

    case SDL_APP_TERMINATING:
        SDL_Log("SDL EVENT: App terminating");
        break;
    case SDL_APP_LOWMEMORY:
        SDL_Log("SDL EVENT: App running low on memory");
        break;
    case SDL_APP_WILLENTERBACKGROUND:
        SDL_Log("SDL EVENT: App will enter the background");
        break;
    case SDL_APP_DIDENTERBACKGROUND:
        SDL_Log("SDL EVENT: App entered the background");
        break;
    case SDL_APP_WILLENTERFOREGROUND:
        SDL_Log("SDL EVENT: App will enter the foreground");
        break;
    case SDL_APP_DIDENTERFOREGROUND:
        SDL_Log("SDL EVENT: App entered the foreground");
        break;
    case SDL_DROPBEGIN:
        SDL_Log("SDL EVENT: Drag and drop beginning");
        break;
    case SDL_DROPFILE:
        SDL_Log("SDL EVENT: Drag and drop file: '%s'", event->drop.file);
        break;
    case SDL_DROPTEXT:
        SDL_Log("SDL EVENT: Drag and drop text: '%s'", event->drop.file);
        break;
    case SDL_DROPCOMPLETE:
        SDL_Log("SDL EVENT: Drag and drop ending");
        break;
    case SDL_QUIT:
        SDL_Log("SDL EVENT: Quit requested");
        break;
    case SDL_USEREVENT:
        SDL_Log("SDL EVENT: User event %" SDL_PRIs32, event->user.code);
        break;
    default:
        SDL_Log("Unknown event 0x%4.4" SDL_PRIu32, event->type);
        break;
    }
}

static void SDLTest_ScreenShot(SDL_Renderer *renderer)
{
    SDL_Rect viewport;
    SDL_Surface *surface;

    if (!renderer) {
        return;
    }

    SDL_RenderGetViewport(renderer, &viewport);
    surface = SDL_CreateRGBSurface(0, viewport.w, viewport.h, 24,
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                                   0x00FF0000, 0x0000FF00, 0x000000FF,
#else
                                   0x000000FF, 0x0000FF00, 0x00FF0000,
#endif
                                   0x00000000);
    if (!surface) {
        SDL_Log("Couldn't create surface: %s\n", SDL_GetError());
        return;
    }

    if (SDL_RenderReadPixels(renderer, NULL, surface->format->format,
                             surface->pixels, surface->pitch) < 0) {
        SDL_Log("Couldn't read screen: %s\n", SDL_GetError());
        SDL_free(surface);
        return;
    }

    if (SDL_SaveBMP(surface, "screenshot.bmp") < 0) {
        SDL_Log("Couldn't save screenshot.bmp: %s\n", SDL_GetError());
        SDL_free(surface);
        return;
    }
}

static void FullscreenTo(int index, int windowId)
{
    Uint32 flags;
    struct SDL_Rect rect = { 0, 0, 0, 0 };
    SDL_Window *window = SDL_GetWindowFromID(windowId);
    if (!window) {
        return;
    }

    SDL_GetDisplayBounds(index, &rect);

    flags = SDL_GetWindowFlags(window);
    if (flags & SDL_WINDOW_FULLSCREEN) {
        SDL_SetWindowFullscreen(window, 0);
        SDL_Delay(15);
    }

    SDL_SetWindowPosition(window, rect.x, rect.y);
    SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
}

void SDLTest_CommonEvent(SDLTest_CommonState *state, SDL_Event *event, int *done)
{
    int i;
    static SDL_MouseMotionEvent lastEvent;

    if (state->verbose & VERBOSE_EVENT) {
        if (((event->type != SDL_MOUSEMOTION) &&
             (event->type != SDL_FINGERMOTION)) ||
            ((state->verbose & VERBOSE_MOTION) != 0)) {
            SDLTest_PrintEvent(event);
        }
    }

    switch (event->type) {
    case SDL_WINDOWEVENT:
        switch (event->window.event) {
        case SDL_WINDOWEVENT_CLOSE:
        {
            SDL_Window *window = SDL_GetWindowFromID(event->window.windowID);
            if (window) {
                for (i = 0; i < state->num_windows; ++i) {
                    if (window == state->windows[i]) {
                        if (state->targets[i]) {
                            SDL_DestroyTexture(state->targets[i]);
                            state->targets[i] = NULL;
                        }
                        if (state->renderers[i]) {
                            SDL_DestroyRenderer(state->renderers[i]);
                            state->renderers[i] = NULL;
                        }
                        SDL_DestroyWindow(state->windows[i]);
                        state->windows[i] = NULL;
                        break;
                    }
                }
            }
        } break;
        case SDL_WINDOWEVENT_FOCUS_LOST:
            if (state->flash_on_focus_loss) {
                SDL_Window *window = SDL_GetWindowFromID(event->window.windowID);
                if (window) {
                    SDL_FlashWindow(window, SDL_FLASH_UNTIL_FOCUSED);
                }
            }
            break;
        default:
            break;
        }
        break;
    case SDL_KEYDOWN:
    {
        SDL_bool withControl = !!(event->key.keysym.mod & KMOD_CTRL);
        SDL_bool withShift = !!(event->key.keysym.mod & KMOD_SHIFT);
        SDL_bool withAlt = !!(event->key.keysym.mod & KMOD_ALT);

        switch (event->key.keysym.sym) {
            /* Add hotkeys here */
        case SDLK_PRINTSCREEN:
        {
            SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
            if (window) {
                for (i = 0; i < state->num_windows; ++i) {
                    if (window == state->windows[i]) {
                        SDLTest_ScreenShot(state->renderers[i]);
                    }
                }
            }
        } break;
        case SDLK_EQUALS:
            if (withControl) {
                /* Ctrl-+ double the size of the window */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    int w, h;
                    SDL_GetWindowSize(window, &w, &h);
                    SDL_SetWindowSize(window, w * 2, h * 2);
                }
            }
            break;
        case SDLK_MINUS:
            if (withControl) {
                /* Ctrl-- half the size of the window */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    int w, h;
                    SDL_GetWindowSize(window, &w, &h);
                    SDL_SetWindowSize(window, w / 2, h / 2);
                }
            }
            break;
        case SDLK_UP:
        case SDLK_DOWN:
        case SDLK_LEFT:
        case SDLK_RIGHT:
            if (withAlt) {
                /* Alt-Up/Down/Left/Right switches between displays */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    int currentIndex = SDL_GetWindowDisplayIndex(window);
                    int numDisplays = SDL_GetNumVideoDisplays();

                    if (currentIndex >= 0 && numDisplays >= 1) {
                        int dest;
                        if (event->key.keysym.sym == SDLK_UP || event->key.keysym.sym == SDLK_LEFT) {
                            dest = (currentIndex + numDisplays - 1) % numDisplays;
                        } else {
                            dest = (currentIndex + numDisplays + 1) % numDisplays;
                        }
                        SDL_Log("Centering on display %d\n", dest);
                        SDL_SetWindowPosition(window,
                                              SDL_WINDOWPOS_CENTERED_DISPLAY(dest),
                                              SDL_WINDOWPOS_CENTERED_DISPLAY(dest));
                    }
                }
            }
            if (withShift) {
                /* Shift-Up/Down/Left/Right shift the window by 100px */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    const int delta = 100;
                    int x, y;
                    SDL_GetWindowPosition(window, &x, &y);

                    if (event->key.keysym.sym == SDLK_UP) {
                        y -= delta;
                    }
                    if (event->key.keysym.sym == SDLK_DOWN) {
                        y += delta;
                    }
                    if (event->key.keysym.sym == SDLK_LEFT) {
                        x -= delta;
                    }
                    if (event->key.keysym.sym == SDLK_RIGHT) {
                        x += delta;
                    }

                    SDL_Log("Setting position to (%d, %d)\n", x, y);
                    SDL_SetWindowPosition(window, x, y);
                }
            }
            break;
        case SDLK_o:
            if (withControl) {
                /* Ctrl-O (or Ctrl-Shift-O) changes window opacity. */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    float opacity;
                    if (SDL_GetWindowOpacity(window, &opacity) == 0) {
                        if (withShift) {
                            opacity += 0.20f;
                        } else {
                            opacity -= 0.20f;
                        }
                        SDL_SetWindowOpacity(window, opacity);
                    }
                }
            }
            break;

        case SDLK_c:
            if (withControl) {
                /* Ctrl-C copy awesome text! */
                SDL_SetClipboardText("SDL rocks!\nYou know it!");
                SDL_Log("Copied text to clipboard\n");
            }
            if (withAlt) {
                /* Alt-C toggle a render clip rectangle */
                for (i = 0; i < state->num_windows; ++i) {
                    int w, h;
                    if (state->renderers[i]) {
                        SDL_Rect clip;
                        SDL_GetWindowSize(state->windows[i], &w, &h);
                        SDL_RenderGetClipRect(state->renderers[i], &clip);
                        if (SDL_RectEmpty(&clip)) {
                            clip.x = w / 4;
                            clip.y = h / 4;
                            clip.w = w / 2;
                            clip.h = h / 2;
                            SDL_RenderSetClipRect(state->renderers[i], &clip);
                        } else {
                            SDL_RenderSetClipRect(state->renderers[i], NULL);
                        }
                    }
                }
            }
            if (withShift) {
                SDL_Window *current_win = SDL_GetKeyboardFocus();
                if (current_win) {
                    const SDL_bool shouldCapture = (SDL_GetWindowFlags(current_win) & SDL_WINDOW_MOUSE_CAPTURE) == 0;
                    const int rc = SDL_CaptureMouse(shouldCapture);
                    SDL_Log("%sapturing mouse %s!\n", shouldCapture ? "C" : "Unc", (rc == 0) ? "succeeded" : "failed");
                }
            }
            break;
        case SDLK_v:
            if (withControl) {
                /* Ctrl-V paste awesome text! */
                char *text = SDL_GetClipboardText();
                if (*text) {
                    SDL_Log("Clipboard: %s\n", text);
                } else {
                    SDL_Log("Clipboard is empty\n");
                }
                SDL_free(text);
            }
            break;
        case SDLK_f:
            if (withControl) {
                /* Ctrl-F flash the window */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    SDL_FlashWindow(window, SDL_FLASH_BRIEFLY);
                }
            }
            break;
        case SDLK_g:
            if (withControl) {
                /* Ctrl-G toggle mouse grab */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    SDL_SetWindowGrab(window, !SDL_GetWindowGrab(window) ? SDL_TRUE : SDL_FALSE);
                }
            }
            break;
        case SDLK_k:
            if (withControl) {
                /* Ctrl-K toggle keyboard grab */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    SDL_SetWindowKeyboardGrab(window, !SDL_GetWindowKeyboardGrab(window) ? SDL_TRUE : SDL_FALSE);
                }
            }
            break;
        case SDLK_m:
            if (withControl) {
                /* Ctrl-M maximize */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    Uint32 flags = SDL_GetWindowFlags(window);
                    if (flags & SDL_WINDOW_MAXIMIZED) {
                        SDL_RestoreWindow(window);
                    } else {
                        SDL_MaximizeWindow(window);
                    }
                }
            }
            break;
        case SDLK_r:
            if (withControl) {
                /* Ctrl-R toggle mouse relative mode */
                SDL_SetRelativeMouseMode(!SDL_GetRelativeMouseMode() ? SDL_TRUE : SDL_FALSE);
            }
            break;
        case SDLK_t:
            if (withControl) {
                /* Ctrl-T toggle topmost mode */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    Uint32 flags = SDL_GetWindowFlags(window);
                    if (flags & SDL_WINDOW_ALWAYS_ON_TOP) {
                        SDL_SetWindowAlwaysOnTop(window, SDL_FALSE);
                    } else {
                        SDL_SetWindowAlwaysOnTop(window, SDL_TRUE);
                    }
                }
            }
            break;
        case SDLK_z:
            if (withControl) {
                /* Ctrl-Z minimize */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    SDL_MinimizeWindow(window);
                }
            }
            break;
        case SDLK_RETURN:
            if (withControl) {
                /* Ctrl-Enter toggle fullscreen */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    Uint32 flags = SDL_GetWindowFlags(window);
                    if (flags & SDL_WINDOW_FULLSCREEN) {
                        SDL_SetWindowFullscreen(window, SDL_FALSE);
                    } else {
                        SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
                    }
                }
            } else if (withAlt) {
                /* Alt-Enter toggle fullscreen desktop */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    Uint32 flags = SDL_GetWindowFlags(window);
                    if (flags & SDL_WINDOW_FULLSCREEN) {
                        SDL_SetWindowFullscreen(window, SDL_FALSE);
                    } else {
                        SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN_DESKTOP);
                    }
                }
            } else if (withShift) {
                /* Shift-Enter toggle fullscreen desktop / fullscreen */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    Uint32 flags = SDL_GetWindowFlags(window);
                    if ((flags & SDL_WINDOW_FULLSCREEN_DESKTOP) == SDL_WINDOW_FULLSCREEN_DESKTOP) {
                        SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
                    } else {
                        SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN_DESKTOP);
                    }
                }
            }

            break;
        case SDLK_b:
            if (withControl) {
                /* Ctrl-B toggle window border */
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                if (window) {
                    const Uint32 flags = SDL_GetWindowFlags(window);
                    const SDL_bool b = (flags & SDL_WINDOW_BORDERLESS) ? SDL_TRUE : SDL_FALSE;
                    SDL_SetWindowBordered(window, b);
                }
            }
            break;
        case SDLK_a:
            if (withControl) {
                /* Ctrl-A reports absolute mouse position. */
                int x, y;
                const Uint32 mask = SDL_GetGlobalMouseState(&x, &y);
                SDL_Log("ABSOLUTE MOUSE: (%d, %d)%s%s%s%s%s\n", x, y,
                        (mask & SDL_BUTTON_LMASK) ? " [LBUTTON]" : "",
                        (mask & SDL_BUTTON_MMASK) ? " [MBUTTON]" : "",
                        (mask & SDL_BUTTON_RMASK) ? " [RBUTTON]" : "",
                        (mask & SDL_BUTTON_X1MASK) ? " [X2BUTTON]" : "",
                        (mask & SDL_BUTTON_X2MASK) ? " [X2BUTTON]" : "");
            }
            break;
        case SDLK_0:
            if (withControl) {
                SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);
                SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "Test Message", "You're awesome!", window);
            }
            break;
        case SDLK_1:
            if (withControl) {
                FullscreenTo(0, event->key.windowID);
            }
            break;
        case SDLK_2:
            if (withControl) {
                FullscreenTo(1, event->key.windowID);
            }
            break;
        case SDLK_ESCAPE:
            *done = 1;
            break;
        case SDLK_SPACE:
        {
            char message[256];
            SDL_Window *window = SDL_GetWindowFromID(event->key.windowID);

            (void)SDL_snprintf(message, sizeof(message), "(%" SDL_PRIs32 ", %" SDL_PRIs32 "), rel (%" SDL_PRIs32 ", %" SDL_PRIs32 ")\n",
                               lastEvent.x, lastEvent.y, lastEvent.xrel, lastEvent.yrel);
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "Last mouse position", message, window);
            break;
        }
        default:
            break;
        }
        break;
    }
    case SDL_QUIT:
        *done = 1;
        break;
    case SDL_MOUSEMOTION:
        lastEvent = event->motion;
        break;

    case SDL_DROPFILE:
    case SDL_DROPTEXT:
        SDL_free(event->drop.file);
        break;
    }
}

void SDLTest_CommonQuit(SDLTest_CommonState *state)
{
    int i;

    SDL_free(common_usage_video);
    SDL_free(common_usage_audio);
    SDL_free(common_usage_videoaudio);
    common_usage_video = NULL;
    common_usage_audio = NULL;
    common_usage_videoaudio = NULL;

    SDL_free(state->windows);
    if (state->targets) {
        for (i = 0; i < state->num_windows; ++i) {
            if (state->targets[i]) {
                SDL_DestroyTexture(state->targets[i]);
            }
        }
        SDL_free(state->targets);
    }
    if (state->renderers) {
        for (i = 0; i < state->num_windows; ++i) {
            if (state->renderers[i]) {
                SDL_DestroyRenderer(state->renderers[i]);
            }
        }
        SDL_free(state->renderers);
    }
    if (state->flags & SDL_INIT_VIDEO) {
        SDL_VideoQuit();
    }
    if (state->flags & SDL_INIT_AUDIO) {
        SDL_AudioQuit();
    }
    SDL_free(state);
    SDL_Quit();
    SDLTest_LogAllocations();
}

void SDLTest_CommonDrawWindowInfo(SDL_Renderer *renderer, SDL_Window *window, int *usedHeight)
{
    char text[1024];
    int textY = 0;
    const int lineHeight = 10;
    int x, y, w, h;
    SDL_Rect rect;
    SDL_DisplayMode mode;
    float ddpi, hdpi, vdpi;
    float scaleX, scaleY;
    Uint32 flags;
    const int windowDisplayIndex = SDL_GetWindowDisplayIndex(window);
    SDL_RendererInfo info;

    /* Video */

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDLTest_DrawString(renderer, 0, textY, "-- Video --");
    textY += lineHeight;

    SDL_SetRenderDrawColor(renderer, 170, 170, 170, 255);

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetCurrentVideoDriver: %s", SDL_GetCurrentVideoDriver());
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    /* Renderer */

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDLTest_DrawString(renderer, 0, textY, "-- Renderer --");
    textY += lineHeight;

    SDL_SetRenderDrawColor(renderer, 170, 170, 170, 255);

    if (0 == SDL_GetRendererInfo(renderer, &info)) {
        (void)SDL_snprintf(text, sizeof(text), "SDL_GetRendererInfo: name: %s", info.name);
        SDLTest_DrawString(renderer, 0, textY, text);
        textY += lineHeight;
    }

    if (0 == SDL_GetRendererOutputSize(renderer, &w, &h)) {
        (void)SDL_snprintf(text, sizeof(text), "SDL_GetRendererOutputSize: %dx%d", w, h);
        SDLTest_DrawString(renderer, 0, textY, text);
        textY += lineHeight;
    }

    SDL_RenderGetViewport(renderer, &rect);
    (void)SDL_snprintf(text, sizeof(text), "SDL_RenderGetViewport: %d,%d, %dx%d",
                       rect.x, rect.y, rect.w, rect.h);
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    SDL_RenderGetScale(renderer, &scaleX, &scaleY);
    (void)SDL_snprintf(text, sizeof(text), "SDL_RenderGetScale: %f,%f",
                       scaleX, scaleY);
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    SDL_RenderGetLogicalSize(renderer, &w, &h);
    (void)SDL_snprintf(text, sizeof(text), "SDL_RenderGetLogicalSize: %dx%d", w, h);
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    /* Window */

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDLTest_DrawString(renderer, 0, textY, "-- Window --");
    textY += lineHeight;

    SDL_SetRenderDrawColor(renderer, 170, 170, 170, 255);

    SDL_GetWindowPosition(window, &x, &y);
    (void)SDL_snprintf(text, sizeof(text), "SDL_GetWindowPosition: %d,%d", x, y);
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    SDL_GetWindowSize(window, &w, &h);
    (void)SDL_snprintf(text, sizeof(text), "SDL_GetWindowSize: %dx%d", w, h);
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetWindowFlags: ");
    SDLTest_PrintWindowFlags(text, sizeof(text), SDL_GetWindowFlags(window));
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    if (0 == SDL_GetWindowDisplayMode(window, &mode)) {
        (void)SDL_snprintf(text, sizeof(text), "SDL_GetWindowDisplayMode: %dx%d@%dHz (%s)",
                           mode.w, mode.h, mode.refresh_rate, SDL_GetPixelFormatName(mode.format));
        SDLTest_DrawString(renderer, 0, textY, text);
        textY += lineHeight;
    }

    /* Display */

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDLTest_DrawString(renderer, 0, textY, "-- Display --");
    textY += lineHeight;

    SDL_SetRenderDrawColor(renderer, 170, 170, 170, 255);

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetWindowDisplayIndex: %d", windowDisplayIndex);
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetDisplayName: %s", SDL_GetDisplayName(windowDisplayIndex));
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    if (0 == SDL_GetDisplayBounds(windowDisplayIndex, &rect)) {
        (void)SDL_snprintf(text, sizeof(text), "SDL_GetDisplayBounds: %d,%d, %dx%d",
                           rect.x, rect.y, rect.w, rect.h);
        SDLTest_DrawString(renderer, 0, textY, text);
        textY += lineHeight;
    }

    if (0 == SDL_GetCurrentDisplayMode(windowDisplayIndex, &mode)) {
        (void)SDL_snprintf(text, sizeof(text), "SDL_GetCurrentDisplayMode: %dx%d@%d",
                           mode.w, mode.h, mode.refresh_rate);
        SDLTest_DrawString(renderer, 0, textY, text);
        textY += lineHeight;
    }

    if (0 == SDL_GetDesktopDisplayMode(windowDisplayIndex, &mode)) {
        (void)SDL_snprintf(text, sizeof(text), "SDL_GetDesktopDisplayMode: %dx%d@%d",
                           mode.w, mode.h, mode.refresh_rate);
        SDLTest_DrawString(renderer, 0, textY, text);
        textY += lineHeight;
    }

    if (0 == SDL_GetDisplayDPI(windowDisplayIndex, &ddpi, &hdpi, &vdpi)) {
        (void)SDL_snprintf(text, sizeof(text), "SDL_GetDisplayDPI: ddpi: %f, hdpi: %f, vdpi: %f",
                           ddpi, hdpi, vdpi);
        SDLTest_DrawString(renderer, 0, textY, text);
        textY += lineHeight;
    }

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetDisplayOrientation: ");
    SDLTest_PrintDisplayOrientation(text, sizeof(text), SDL_GetDisplayOrientation(windowDisplayIndex));
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    /* Mouse */

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDLTest_DrawString(renderer, 0, textY, "-- Mouse --");
    textY += lineHeight;

    SDL_SetRenderDrawColor(renderer, 170, 170, 170, 255);

    flags = SDL_GetMouseState(&x, &y);
    (void)SDL_snprintf(text, sizeof(text), "SDL_GetMouseState: %d,%d ", x, y);
    SDLTest_PrintButtonMask(text, sizeof(text), flags);
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    flags = SDL_GetGlobalMouseState(&x, &y);
    (void)SDL_snprintf(text, sizeof(text), "SDL_GetGlobalMouseState: %d,%d ", x, y);
    SDLTest_PrintButtonMask(text, sizeof(text), flags);
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    /* Keyboard */

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDLTest_DrawString(renderer, 0, textY, "-- Keyboard --");
    textY += lineHeight;

    SDL_SetRenderDrawColor(renderer, 170, 170, 170, 255);

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetModState: ");
    SDLTest_PrintModState(text, sizeof(text), SDL_GetModState());
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    if (usedHeight) {
        *usedHeight = textY;
    }
}

/* vi: set ts=4 sw=4 expandtab: */
