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

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <signal.h>
#include <errno.h>

#include "../SDL_sysvideo.h"

#include "SDL_mouse.h"
#include "../../events/SDL_mouse_c.h"
#include "SDL_waylandvideo.h"
#include "../SDL_pixels_c.h"
#include "SDL_waylandevents_c.h"

#include "wayland-cursor.h"
#include "SDL_waylandmouse.h"

#include "SDL_hints.h"
#include "../../SDL_hints_c.h"

static int Wayland_SetRelativeMouseMode(SDL_bool enabled);

typedef struct
{
    struct wl_buffer *buffer;
    struct wl_surface *surface;

    int hot_x, hot_y;
    int w, h;

    /* shm_data is non-NULL for custom cursors.
     * When shm_data is NULL, system_cursor must be valid
     */
    SDL_SystemCursor system_cursor;
    void *shm_data;
    size_t shm_data_size;
} Wayland_CursorData;

#ifdef SDL_USE_LIBDBUS

#include "../../core/linux/SDL_dbus.h"

static DBusMessage *wayland_read_dbus_setting(SDL_DBusContext *dbus, const char *key)
{
    static const char *iface = "org.gnome.desktop.interface";

    DBusMessage *reply = NULL;
    DBusMessage *msg = dbus->message_new_method_call("org.freedesktop.portal.Desktop",  /* Node */
                                                     "/org/freedesktop/portal/desktop", /* Path */
                                                     "org.freedesktop.portal.Settings", /* Interface */
                                                     "Read");                           /* Method */

    if (msg) {
        if (dbus->message_append_args(msg, DBUS_TYPE_STRING, &iface, DBUS_TYPE_STRING, &key, DBUS_TYPE_INVALID)) {
            reply = dbus->connection_send_with_reply_and_block(dbus->session_conn, msg, DBUS_TIMEOUT_USE_DEFAULT, NULL);
        }
        dbus->message_unref(msg);
    }

    return reply;
}

static SDL_bool wayland_parse_dbus_reply(SDL_DBusContext *dbus, DBusMessage *reply, int type, void *value)
{
    DBusMessageIter iter[3];

    dbus->message_iter_init(reply, &iter[0]);
    if (dbus->message_iter_get_arg_type(&iter[0]) != DBUS_TYPE_VARIANT) {
        return SDL_FALSE;
    }

    dbus->message_iter_recurse(&iter[0], &iter[1]);
    if (dbus->message_iter_get_arg_type(&iter[1]) != DBUS_TYPE_VARIANT) {
        return SDL_FALSE;
    }

    dbus->message_iter_recurse(&iter[1], &iter[2]);
    if (dbus->message_iter_get_arg_type(&iter[2]) != type) {
        return SDL_FALSE;
    }

    dbus->message_iter_get_basic(&iter[2], value);

    return SDL_TRUE;
}

static SDL_bool wayland_dbus_read_cursor_size(int *size)
{
    static const char *cursor_size_value = "cursor-size";

    DBusMessage *reply;
    SDL_DBusContext *dbus = SDL_DBus_GetContext();

    if (!dbus || !size) {
        return SDL_FALSE;
    }

    if ((reply = wayland_read_dbus_setting(dbus, cursor_size_value))) {
        if (wayland_parse_dbus_reply(dbus, reply, DBUS_TYPE_INT32, size)) {
            dbus->message_unref(reply);
            return SDL_TRUE;
        }
        dbus->message_unref(reply);
    }

    return SDL_FALSE;
}

static SDL_bool wayland_dbus_read_cursor_theme(char **theme)
{
    static const char *cursor_theme_value = "cursor-theme";

    DBusMessage *reply;
    SDL_DBusContext *dbus = SDL_DBus_GetContext();

    if (!dbus || !theme) {
        return SDL_FALSE;
    }

    if ((reply = wayland_read_dbus_setting(dbus, cursor_theme_value))) {
        const char *temp;
        if (wayland_parse_dbus_reply(dbus, reply, DBUS_TYPE_STRING, &temp)) {
            *theme = SDL_strdup(temp);
            dbus->message_unref(reply);
            return SDL_TRUE;
        }
        dbus->message_unref(reply);
    }

    return SDL_FALSE;
}

#endif


static const char *GetLegacyCursorName(SDL_SystemCursor system_cursor)
{
    switch (system_cursor) {
        case SDL_SYSTEM_CURSOR_ARROW: return "left_ptr";
        case SDL_SYSTEM_CURSOR_IBEAM: return "xterm";
        case SDL_SYSTEM_CURSOR_WAIT: return "watch";
        case SDL_SYSTEM_CURSOR_CROSSHAIR: return "tcross";
        case SDL_SYSTEM_CURSOR_WAITARROW: return "watch";
        case SDL_SYSTEM_CURSOR_SIZENWSE: return "top_left_corner";
        case SDL_SYSTEM_CURSOR_SIZENESW: return "top_right_corner";
        case SDL_SYSTEM_CURSOR_SIZEWE: return "sb_h_double_arrow";
        case SDL_SYSTEM_CURSOR_SIZENS: return "sb_v_double_arrow";
        case SDL_SYSTEM_CURSOR_SIZEALL: return "fleur";
        case SDL_SYSTEM_CURSOR_NO: return "pirate";
        case SDL_SYSTEM_CURSOR_HAND: return "hand2";
        case SDL_NUM_SYSTEM_CURSORS: break;  /* so the compiler might notice if an enum value is missing here. */
    }

    SDL_assert(0);
    return NULL;
}

static SDL_bool wayland_get_system_cursor(SDL_VideoData *vdata, Wayland_CursorData *cdata, float *scale)
{
    struct wl_cursor_theme *theme = NULL;
    struct wl_cursor *cursor;
    const char *cssname = NULL;
    const char *fallback_name = NULL;

    char *xcursor_size;
    int size = 0;

    SDL_Window *focus;
    int i;

    /*
     * GNOME based desktops expose the cursor size and theme via the
     * org.freedesktop.portal.Settings interface of the xdg-desktop portal.
     * Try XCURSOR_SIZE and XCURSOR_THEME first, so user specified sizes and
     * themes take precedence over all, then try D-Bus if the envvar isn't
     * set, then fall back to the defaults if none of the preceding values
     * are available or valid.
     */
    if ((xcursor_size = SDL_getenv("XCURSOR_SIZE"))) {
        size = SDL_atoi(xcursor_size);
    }
#ifdef SDL_USE_LIBDBUS
    if (size <= 0) {
        wayland_dbus_read_cursor_size(&size);
    }
#endif
    if (size <= 0) {
        size = 24;
    }
    /* First, find the appropriate theme based on the current scale... */
    focus = SDL_GetMouse()->focus;
    if (focus) {
        SDL_WindowData *focusdata = (SDL_WindowData*)focus->driverdata;

        /* Cursors use integer scaling. */
        *scale = SDL_ceilf(focusdata->scale_factor);
    } else {
        *scale = 1.0f;
    }

    size *= (int)*scale;
    for (i = 0; i < vdata->num_cursor_themes; ++i) {
        if (vdata->cursor_themes[i].size == size) {
            theme = vdata->cursor_themes[i].theme;
            break;
        }
    }
    if (!theme) {
        char *xcursor_theme = NULL;
        SDL_bool free_theme_str = SDL_FALSE;

        vdata->cursor_themes = SDL_realloc(vdata->cursor_themes,
                                           sizeof(SDL_WaylandCursorTheme) * (vdata->num_cursor_themes + 1));
        if (!vdata->cursor_themes) {
            SDL_OutOfMemory();
            return SDL_FALSE;
        }
        xcursor_theme = SDL_getenv("XCURSOR_THEME");
#ifdef SDL_USE_LIBDBUS
        if (!xcursor_theme) {
            /* Allocates the string with SDL_strdup, which must be freed. */
            free_theme_str = wayland_dbus_read_cursor_theme(&xcursor_theme);
        }
#endif
        theme = WAYLAND_wl_cursor_theme_load(xcursor_theme, size, vdata->shm);
        vdata->cursor_themes[vdata->num_cursor_themes].size = size;
        vdata->cursor_themes[vdata->num_cursor_themes++].theme = theme;

        if (free_theme_str) {
            SDL_free(xcursor_theme);
        }
    }

    /* Next, find the cursor from the theme... */
    cssname = SDL_GetCSSCursorName(cdata->system_cursor, &fallback_name);

    cursor = WAYLAND_wl_cursor_theme_get_cursor(theme, cssname);
    if (!cursor && fallback_name) {
        cursor = WAYLAND_wl_cursor_theme_get_cursor(theme, fallback_name);
    }

    /* try the legacy name if the fancy new CSS name doesn't work... */
    if (!cursor) {
        cursor = WAYLAND_wl_cursor_theme_get_cursor(theme, GetLegacyCursorName(cdata->system_cursor));
    }

    /* Fallback to the default cursor if the chosen one wasn't found */
    if (!cursor) {
        cursor = WAYLAND_wl_cursor_theme_get_cursor(theme, "default");
    }
    /* Try the old X11 name as a last resort */
    if (!cursor) {
        cursor = WAYLAND_wl_cursor_theme_get_cursor(theme, "left_ptr");
    }
    if (!cursor) {
        return SDL_FALSE;
    }

    /* ... Set the cursor data, finally. */
    cdata->buffer = WAYLAND_wl_cursor_image_get_buffer(cursor->images[0]);
    cdata->hot_x = cursor->images[0]->hotspot_x;
    cdata->hot_y = cursor->images[0]->hotspot_y;
    cdata->w = cursor->images[0]->width;
    cdata->h = cursor->images[0]->height;
    return SDL_TRUE;
}

static int set_tmp_file_size(int fd, off_t size)
{
#ifdef HAVE_POSIX_FALLOCATE
    sigset_t set, old_set;
    int ret;

    /* SIGALRM can potentially block a large posix_fallocate() operation
     * from succeeding, so block it.
     */
    sigemptyset(&set);
    sigaddset(&set, SIGALRM);
    sigprocmask(SIG_BLOCK, &set, &old_set);

    do {
        ret = posix_fallocate(fd, 0, size);
    } while (ret == EINTR);

    sigprocmask(SIG_SETMASK, &old_set, NULL);

    if (ret == 0) {
        return 0;
    }
    else if (ret != EINVAL && errno != EOPNOTSUPP) {
        return -1;
    }
#endif

    if (ftruncate(fd, size) < 0) {
        return -1;
    }
    return 0;
}

static int wayland_create_tmp_file(off_t size)
{
    int fd;

#ifdef HAVE_MEMFD_CREATE
    fd = memfd_create("SDL", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd >= 0) {
        fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_SEAL);
    } else
#endif
    {
        static const char template[] = "/sdl-shared-XXXXXX";
        char *xdg_path;
        char tmp_path[PATH_MAX];

        xdg_path = SDL_getenv("XDG_RUNTIME_DIR");
        if (!xdg_path) {
            return -1;
        }

        SDL_strlcpy(tmp_path, xdg_path, PATH_MAX);
        SDL_strlcat(tmp_path, template, PATH_MAX);

        fd = mkostemp(tmp_path, O_CLOEXEC);
        if (fd < 0) {
            return -1;
        }

        /* Need to manually unlink the temp files, or they can persist after close and fill up the temp storage. */
        unlink(tmp_path);
    }

    if (set_tmp_file_size(fd, size) < 0) {
        close(fd);
        return -1;
    }

    return fd;
}

static void mouse_buffer_release(void *data, struct wl_buffer *buffer)
{
}

static const struct wl_buffer_listener mouse_buffer_listener = {
    mouse_buffer_release
};

static int create_buffer_from_shm(Wayland_CursorData *d,
                                  int width,
                                  int height,
                                  uint32_t format)
{
    SDL_VideoDevice *vd = SDL_GetVideoDevice();
    SDL_VideoData *data = (SDL_VideoData *)vd->driverdata;
    struct wl_shm_pool *shm_pool;
    int shm_fd;

    int stride = width * 4;
    d->shm_data_size = stride * height;

    shm_fd = wayland_create_tmp_file(d->shm_data_size);
    if (shm_fd < 0) {
        return SDL_SetError("Creating mouse cursor buffer failed.");
    }

    d->shm_data = mmap(NULL,
                       d->shm_data_size,
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED,
                       shm_fd,
                       0);
    if (d->shm_data == MAP_FAILED) {
        d->shm_data = NULL;
        close(shm_fd);
        return SDL_SetError("mmap() failed.");
    }

    SDL_assert(d->shm_data != NULL);

    shm_pool = wl_shm_create_pool(data->shm, shm_fd, d->shm_data_size);
    d->buffer = wl_shm_pool_create_buffer(shm_pool,
                                          0,
                                          width,
                                          height,
                                          stride,
                                          format);
    wl_buffer_add_listener(d->buffer,
                           &mouse_buffer_listener,
                           d);

    wl_shm_pool_destroy(shm_pool);
    close(shm_fd);

    return 0;
}

static SDL_Cursor *Wayland_CreateCursor(SDL_Surface *surface, int hot_x, int hot_y)
{
    SDL_Cursor *cursor;

    cursor = SDL_calloc(1, sizeof(*cursor));
    if (cursor) {
        SDL_VideoDevice *vd = SDL_GetVideoDevice();
        SDL_VideoData *wd = (SDL_VideoData *)vd->driverdata;
        Wayland_CursorData *data = SDL_calloc(1, sizeof(Wayland_CursorData));
        if (!data) {
            SDL_OutOfMemory();
            SDL_free(cursor);
            return NULL;
        }
        cursor->driverdata = (void *)data;

        /* Allocate shared memory buffer for this cursor */
        if (create_buffer_from_shm(data,
                                   surface->w,
                                   surface->h,
                                   WL_SHM_FORMAT_ARGB8888) < 0) {
            SDL_free(cursor->driverdata);
            SDL_free(cursor);
            return NULL;
        }

        /* Wayland requires premultiplied alpha for its surfaces. */
        SDL_PremultiplyAlpha(surface->w, surface->h,
                             surface->format->format, surface->pixels, surface->pitch,
                             SDL_PIXELFORMAT_ARGB8888, data->shm_data, surface->w * 4);

        data->surface = wl_compositor_create_surface(wd->compositor);
        wl_surface_set_user_data(data->surface, NULL);

        data->hot_x = hot_x;
        data->hot_y = hot_y;
        data->w = surface->w;
        data->h = surface->h;
    } else {
        SDL_OutOfMemory();
    }

    return cursor;
}

static SDL_Cursor *Wayland_CreateSystemCursor(SDL_SystemCursor id)
{
    SDL_VideoData *data = SDL_GetVideoDevice()->driverdata;
    SDL_Cursor *cursor;

    cursor = SDL_calloc(1, sizeof(*cursor));
    if (cursor) {
        Wayland_CursorData *cdata = SDL_calloc(1, sizeof(Wayland_CursorData));
        if (!cdata) {
            SDL_OutOfMemory();
            SDL_free(cursor);
            return NULL;
        }
        cursor->driverdata = (void *)cdata;

        cdata->surface = wl_compositor_create_surface(data->compositor);
        wl_surface_set_user_data(cdata->surface, NULL);

        /* Note that we can't actually set any other cursor properties, as this
         * is output-specific. See wayland_get_system_cursor for the rest!
         */
        cdata->system_cursor = id;
    } else {
        SDL_OutOfMemory();
    }

    return cursor;
}

static SDL_Cursor *Wayland_CreateDefaultCursor()
{
    return Wayland_CreateSystemCursor(SDL_SYSTEM_CURSOR_ARROW);
}

static void Wayland_FreeCursorData(Wayland_CursorData *d)
{
    if (d->buffer) {
        if (d->shm_data) {
            wl_buffer_destroy(d->buffer);
            munmap(d->shm_data, d->shm_data_size);
        }
        d->buffer = NULL;
    }

    if (d->surface) {
        wl_surface_destroy(d->surface);
        d->surface = NULL;
    }
}

static void Wayland_FreeCursor(SDL_Cursor *cursor)
{
    if (!cursor) {
        return;
    }

    /* Probably not a cursor we own */
    if (!cursor->driverdata) {
        return;
    }

    Wayland_FreeCursorData((Wayland_CursorData *)cursor->driverdata);

    SDL_free(cursor->driverdata);
    SDL_free(cursor);
}

static int Wayland_ShowCursor(SDL_Cursor *cursor)
{
    SDL_VideoDevice *vd = SDL_GetVideoDevice();
    SDL_VideoData *d = vd->driverdata;
    struct SDL_WaylandInput *input = d->input;
    struct wl_pointer *pointer = d->pointer;
    float scale = 1.0f;

    if (!pointer) {
        return -1;
    }

    if (cursor) {
        Wayland_CursorData *data = cursor->driverdata;

        /* TODO: High-DPI custom cursors? -flibit */
        if (!data->shm_data) {
            if (!wayland_get_system_cursor(d, data, &scale)) {
                return -1;
            }
        }

        wl_surface_set_buffer_scale(data->surface, scale);
        wl_pointer_set_cursor(pointer,
                              input->pointer_enter_serial,
                              data->surface,
                              data->hot_x / scale,
                              data->hot_y / scale);
        wl_surface_attach(data->surface, data->buffer, 0, 0);
        wl_surface_damage(data->surface, 0, 0, data->w, data->h);
        wl_surface_commit(data->surface);

        input->cursor_visible = SDL_TRUE;

        if (input->relative_mode_override) {
            Wayland_input_unlock_pointer(input);
            input->relative_mode_override = SDL_FALSE;
        }

    } else {
        input->cursor_visible = SDL_FALSE;
        wl_pointer_set_cursor(pointer, input->pointer_enter_serial, NULL, 0, 0);
    }

    return 0;
}

static void Wayland_WarpMouse(SDL_Window *window, int x, int y)
{
    SDL_VideoDevice *vd = SDL_GetVideoDevice();
    SDL_VideoData *d = vd->driverdata;
    struct SDL_WaylandInput *input = d->input;

    if (input->cursor_visible == SDL_TRUE) {
        SDL_Unsupported();
    } else if (input->warp_emulation_prohibited) {
        SDL_Unsupported();
    } else {
        if (!d->relative_mouse_mode) {
            Wayland_input_lock_pointer(input);
            input->relative_mode_override = SDL_TRUE;
        }
        SDL_SendMouseMotion(window, 0, 0, x, y);
    }
}

static int Wayland_WarpMouseGlobal(int x, int y)
{
    return SDL_Unsupported();
}

static int Wayland_SetRelativeMouseMode(SDL_bool enabled)
{
    SDL_VideoDevice *vd = SDL_GetVideoDevice();
    SDL_VideoData *data = (SDL_VideoData *)vd->driverdata;

    if (enabled) {
        /* Disable mouse warp emulation if it's enabled. */
        if (data->input->relative_mode_override) {
            data->input->relative_mode_override = SDL_FALSE;
        }

        /* If the app has used relative mode before, it probably shouldn't
         * also be emulating it using repeated mouse warps, so disable
         * mouse warp emulation by default.
         */
        data->input->warp_emulation_prohibited = SDL_TRUE;
        return Wayland_input_lock_pointer(data->input);
    } else {
        return Wayland_input_unlock_pointer(data->input);
    }
}

static void SDLCALL Wayland_EmulateMouseWarpChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    struct SDL_WaylandInput *input = (struct SDL_WaylandInput *)userdata;

    input->warp_emulation_prohibited = !SDL_GetStringBoolean(hint, !input->warp_emulation_prohibited);
}

#if 0  /* TODO RECONNECT: See waylandvideo.c for more information! */
static void Wayland_RecreateCursor(SDL_Cursor *cursor, SDL_VideoData *vdata)
{
    Wayland_CursorData *cdata = (Wayland_CursorData *) cursor->driverdata;

    /* Probably not a cursor we own */
    if (cdata == NULL) {
        return;
    }

    Wayland_FreeCursorData(cdata);

    /* We're not currently freeing this, so... yolo? */
    if (cdata->shm_data != NULL) {
        void *old_data_pointer = cdata->shm_data;
        int stride = cdata->w * 4;

        create_buffer_from_shm(cdata, cdata->w, cdata->h, WL_SHM_FORMAT_ARGB8888);

        SDL_memcpy(cdata->shm_data, old_data_pointer, stride * cdata->h);
    }
    cdata->surface = wl_compositor_create_surface(vdata->compositor);
    wl_surface_set_user_data(cdata->surface, NULL);
}

void Wayland_RecreateCursors(void)
{
    SDL_Cursor *cursor;
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_VideoData *vdata = SDL_GetVideoDevice()->driverdata;

    if (vdata && vdata->cursor_themes) {
        SDL_free(vdata->cursor_themes);
        vdata->cursor_themes = NULL;
        vdata->num_cursor_themes = 0;
    }

    if (mouse == NULL) {
        return;
    }

    for (cursor = mouse->cursors; cursor != NULL; cursor = cursor->next) {
        Wayland_RecreateCursor(cursor, vdata);
    }
    if (mouse->def_cursor) {
        Wayland_RecreateCursor(mouse->def_cursor, vdata);
    }
    if (mouse->cur_cursor) {
        Wayland_RecreateCursor(mouse->cur_cursor, vdata);
        if (mouse->cursor_shown) {
            Wayland_ShowCursor(mouse->cur_cursor);
        }
    }
}
#endif /* 0 */

void Wayland_InitMouse(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_VideoDevice *vd = SDL_GetVideoDevice();
    SDL_VideoData *d = vd->driverdata;
    struct SDL_WaylandInput *input = d->input;

    mouse->CreateCursor = Wayland_CreateCursor;
    mouse->CreateSystemCursor = Wayland_CreateSystemCursor;
    mouse->ShowCursor = Wayland_ShowCursor;
    mouse->FreeCursor = Wayland_FreeCursor;
    mouse->WarpMouse = Wayland_WarpMouse;
    mouse->WarpMouseGlobal = Wayland_WarpMouseGlobal;
    mouse->SetRelativeMouseMode = Wayland_SetRelativeMouseMode;

    input->relative_mode_override = SDL_FALSE;
    input->cursor_visible = SDL_TRUE;

    SDL_SetDefaultCursor(Wayland_CreateDefaultCursor());

    SDL_AddHintCallback(SDL_HINT_VIDEO_WAYLAND_EMULATE_MOUSE_WARP,
                        Wayland_EmulateMouseWarpChanged, input);
}

void Wayland_FiniMouse(SDL_VideoData *data)
{
    struct SDL_WaylandInput *input = data->input;
    int i;
    for (i = 0; i < data->num_cursor_themes; i += 1) {
        WAYLAND_wl_cursor_theme_destroy(data->cursor_themes[i].theme);
    }
    data->num_cursor_themes = 0;
    SDL_free(data->cursor_themes);
    data->cursor_themes = NULL;

    SDL_DelHintCallback(SDL_HINT_VIDEO_WAYLAND_EMULATE_MOUSE_WARP,
                        Wayland_EmulateMouseWarpChanged, input);
}

#endif /* SDL_VIDEO_DRIVER_WAYLAND */

/* vi: set ts=4 sw=4 expandtab: */
