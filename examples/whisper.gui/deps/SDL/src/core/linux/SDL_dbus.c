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
#include "SDL_hints.h"
#include "SDL_dbus.h"
#include "SDL_atomic.h"
#include "SDL_sandbox.h"
#include "../../stdlib/SDL_vacopy.h"

#ifdef SDL_USE_LIBDBUS
/* we never link directly to libdbus. */
#include "SDL_loadso.h"
static const char *dbus_library = "libdbus-1.so.3";
static void *dbus_handle = NULL;
static char *inhibit_handle = NULL;
static unsigned int screensaver_cookie = 0;
static SDL_DBusContext dbus;

static int LoadDBUSSyms(void)
{
#define SDL_DBUS_SYM2_OPTIONAL(x, y)                   \
    dbus.x = SDL_LoadFunction(dbus_handle, #y)

#define SDL_DBUS_SYM2(x, y)                            \
    if (!(dbus.x = SDL_LoadFunction(dbus_handle, #y))) \
    return -1

#define SDL_DBUS_SYM(x) \
    SDL_DBUS_SYM2(x, dbus_##x)

#define SDL_DBUS_SYM_OPTIONAL(x) \
    SDL_DBUS_SYM2_OPTIONAL(x, dbus_##x)

    SDL_DBUS_SYM(bus_get_private);
    SDL_DBUS_SYM(bus_register);
    SDL_DBUS_SYM(bus_add_match);
    SDL_DBUS_SYM(connection_open_private);
    SDL_DBUS_SYM(connection_set_exit_on_disconnect);
    SDL_DBUS_SYM(connection_get_is_connected);
    SDL_DBUS_SYM(connection_add_filter);
    SDL_DBUS_SYM(connection_try_register_object_path);
    SDL_DBUS_SYM(connection_send);
    SDL_DBUS_SYM(connection_send_with_reply_and_block);
    SDL_DBUS_SYM(connection_close);
    SDL_DBUS_SYM(connection_ref);
    SDL_DBUS_SYM(connection_unref);
    SDL_DBUS_SYM(connection_flush);
    SDL_DBUS_SYM(connection_read_write);
    SDL_DBUS_SYM(connection_dispatch);
    SDL_DBUS_SYM(message_is_signal);
    SDL_DBUS_SYM(message_new_method_call);
    SDL_DBUS_SYM(message_append_args);
    SDL_DBUS_SYM(message_append_args_valist);
    SDL_DBUS_SYM(message_iter_init_append);
    SDL_DBUS_SYM(message_iter_open_container);
    SDL_DBUS_SYM(message_iter_append_basic);
    SDL_DBUS_SYM(message_iter_close_container);
    SDL_DBUS_SYM(message_get_args);
    SDL_DBUS_SYM(message_get_args_valist);
    SDL_DBUS_SYM(message_iter_init);
    SDL_DBUS_SYM(message_iter_next);
    SDL_DBUS_SYM(message_iter_get_basic);
    SDL_DBUS_SYM(message_iter_get_arg_type);
    SDL_DBUS_SYM(message_iter_recurse);
    SDL_DBUS_SYM(message_unref);
    SDL_DBUS_SYM(threads_init_default);
    SDL_DBUS_SYM(error_init);
    SDL_DBUS_SYM(error_is_set);
    SDL_DBUS_SYM(error_free);
    SDL_DBUS_SYM(get_local_machine_id);
    SDL_DBUS_SYM_OPTIONAL(try_get_local_machine_id);
    SDL_DBUS_SYM(free);
    SDL_DBUS_SYM(free_string_array);
    SDL_DBUS_SYM(shutdown);

#undef SDL_DBUS_SYM
#undef SDL_DBUS_SYM2

    return 0;
}

static void UnloadDBUSLibrary(void)
{
    if (dbus_handle) {
        SDL_UnloadObject(dbus_handle);
        dbus_handle = NULL;
    }
}

static int LoadDBUSLibrary(void)
{
    int retval = 0;
    if (!dbus_handle) {
        dbus_handle = SDL_LoadObject(dbus_library);
        if (!dbus_handle) {
            retval = -1;
            /* Don't call SDL_SetError(): SDL_LoadObject already did. */
        } else {
            retval = LoadDBUSSyms();
            if (retval < 0) {
                UnloadDBUSLibrary();
            }
        }
    }

    return retval;
}

static SDL_SpinLock spinlock_dbus_init = 0;

/* you must hold spinlock_dbus_init before calling this! */
static void SDL_DBus_Init_Spinlocked(void)
{
    static SDL_bool is_dbus_available = SDL_TRUE;
    if (!is_dbus_available) {
        return; /* don't keep trying if this fails. */
    }

    if (!dbus.session_conn) {
        DBusError err;

        if (LoadDBUSLibrary() == -1) {
            is_dbus_available = SDL_FALSE; /* can't load at all? Don't keep trying. */
            return;
        }

        if (!dbus.threads_init_default()) {
            is_dbus_available = SDL_FALSE;
            return;
        }

        dbus.error_init(&err);
        /* session bus is required */

        dbus.session_conn = dbus.bus_get_private(DBUS_BUS_SESSION, &err);
        if (dbus.error_is_set(&err)) {
            dbus.error_free(&err);
            SDL_DBus_Quit();
            is_dbus_available = SDL_FALSE;
            return; /* oh well */
        }
        dbus.connection_set_exit_on_disconnect(dbus.session_conn, 0);

        /* system bus is optional */
        dbus.system_conn = dbus.bus_get_private(DBUS_BUS_SYSTEM, &err);
        if (!dbus.error_is_set(&err)) {
            dbus.connection_set_exit_on_disconnect(dbus.system_conn, 0);
        }

        dbus.error_free(&err);
    }
}

void SDL_DBus_Init(void)
{
    SDL_AtomicLock(&spinlock_dbus_init); /* make sure two threads can't init at same time, since this can happen before SDL_Init. */
    SDL_DBus_Init_Spinlocked();
    SDL_AtomicUnlock(&spinlock_dbus_init);
}

void SDL_DBus_Quit(void)
{
    if (dbus.system_conn) {
        dbus.connection_close(dbus.system_conn);
        dbus.connection_unref(dbus.system_conn);
    }
    if (dbus.session_conn) {
        dbus.connection_close(dbus.session_conn);
        dbus.connection_unref(dbus.session_conn);
    }

    if (SDL_GetHintBoolean(SDL_HINT_SHUTDOWN_DBUS_ON_QUIT, SDL_FALSE)) {
        if (dbus.shutdown) {
            dbus.shutdown();
        }
    }

    SDL_zero(dbus);
    UnloadDBUSLibrary();
    SDL_free(inhibit_handle);
    inhibit_handle = NULL;
}

SDL_DBusContext *SDL_DBus_GetContext(void)
{
    if (!dbus_handle || !dbus.session_conn) {
        SDL_DBus_Init();
    }

    return (dbus_handle && dbus.session_conn) ? &dbus : NULL;
}

static SDL_bool SDL_DBus_CallMethodInternal(DBusConnection *conn, const char *node, const char *path, const char *interface, const char *method, va_list ap)
{
    SDL_bool retval = SDL_FALSE;

    if (conn) {
        DBusMessage *msg = dbus.message_new_method_call(node, path, interface, method);
        if (msg) {
            int firstarg;
            va_list ap_reply;
            va_copy(ap_reply, ap); /* copy the arg list so we don't compete with D-Bus for it */
            firstarg = va_arg(ap, int);
            if ((firstarg == DBUS_TYPE_INVALID) || dbus.message_append_args_valist(msg, firstarg, ap)) {
                DBusMessage *reply = dbus.connection_send_with_reply_and_block(conn, msg, 300, NULL);
                if (reply) {
                    /* skip any input args, get to output args. */
                    while ((firstarg = va_arg(ap_reply, int)) != DBUS_TYPE_INVALID) {
                        /* we assume D-Bus already validated all this. */
                        {
                            void *dumpptr = va_arg(ap_reply, void *);
                            (void)dumpptr;
                        }
                        if (firstarg == DBUS_TYPE_ARRAY) {
                            {
                                const int dumpint = va_arg(ap_reply, int);
                                (void)dumpint;
                            }
                        }
                    }
                    firstarg = va_arg(ap_reply, int);
                    if ((firstarg == DBUS_TYPE_INVALID) || dbus.message_get_args_valist(reply, NULL, firstarg, ap_reply)) {
                        retval = SDL_TRUE;
                    }
                    dbus.message_unref(reply);
                }
            }
            va_end(ap_reply);
            dbus.message_unref(msg);
        }
    }

    return retval;
}

SDL_bool SDL_DBus_CallMethodOnConnection(DBusConnection *conn, const char *node, const char *path, const char *interface, const char *method, ...)
{
    SDL_bool retval;
    va_list ap;
    va_start(ap, method);
    retval = SDL_DBus_CallMethodInternal(conn, node, path, interface, method, ap);
    va_end(ap);
    return retval;
}

SDL_bool SDL_DBus_CallMethod(const char *node, const char *path, const char *interface, const char *method, ...)
{
    SDL_bool retval;
    va_list ap;
    va_start(ap, method);
    retval = SDL_DBus_CallMethodInternal(dbus.session_conn, node, path, interface, method, ap);
    va_end(ap);
    return retval;
}

static SDL_bool SDL_DBus_CallVoidMethodInternal(DBusConnection *conn, const char *node, const char *path, const char *interface, const char *method, va_list ap)
{
    SDL_bool retval = SDL_FALSE;

    if (conn) {
        DBusMessage *msg = dbus.message_new_method_call(node, path, interface, method);
        if (msg) {
            int firstarg = va_arg(ap, int);
            if ((firstarg == DBUS_TYPE_INVALID) || dbus.message_append_args_valist(msg, firstarg, ap)) {
                if (dbus.connection_send(conn, msg, NULL)) {
                    dbus.connection_flush(conn);
                    retval = SDL_TRUE;
                }
            }

            dbus.message_unref(msg);
        }
    }

    return retval;
}

static SDL_bool SDL_DBus_CallWithBasicReply(DBusConnection *conn, DBusMessage *msg, const int expectedtype, void *result)
{
    SDL_bool retval = SDL_FALSE;

    DBusMessage *reply = dbus.connection_send_with_reply_and_block(conn, msg, 300, NULL);
    if (reply) {
        DBusMessageIter iter, actual_iter;
        dbus.message_iter_init(reply, &iter);
        if (dbus.message_iter_get_arg_type(&iter) == DBUS_TYPE_VARIANT) {
            dbus.message_iter_recurse(&iter, &actual_iter);
        } else {
            actual_iter = iter;
        }

        if (dbus.message_iter_get_arg_type(&actual_iter) == expectedtype) {
            dbus.message_iter_get_basic(&actual_iter, result);
            retval = SDL_TRUE;
        }

        dbus.message_unref(reply);
    }

    return retval;
}

SDL_bool SDL_DBus_CallVoidMethodOnConnection(DBusConnection *conn, const char *node, const char *path, const char *interface, const char *method, ...)
{
    SDL_bool retval;
    va_list ap;
    va_start(ap, method);
    retval = SDL_DBus_CallVoidMethodInternal(conn, node, path, interface, method, ap);
    va_end(ap);
    return retval;
}

SDL_bool SDL_DBus_CallVoidMethod(const char *node, const char *path, const char *interface, const char *method, ...)
{
    SDL_bool retval;
    va_list ap;
    va_start(ap, method);
    retval = SDL_DBus_CallVoidMethodInternal(dbus.session_conn, node, path, interface, method, ap);
    va_end(ap);
    return retval;
}

SDL_bool SDL_DBus_QueryPropertyOnConnection(DBusConnection *conn, const char *node, const char *path, const char *interface, const char *property, const int expectedtype, void *result)
{
    SDL_bool retval = SDL_FALSE;

    if (conn) {
        DBusMessage *msg = dbus.message_new_method_call(node, path, "org.freedesktop.DBus.Properties", "Get");
        if (msg) {
            if (dbus.message_append_args(msg, DBUS_TYPE_STRING, &interface, DBUS_TYPE_STRING, &property, DBUS_TYPE_INVALID)) {
                retval = SDL_DBus_CallWithBasicReply(conn, msg, expectedtype, result);
            }
            dbus.message_unref(msg);
        }
    }

    return retval;
}

SDL_bool SDL_DBus_QueryProperty(const char *node, const char *path, const char *interface, const char *property, const int expectedtype, void *result)
{
    return SDL_DBus_QueryPropertyOnConnection(dbus.session_conn, node, path, interface, property, expectedtype, result);
}

void SDL_DBus_ScreensaverTickle(void)
{
    if (screensaver_cookie == 0 && !inhibit_handle) { /* no need to tickle if we're inhibiting. */
        /* org.gnome.ScreenSaver is the legacy interface, but it'll either do nothing or just be a second harmless tickle on newer systems, so we leave it for now. */
        SDL_DBus_CallVoidMethod("org.gnome.ScreenSaver", "/org/gnome/ScreenSaver", "org.gnome.ScreenSaver", "SimulateUserActivity", DBUS_TYPE_INVALID);
        SDL_DBus_CallVoidMethod("org.freedesktop.ScreenSaver", "/org/freedesktop/ScreenSaver", "org.freedesktop.ScreenSaver", "SimulateUserActivity", DBUS_TYPE_INVALID);
    }
}

static SDL_bool SDL_DBus_AppendDictWithKeyValue(DBusMessageIter *iterInit, const char *key, const char *value)
{
    DBusMessageIter iterDict, iterEntry, iterValue;

    if (!dbus.message_iter_open_container(iterInit, DBUS_TYPE_ARRAY, "{sv}", &iterDict)) {
        goto failed;
    }

    if (!dbus.message_iter_open_container(&iterDict, DBUS_TYPE_DICT_ENTRY, NULL, &iterEntry)) {
        goto failed;
    }

    if (!dbus.message_iter_append_basic(&iterEntry, DBUS_TYPE_STRING, &key)) {
        goto failed;
    }

    if (!dbus.message_iter_open_container(&iterEntry, DBUS_TYPE_VARIANT, DBUS_TYPE_STRING_AS_STRING, &iterValue)) {
        goto failed;
    }

    if (!dbus.message_iter_append_basic(&iterValue, DBUS_TYPE_STRING, &value)) {
        goto failed;
    }

    if (!dbus.message_iter_close_container(&iterEntry, &iterValue) || !dbus.message_iter_close_container(&iterDict, &iterEntry) || !dbus.message_iter_close_container(iterInit, &iterDict)) {
        goto failed;
    }

    return SDL_TRUE;

failed:
    /* message_iter_abandon_container_if_open() and message_iter_abandon_container() might be
     * missing if libdbus is too old. Instead, we just return without cleaning up any eventual
     * open container */
    return SDL_FALSE;
}

SDL_bool SDL_DBus_ScreensaverInhibit(SDL_bool inhibit)
{
    const char *default_inhibit_reason = "Playing a game";

    if ((inhibit && (screensaver_cookie != 0 || inhibit_handle)) || (!inhibit && (screensaver_cookie == 0 && !inhibit_handle))) {
        return SDL_TRUE;
    }

    if (!dbus.session_conn) {
        /* We either lost connection to the session bus or were not able to
         * load the D-Bus library at all. */
        return SDL_FALSE;
    }

    if (SDL_DetectSandbox() != SDL_SANDBOX_NONE) {
        const char *bus_name = "org.freedesktop.portal.Desktop";
        const char *path = "/org/freedesktop/portal/desktop";
        const char *interface = "org.freedesktop.portal.Inhibit";
        const char *window = "";                    /* As a future improvement we could gather the X11 XID or Wayland surface identifier */
        static const unsigned int INHIBIT_IDLE = 8; /* Taken from the portal API reference */
        DBusMessageIter iterInit;

        if (inhibit) {
            DBusMessage *msg;
            SDL_bool retval = SDL_FALSE;
            const char *key = "reason";
            const char *reply = NULL;
            const char *reason = SDL_GetHint(SDL_HINT_SCREENSAVER_INHIBIT_ACTIVITY_NAME);
            if (!reason || !reason[0]) {
                reason = default_inhibit_reason;
            }

            msg = dbus.message_new_method_call(bus_name, path, interface, "Inhibit");
            if (!msg) {
                return SDL_FALSE;
            }

            if (!dbus.message_append_args(msg, DBUS_TYPE_STRING, &window, DBUS_TYPE_UINT32, &INHIBIT_IDLE, DBUS_TYPE_INVALID)) {
                dbus.message_unref(msg);
                return SDL_FALSE;
            }

            dbus.message_iter_init_append(msg, &iterInit);
            if (!SDL_DBus_AppendDictWithKeyValue(&iterInit, key, reason)) {
                dbus.message_unref(msg);
                return SDL_FALSE;
            }

            if (SDL_DBus_CallWithBasicReply(dbus.session_conn, msg, DBUS_TYPE_OBJECT_PATH, &reply)) {
                inhibit_handle = SDL_strdup(reply);
                retval = SDL_TRUE;
            }

            dbus.message_unref(msg);
            return retval;
        } else {
            if (!SDL_DBus_CallVoidMethod(bus_name, inhibit_handle, "org.freedesktop.portal.Request", "Close", DBUS_TYPE_INVALID)) {
                return SDL_FALSE;
            }
            SDL_free(inhibit_handle);
            inhibit_handle = NULL;
        }
    } else {
        const char *bus_name = "org.freedesktop.ScreenSaver";
        const char *path = "/org/freedesktop/ScreenSaver";
        const char *interface = "org.freedesktop.ScreenSaver";

        if (inhibit) {
            const char *app = SDL_GetHint(SDL_HINT_APP_NAME);
            const char *reason = SDL_GetHint(SDL_HINT_SCREENSAVER_INHIBIT_ACTIVITY_NAME);
            if (!app || !app[0]) {
                app = "My SDL application";
            }
            if (!reason || !reason[0]) {
                reason = default_inhibit_reason;
            }

            if (!SDL_DBus_CallMethod(bus_name, path, interface, "Inhibit",
                                     DBUS_TYPE_STRING, &app, DBUS_TYPE_STRING, &reason, DBUS_TYPE_INVALID,
                                     DBUS_TYPE_UINT32, &screensaver_cookie, DBUS_TYPE_INVALID)) {
                return SDL_FALSE;
            }
            return (screensaver_cookie != 0) ? SDL_TRUE : SDL_FALSE;
        } else {
            if (!SDL_DBus_CallVoidMethod(bus_name, path, interface, "UnInhibit", DBUS_TYPE_UINT32, &screensaver_cookie, DBUS_TYPE_INVALID)) {
                return SDL_FALSE;
            }
            screensaver_cookie = 0;
        }
    }

    return SDL_TRUE;
}

/*
 * Get the machine ID if possible. Result must be freed with dbus->free().
 */
char *SDL_DBus_GetLocalMachineId(void)
{
    DBusError err;
    char *result;

    dbus.error_init(&err);

    if (dbus.try_get_local_machine_id) {
        /* Available since dbus 1.12.0, has proper error-handling */
        result = dbus.try_get_local_machine_id(&err);
    } else {
        /* Available since time immemorial, but has no error-handling:
         * if the machine ID can't be read, many versions of libdbus will
         * treat that as a fatal mis-installation and abort() */
        result = dbus.get_local_machine_id();
    }

    if (result) {
        return result;
    }

    if (dbus.error_is_set(&err)) {
        SDL_SetError("%s: %s", err.name, err.message);
        dbus.error_free(&err);
    } else {
        SDL_SetError("Error getting D-Bus machine ID");
    }

    return NULL;
}

/*
 * Convert file drops with mime type "application/vnd.portal.filetransfer" to file paths
 * Result must be freed with dbus->free_string_array().
 * https://flatpak.github.io/xdg-desktop-portal/#gdbus-method-org-freedesktop-portal-FileTransfer.RetrieveFiles
 */
char **SDL_DBus_DocumentsPortalRetrieveFiles(const char *key, int *path_count)
{
    DBusError err;
    DBusMessageIter iter, iterDict;
    char **paths = NULL;
    DBusMessage *reply = NULL;
    DBusMessage *msg = dbus.message_new_method_call("org.freedesktop.portal.Documents",    /* Node */
                                                    "/org/freedesktop/portal/documents",   /* Path */
                                                    "org.freedesktop.portal.FileTransfer", /* Interface */
                                                    "RetrieveFiles");                      /* Method */

    /* Make sure we have a connection to the dbus session bus */
    if (!SDL_DBus_GetContext() || !dbus.session_conn) {
        /* We either cannot connect to the session bus or were unable to
         * load the D-Bus library at all. */
        return NULL;
    }

    dbus.error_init(&err);

    /* First argument is a "application/vnd.portal.filetransfer" key from a DnD or clipboard event */
    if (!dbus.message_append_args(msg, DBUS_TYPE_STRING, &key, DBUS_TYPE_INVALID)) {
        SDL_OutOfMemory();
        dbus.message_unref(msg);
        goto failed;
    }

    /* Second argument is a variant dictionary for options.
     * The spec doesn't define any entries yet so it's empty. */
    dbus.message_iter_init_append(msg, &iter);
    if (!dbus.message_iter_open_container(&iter, DBUS_TYPE_ARRAY, "{sv}", &iterDict) ||
        !dbus.message_iter_close_container(&iter,  &iterDict)) {
        SDL_OutOfMemory();
        dbus.message_unref(msg);
        goto failed;
    }

    reply = dbus.connection_send_with_reply_and_block(dbus.session_conn, msg, DBUS_TIMEOUT_USE_DEFAULT, &err);
    dbus.message_unref(msg);

    if (reply) {
        dbus.message_get_args(reply, &err, DBUS_TYPE_ARRAY, DBUS_TYPE_STRING, &paths, path_count, DBUS_TYPE_INVALID);
        dbus.message_unref(reply);
    }

    if (paths) {
        return paths;
    }

failed:
    if (dbus.error_is_set(&err)) {
        SDL_SetError("%s: %s", err.name, err.message);
        dbus.error_free(&err);
    } else {
        SDL_SetError("Error retrieving paths for documents portal \"%s\"", key);
    }

    return NULL;
}
#endif

/* vi: set ts=4 sw=4 expandtab: */
