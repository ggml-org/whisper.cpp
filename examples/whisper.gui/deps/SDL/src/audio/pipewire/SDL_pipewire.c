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

#ifdef SDL_AUDIO_DRIVER_PIPEWIRE

#include "SDL_audio.h"
#include "SDL_loadso.h"
#include "SDL_pipewire.h"

#include <pipewire/extensions/metadata.h>
#include <spa/param/audio/format-utils.h>
#include <spa/utils/json.h>

/*
 * The following keys are defined for compatability when building against older versions of Pipewire
 * prior to their introduction and can be removed if the minimum required Pipewire build version is
 * increased to or beyond their point of introduction.
 */

/*
 * Introduced in 0.3.22
 * Taken from /src/pipewire/keys.h
 */
#ifndef PW_KEY_CONFIG_NAME
#define PW_KEY_CONFIG_NAME "config.name"
#endif

/*
 * Introduced in 0.3.33
 * Taken from src/pipewire/keys.h
 */
#ifndef PW_KEY_NODE_RATE
#define PW_KEY_NODE_RATE "node.rate"
#endif

/*
 * Introduced in 0.3.44
 * Taken from src/pipewire/keys.h
 */
#ifndef PW_KEY_TARGET_OBJECT
#define PW_KEY_TARGET_OBJECT "target.object"
#endif

/*
 * This seems to be a sane lower limit as Pipewire
 * uses it in several of it's own modules.
 */
#define PW_MIN_SAMPLES     32 /* About 0.67ms at 48kHz */
#define PW_BASE_CLOCK_RATE 48000

#define PW_POD_BUFFER_LENGTH         1024
#define PW_THREAD_NAME_BUFFER_LENGTH 128
#define PW_MAX_IDENTIFIER_LENGTH     256

enum PW_READY_FLAGS
{
    PW_READY_FLAG_BUFFER_ADDED = 0x1,
    PW_READY_FLAG_STREAM_READY = 0x2,
    PW_READY_FLAG_ALL_BITS = 0x3
};

#define PW_ID_TO_HANDLE(x) (void *)((uintptr_t)x)
#define PW_HANDLE_TO_ID(x) (uint32_t)((uintptr_t)x)

static SDL_bool pipewire_initialized = SDL_FALSE;

/* Pipewire entry points */
static const char *(*PIPEWIRE_pw_get_library_version)(void);
static void (*PIPEWIRE_pw_init)(int *, char ***);
static void (*PIPEWIRE_pw_deinit)(void);
static struct pw_thread_loop *(*PIPEWIRE_pw_thread_loop_new)(const char *, const struct spa_dict *);
static void (*PIPEWIRE_pw_thread_loop_destroy)(struct pw_thread_loop *);
static void (*PIPEWIRE_pw_thread_loop_stop)(struct pw_thread_loop *);
static struct pw_loop *(*PIPEWIRE_pw_thread_loop_get_loop)(struct pw_thread_loop *);
static void (*PIPEWIRE_pw_thread_loop_lock)(struct pw_thread_loop *);
static void (*PIPEWIRE_pw_thread_loop_unlock)(struct pw_thread_loop *);
static void (*PIPEWIRE_pw_thread_loop_signal)(struct pw_thread_loop *, bool);
static void (*PIPEWIRE_pw_thread_loop_wait)(struct pw_thread_loop *);
static int (*PIPEWIRE_pw_thread_loop_start)(struct pw_thread_loop *);
static struct pw_context *(*PIPEWIRE_pw_context_new)(struct pw_loop *, struct pw_properties *, size_t);
static void (*PIPEWIRE_pw_context_destroy)(struct pw_context *);
static struct pw_core *(*PIPEWIRE_pw_context_connect)(struct pw_context *, struct pw_properties *, size_t);
static void (*PIPEWIRE_pw_proxy_add_object_listener)(struct pw_proxy *, struct spa_hook *, const void *, void *);
static void *(*PIPEWIRE_pw_proxy_get_user_data)(struct pw_proxy *);
static void (*PIPEWIRE_pw_proxy_destroy)(struct pw_proxy *);
static int (*PIPEWIRE_pw_core_disconnect)(struct pw_core *);
static struct pw_stream *(*PIPEWIRE_pw_stream_new_simple)(struct pw_loop *, const char *, struct pw_properties *,
                                                          const struct pw_stream_events *, void *);
static void (*PIPEWIRE_pw_stream_destroy)(struct pw_stream *);
static int (*PIPEWIRE_pw_stream_connect)(struct pw_stream *, enum pw_direction, uint32_t, enum pw_stream_flags,
                                         const struct spa_pod **, uint32_t);
static enum pw_stream_state (*PIPEWIRE_pw_stream_get_state)(struct pw_stream *stream, const char **error);
static struct pw_buffer *(*PIPEWIRE_pw_stream_dequeue_buffer)(struct pw_stream *);
static int (*PIPEWIRE_pw_stream_queue_buffer)(struct pw_stream *, struct pw_buffer *);
static struct pw_properties *(*PIPEWIRE_pw_properties_new)(const char *, ...)SPA_SENTINEL;
static int (*PIPEWIRE_pw_properties_set)(struct pw_properties *, const char *, const char *);
static int (*PIPEWIRE_pw_properties_setf)(struct pw_properties *, const char *, const char *, ...) SPA_PRINTF_FUNC(3, 4);

static int pipewire_version_major;
static int pipewire_version_minor;
static int pipewire_version_patch;

#ifdef SDL_AUDIO_DRIVER_PIPEWIRE_DYNAMIC

static const char *pipewire_library = SDL_AUDIO_DRIVER_PIPEWIRE_DYNAMIC;
static void *pipewire_handle = NULL;

static int pipewire_dlsym(const char *fn, void **addr)
{
    *addr = SDL_LoadFunction(pipewire_handle, fn);
    if (!*addr) {
        /* Don't call SDL_SetError(): SDL_LoadFunction already did. */
        return 0;
    }

    return 1;
}

#define SDL_PIPEWIRE_SYM(x)                                    \
    if (!pipewire_dlsym(#x, (void **)(char *)&PIPEWIRE_##x)) { \
        return -1;                                             \
    }

static int load_pipewire_library(void)
{
    pipewire_handle = SDL_LoadObject(pipewire_library);
    return pipewire_handle ? 0 : -1;
}

static void unload_pipewire_library(void)
{
    if (pipewire_handle) {
        SDL_UnloadObject(pipewire_handle);
        pipewire_handle = NULL;
    }
}

#else

#define SDL_PIPEWIRE_SYM(x) PIPEWIRE_##x = x

static int load_pipewire_library(void)
{
    return 0;
}

static void unload_pipewire_library(void)
{ /* Nothing to do */
}

#endif /* SDL_AUDIO_DRIVER_PIPEWIRE_DYNAMIC */

static int load_pipewire_syms(void)
{
    SDL_PIPEWIRE_SYM(pw_get_library_version);
    SDL_PIPEWIRE_SYM(pw_init);
    SDL_PIPEWIRE_SYM(pw_deinit);
    SDL_PIPEWIRE_SYM(pw_thread_loop_new);
    SDL_PIPEWIRE_SYM(pw_thread_loop_destroy);
    SDL_PIPEWIRE_SYM(pw_thread_loop_stop);
    SDL_PIPEWIRE_SYM(pw_thread_loop_get_loop);
    SDL_PIPEWIRE_SYM(pw_thread_loop_lock);
    SDL_PIPEWIRE_SYM(pw_thread_loop_unlock);
    SDL_PIPEWIRE_SYM(pw_thread_loop_signal);
    SDL_PIPEWIRE_SYM(pw_thread_loop_wait);
    SDL_PIPEWIRE_SYM(pw_thread_loop_start);
    SDL_PIPEWIRE_SYM(pw_context_new);
    SDL_PIPEWIRE_SYM(pw_context_destroy);
    SDL_PIPEWIRE_SYM(pw_context_connect);
    SDL_PIPEWIRE_SYM(pw_proxy_add_object_listener);
    SDL_PIPEWIRE_SYM(pw_proxy_get_user_data);
    SDL_PIPEWIRE_SYM(pw_proxy_destroy);
    SDL_PIPEWIRE_SYM(pw_core_disconnect);
    SDL_PIPEWIRE_SYM(pw_stream_new_simple);
    SDL_PIPEWIRE_SYM(pw_stream_destroy);
    SDL_PIPEWIRE_SYM(pw_stream_connect);
    SDL_PIPEWIRE_SYM(pw_stream_get_state);
    SDL_PIPEWIRE_SYM(pw_stream_dequeue_buffer);
    SDL_PIPEWIRE_SYM(pw_stream_queue_buffer);
    SDL_PIPEWIRE_SYM(pw_properties_new);
    SDL_PIPEWIRE_SYM(pw_properties_set);
    SDL_PIPEWIRE_SYM(pw_properties_setf);

    return 0;
}

SDL_FORCE_INLINE SDL_bool pipewire_version_at_least(int major, int minor, int patch)
{
    return (pipewire_version_major >= major) &&
           (pipewire_version_major > major || pipewire_version_minor >= minor) &&
           (pipewire_version_major > major || pipewire_version_minor > minor || pipewire_version_patch >= patch);
}

static int init_pipewire_library(void)
{
    if (!load_pipewire_library()) {
        if (!load_pipewire_syms()) {
            int nargs;
            const char *version = PIPEWIRE_pw_get_library_version();
            nargs = SDL_sscanf(version, "%d.%d.%d", &pipewire_version_major, &pipewire_version_minor, &pipewire_version_patch);
            if (nargs < 3) {
                return -1;
            }

            /* SDL can build against 0.3.20, but requires 0.3.24 */
            if (pipewire_version_at_least(0, 3, 24)) {
                PIPEWIRE_pw_init(NULL, NULL);
                return 0;
            }
        }
    }

    return -1;
}

static void deinit_pipewire_library(void)
{
    PIPEWIRE_pw_deinit();
    unload_pipewire_library();
}

/* A generic Pipewire node object used for enumeration. */
struct node_object
{
    struct spa_list link;

    Uint32 id;
    int seq;
    SDL_bool persist;

    /*
     * NOTE: If used, this is *must* be allocated with SDL_malloc() or similar
     * as SDL_free() will be called on it when the node_object is destroyed.
     *
     * If ownership of the referenced memory is transferred, this must be set
     * to NULL or the memory will be freed when the node_object is destroyed.
     */
    void *userdata;

    struct pw_proxy *proxy;
    struct spa_hook node_listener;
    struct spa_hook core_listener;
};

/* A sink/source node used for stream I/O. */
struct io_node
{
    struct spa_list link;

    Uint32 id;
    SDL_bool is_capture;
    SDL_AudioSpec spec;

    const char *name; /* Friendly name */
    const char *path; /* OS identifier (i.e. ALSA endpoint) */

    char buf[]; /* Buffer to hold the name and path strings. */
};

/* The global hotplug thread and associated objects. */
static struct pw_thread_loop *hotplug_loop;
static struct pw_core *hotplug_core;
static struct pw_context *hotplug_context;
static struct pw_registry *hotplug_registry;
static struct spa_hook hotplug_registry_listener;
static struct spa_hook hotplug_core_listener;
static struct spa_list hotplug_pending_list;
static struct spa_list hotplug_io_list;
static int hotplug_init_seq_val;
static SDL_bool hotplug_init_complete;
static SDL_bool hotplug_events_enabled;

static char *pipewire_default_sink_id = NULL;
static char *pipewire_default_source_id = NULL;

/* The active node list */
static SDL_bool io_list_check_add(struct io_node *node)
{
    struct io_node *n;
    SDL_bool ret = SDL_TRUE;

    /* See if the node is already in the list */
    spa_list_for_each (n, &hotplug_io_list, link) {
        if (n->id == node->id) {
            ret = SDL_FALSE;
            goto dup_found;
        }
    }

    /* Add to the list if the node doesn't already exist */
    spa_list_append(&hotplug_io_list, &node->link);

    if (hotplug_events_enabled) {
        SDL_AddAudioDevice(node->is_capture, node->name, &node->spec, PW_ID_TO_HANDLE(node->id));
    }

dup_found:

    return ret;
}

static void io_list_remove(Uint32 id)
{
    struct io_node *n, *temp;

    /* Find and remove the node from the list */
    spa_list_for_each_safe (n, temp, &hotplug_io_list, link) {
        if (n->id == id) {
            spa_list_remove(&n->link);

            if (hotplug_events_enabled) {
                SDL_RemoveAudioDevice(n->is_capture, PW_ID_TO_HANDLE(id));
            }

            SDL_free(n);

            break;
        }
    }
}

static void io_list_sort(void)
{
    struct io_node *default_sink = NULL, *default_source = NULL;
    struct io_node *n, *temp;

    /* Find and move the default nodes to the beginning of the list */
    spa_list_for_each_safe (n, temp, &hotplug_io_list, link) {
        if (pipewire_default_sink_id && SDL_strcmp(n->path, pipewire_default_sink_id) == 0) {
            default_sink = n;
            spa_list_remove(&n->link);
        } else if (pipewire_default_source_id && SDL_strcmp(n->path, pipewire_default_source_id) == 0) {
            default_source = n;
            spa_list_remove(&n->link);
        }
    }

    if (default_source) {
        spa_list_prepend(&hotplug_io_list, &default_source->link);
    }

    if (default_sink) {
        spa_list_prepend(&hotplug_io_list, &default_sink->link);
    }
}

static void io_list_clear(void)
{
    struct io_node *n, *temp;

    spa_list_for_each_safe (n, temp, &hotplug_io_list, link) {
        spa_list_remove(&n->link);
        SDL_free(n);
    }
}

static struct io_node *io_list_get_by_id(Uint32 id)
{
    struct io_node *n, *temp;
    spa_list_for_each_safe (n, temp, &hotplug_io_list, link) {
        if (n->id == id) {
            return n;
        }
    }
    return NULL;
}

static struct io_node *io_list_get_by_path(char *path)
{
    struct io_node *n, *temp;
    spa_list_for_each_safe (n, temp, &hotplug_io_list, link) {
        if (SDL_strcmp(n->path, path) == 0) {
            return n;
        }
    }
    return NULL;
}

static void node_object_destroy(struct node_object *node)
{
    SDL_assert(node != NULL);

    spa_list_remove(&node->link);
    spa_hook_remove(&node->node_listener);
    spa_hook_remove(&node->core_listener);
    SDL_free(node->userdata);
    PIPEWIRE_pw_proxy_destroy(node->proxy);
}

/* The pending node list */
static void pending_list_add(struct node_object *node)
{
    SDL_assert(node != NULL);
    spa_list_append(&hotplug_pending_list, &node->link);
}

static void pending_list_remove(Uint32 id)
{
    struct node_object *node, *temp;

    spa_list_for_each_safe (node, temp, &hotplug_pending_list, link) {
        if (node->id == id) {
            node_object_destroy(node);
        }
    }
}

static void pending_list_clear(void)
{
    struct node_object *node, *temp;

    spa_list_for_each_safe (node, temp, &hotplug_pending_list, link) {
        node_object_destroy(node);
    }
}

static void *node_object_new(Uint32 id, const char *type, Uint32 version, const void *funcs, const struct pw_core_events *core_events)
{
    struct pw_proxy *proxy;
    struct node_object *node;

    /* Create the proxy object */
    proxy = pw_registry_bind(hotplug_registry, id, type, version, sizeof(struct node_object));
    if (!proxy) {
        SDL_SetError("Pipewire: Failed to create proxy object (%i)", errno);
        return NULL;
    }

    node = PIPEWIRE_pw_proxy_get_user_data(proxy);
    SDL_zerop(node);

    node->id = id;
    node->proxy = proxy;

    /* Add the callbacks */
    pw_core_add_listener(hotplug_core, &node->core_listener, core_events, node);
    PIPEWIRE_pw_proxy_add_object_listener(node->proxy, &node->node_listener, funcs, node);

    /* Add the node to the active list */
    pending_list_add(node);

    return node;
}

/* Core sync points */
static void core_events_hotplug_init_callback(void *object, uint32_t id, int seq)
{
    if (id == PW_ID_CORE && seq == hotplug_init_seq_val) {
        /* This core listener is no longer needed. */
        spa_hook_remove(&hotplug_core_listener);

        /* Signal that the initial I/O list is populated */
        hotplug_init_complete = SDL_TRUE;
        PIPEWIRE_pw_thread_loop_signal(hotplug_loop, false);
    }
}

static void core_events_interface_callback(void *object, uint32_t id, int seq)
{
    struct node_object *node = object;
    struct io_node *io = node->userdata;

    if (id == PW_ID_CORE && seq == node->seq) {
        /*
         * Move the I/O node to the connected list.
         * On success, the list owns the I/O node object.
         */
        if (io_list_check_add(io)) {
            node->userdata = NULL;
        }

        node_object_destroy(node);
    }
}

static void core_events_metadata_callback(void *object, uint32_t id, int seq)
{
    struct node_object *node = object;

    if (id == PW_ID_CORE && seq == node->seq && !node->persist) {
        node_object_destroy(node);
    }
}

static const struct pw_core_events hotplug_init_core_events = { PW_VERSION_CORE_EVENTS, .done = core_events_hotplug_init_callback };
static const struct pw_core_events interface_core_events = { PW_VERSION_CORE_EVENTS, .done = core_events_interface_callback };
static const struct pw_core_events metadata_core_events = { PW_VERSION_CORE_EVENTS, .done = core_events_metadata_callback };

static void hotplug_core_sync(struct node_object *node)
{
    /*
     * Node sync events *must* come before the hotplug init sync events or the initial
     * I/O list will be incomplete when the main hotplug sync point is hit.
     */
    if (node) {
        node->seq = pw_core_sync(hotplug_core, PW_ID_CORE, node->seq);
    }

    if (!hotplug_init_complete) {
        hotplug_init_seq_val = pw_core_sync(hotplug_core, PW_ID_CORE, hotplug_init_seq_val);
    }
}

/* Helpers for retrieving values from params */
static SDL_bool get_range_param(const struct spa_pod *param, Uint32 key, int *def, int *min, int *max)
{
    const struct spa_pod_prop *prop;
    struct spa_pod *value;
    Uint32 n_values, choice;

    prop = spa_pod_find_prop(param, NULL, key);

    if (prop && prop->value.type == SPA_TYPE_Choice) {
        value = spa_pod_get_values(&prop->value, &n_values, &choice);

        if (n_values == 3 && choice == SPA_CHOICE_Range) {
            Uint32 *v = SPA_POD_BODY(value);

            if (v) {
                if (def) {
                    *def = (int)v[0];
                }
                if (min) {
                    *min = (int)v[1];
                }
                if (max) {
                    *max = (int)v[2];
                }

                return SDL_TRUE;
            }
        }
    }

    return SDL_FALSE;
}

static SDL_bool get_int_param(const struct spa_pod *param, Uint32 key, int *val)
{
    const struct spa_pod_prop *prop;
    Sint32 v;

    prop = spa_pod_find_prop(param, NULL, key);

    if (prop && spa_pod_get_int(&prop->value, &v) == 0) {
        if (val) {
            *val = (int)v;
        }

        return SDL_TRUE;
    }

    return SDL_FALSE;
}

/* Interface node callbacks */
static void node_event_info(void *object, const struct pw_node_info *info)
{
    struct node_object *node = object;
    struct io_node *io = node->userdata;
    const char *prop_val;
    Uint32 i;

    if (info) {
        prop_val = spa_dict_lookup(info->props, PW_KEY_AUDIO_CHANNELS);
        if (prop_val) {
            io->spec.channels = (Uint8)SDL_atoi(prop_val);
        }

        /* Need to parse the parameters to get the sample rate */
        for (i = 0; i < info->n_params; ++i) {
            pw_node_enum_params(node->proxy, 0, info->params[i].id, 0, 0, NULL);
        }

        hotplug_core_sync(node);
    }
}

static void node_event_param(void *object, int seq, uint32_t id, uint32_t index, uint32_t next, const struct spa_pod *param)
{
    struct node_object *node = object;
    struct io_node *io = node->userdata;

    /* Get the default frequency */
    if (io->spec.freq == 0) {
        get_range_param(param, SPA_FORMAT_AUDIO_rate, &io->spec.freq, NULL, NULL);
    }

    /*
     * The channel count should have come from the node properties,
     * but it is stored here as well. If one failed, try the other.
     */
    if (io->spec.channels == 0) {
        int channels;
        if (get_int_param(param, SPA_FORMAT_AUDIO_channels, &channels)) {
            io->spec.channels = (Uint8)channels;
        }
    }
}

static const struct pw_node_events interface_node_events = { PW_VERSION_NODE_EVENTS, .info = node_event_info,
                                                             .param = node_event_param };

static char *get_name_from_json(const char *json)
{
    struct spa_json parser[2];
    char key[7]; /* "name" */
    char value[PW_MAX_IDENTIFIER_LENGTH];
    spa_json_init(&parser[0], json, SDL_strlen(json));
    if (spa_json_enter_object(&parser[0], &parser[1]) <= 0) {
        /* Not actually JSON */
        return NULL;
    }
    if (spa_json_get_string(&parser[1], key, sizeof(key)) <= 0) {
        /* Not actually a key/value pair */
        return NULL;
    }
    if (spa_json_get_string(&parser[1], value, sizeof(value)) <= 0) {
        /* Somehow had a key with no value? */
        return NULL;
    }
    return SDL_strdup(value);
}

/* Metadata node callback */
static int metadata_property(void *object, Uint32 subject, const char *key, const char *type, const char *value)
{
    struct node_object *node = object;

    if (subject == PW_ID_CORE && key && value) {
        if (!SDL_strcmp(key, "default.audio.sink")) {
            if (pipewire_default_sink_id) {
                SDL_free(pipewire_default_sink_id);
            }
            pipewire_default_sink_id = get_name_from_json(value);
            node->persist = SDL_TRUE;
        } else if (!SDL_strcmp(key, "default.audio.source")) {
            if (pipewire_default_source_id) {
                SDL_free(pipewire_default_source_id);
            }
            pipewire_default_source_id = get_name_from_json(value);
            node->persist = SDL_TRUE;
        }
    }

    return 0;
}

static const struct pw_metadata_events metadata_node_events = { PW_VERSION_METADATA_EVENTS, .property = metadata_property };

/* Global registry callbacks */
static void registry_event_global_callback(void *object, uint32_t id, uint32_t permissions, const char *type, uint32_t version,
                                           const struct spa_dict *props)
{
    struct node_object *node;

    /* We're only interested in interface and metadata nodes. */
    if (!SDL_strcmp(type, PW_TYPE_INTERFACE_Node)) {
        const char *media_class = spa_dict_lookup(props, PW_KEY_MEDIA_CLASS);

        if (media_class) {
            const char *node_desc;
            const char *node_path;
            struct io_node *io;
            SDL_bool is_capture;
            int desc_buffer_len;
            int path_buffer_len;

            /* Just want sink and capture */
            if (!SDL_strcasecmp(media_class, "Audio/Sink")) {
                is_capture = SDL_FALSE;
            } else if (!SDL_strcasecmp(media_class, "Audio/Source")) {
                is_capture = SDL_TRUE;
            } else {
                return;
            }

            node_desc = spa_dict_lookup(props, PW_KEY_NODE_DESCRIPTION);
            node_path = spa_dict_lookup(props, PW_KEY_NODE_NAME);

            if (node_desc && node_path) {
                node = node_object_new(id, type, version, &interface_node_events, &interface_core_events);
                if (!node) {
                    SDL_SetError("Pipewire: Failed to allocate interface node");
                    return;
                }

                /* Allocate and initialize the I/O node information struct */
                desc_buffer_len = SDL_strlen(node_desc) + 1;
                path_buffer_len = SDL_strlen(node_path) + 1;
                node->userdata = io = SDL_calloc(1, sizeof(struct io_node) + desc_buffer_len + path_buffer_len);
                if (!io) {
                    node_object_destroy(node);
                    SDL_OutOfMemory();
                    return;
                }

                /* Begin setting the node properties */
                io->id = id;
                io->is_capture = is_capture;
                io->spec.format = AUDIO_F32; /* Pipewire uses floats internally, other formats require conversion. */
                io->name = io->buf;
                io->path = io->buf + desc_buffer_len;
                SDL_strlcpy(io->buf, node_desc, desc_buffer_len);
                SDL_strlcpy(io->buf + desc_buffer_len, node_path, path_buffer_len);

                /* Update sync points */
                hotplug_core_sync(node);
            }
        }
    } else if (!SDL_strcmp(type, PW_TYPE_INTERFACE_Metadata)) {
        node = node_object_new(id, type, version, &metadata_node_events, &metadata_core_events);
        if (!node) {
            SDL_SetError("Pipewire: Failed to allocate metadata node");
            return;
        }

        /* Update sync points */
        hotplug_core_sync(node);
    }
}

static void registry_event_remove_callback(void *object, uint32_t id)
{
    io_list_remove(id);
    pending_list_remove(id);
}

static const struct pw_registry_events registry_events = { PW_VERSION_REGISTRY_EVENTS, .global = registry_event_global_callback,
                                                           .global_remove = registry_event_remove_callback };

/* The hotplug thread */
static int hotplug_loop_init(void)
{
    int res;

    spa_list_init(&hotplug_pending_list);
    spa_list_init(&hotplug_io_list);

    hotplug_loop = PIPEWIRE_pw_thread_loop_new("SDLAudioHotplug", NULL);
    if (!hotplug_loop) {
        return SDL_SetError("Pipewire: Failed to create hotplug detection loop (%i)", errno);
    }

    hotplug_context = PIPEWIRE_pw_context_new(PIPEWIRE_pw_thread_loop_get_loop(hotplug_loop), NULL, 0);
    if (!hotplug_context) {
        return SDL_SetError("Pipewire: Failed to create hotplug detection context (%i)", errno);
    }

    hotplug_core = PIPEWIRE_pw_context_connect(hotplug_context, NULL, 0);
    if (!hotplug_core) {
        return SDL_SetError("Pipewire: Failed to connect hotplug detection context (%i)", errno);
    }

    hotplug_registry = pw_core_get_registry(hotplug_core, PW_VERSION_REGISTRY, 0);
    if (!hotplug_registry) {
        return SDL_SetError("Pipewire: Failed to acquire hotplug detection registry (%i)", errno);
    }

    spa_zero(hotplug_registry_listener);
    pw_registry_add_listener(hotplug_registry, &hotplug_registry_listener, &registry_events, NULL);

    spa_zero(hotplug_core_listener);
    pw_core_add_listener(hotplug_core, &hotplug_core_listener, &hotplug_init_core_events, NULL);

    hotplug_init_seq_val = pw_core_sync(hotplug_core, PW_ID_CORE, 0);

    res = PIPEWIRE_pw_thread_loop_start(hotplug_loop);
    if (res != 0) {
        return SDL_SetError("Pipewire: Failed to start hotplug detection loop");
    }

    return 0;
}

static void hotplug_loop_destroy(void)
{
    if (hotplug_loop) {
        PIPEWIRE_pw_thread_loop_stop(hotplug_loop);
    }

    pending_list_clear();
    io_list_clear();

    hotplug_init_complete = SDL_FALSE;
    hotplug_events_enabled = SDL_FALSE;

    if (pipewire_default_sink_id) {
        SDL_free(pipewire_default_sink_id);
        pipewire_default_sink_id = NULL;
    }
    if (pipewire_default_source_id) {
        SDL_free(pipewire_default_source_id);
        pipewire_default_source_id = NULL;
    }

    if (hotplug_registry) {
        PIPEWIRE_pw_proxy_destroy((struct pw_proxy *)hotplug_registry);
        hotplug_registry = NULL;
    }

    if (hotplug_core) {
        PIPEWIRE_pw_core_disconnect(hotplug_core);
        hotplug_core = NULL;
    }

    if (hotplug_context) {
        PIPEWIRE_pw_context_destroy(hotplug_context);
        hotplug_context = NULL;
    }

    if (hotplug_loop) {
        PIPEWIRE_pw_thread_loop_destroy(hotplug_loop);
        hotplug_loop = NULL;
    }
}

static void PIPEWIRE_DetectDevices(void)
{
    struct io_node *io;

    PIPEWIRE_pw_thread_loop_lock(hotplug_loop);

    /* Wait until the initial registry enumeration is complete */
    if (!hotplug_init_complete) {
        PIPEWIRE_pw_thread_loop_wait(hotplug_loop);
    }

    /* Sort the I/O list so the default source/sink are listed first */
    io_list_sort();

    spa_list_for_each (io, &hotplug_io_list, link) {
        SDL_AddAudioDevice(io->is_capture, io->name, &io->spec, PW_ID_TO_HANDLE(io->id));
    }

    hotplug_events_enabled = SDL_TRUE;

    PIPEWIRE_pw_thread_loop_unlock(hotplug_loop);
}

/* Channel maps that match the order in SDL_Audio.h */
static const enum spa_audio_channel PIPEWIRE_channel_map_1[] = { SPA_AUDIO_CHANNEL_MONO };
static const enum spa_audio_channel PIPEWIRE_channel_map_2[] = { SPA_AUDIO_CHANNEL_FL, SPA_AUDIO_CHANNEL_FR };
static const enum spa_audio_channel PIPEWIRE_channel_map_3[] = { SPA_AUDIO_CHANNEL_FL, SPA_AUDIO_CHANNEL_FR, SPA_AUDIO_CHANNEL_LFE };
static const enum spa_audio_channel PIPEWIRE_channel_map_4[] = { SPA_AUDIO_CHANNEL_FL, SPA_AUDIO_CHANNEL_FR, SPA_AUDIO_CHANNEL_RL,
                                                                 SPA_AUDIO_CHANNEL_RR };
static const enum spa_audio_channel PIPEWIRE_channel_map_5[] = { SPA_AUDIO_CHANNEL_FL, SPA_AUDIO_CHANNEL_FR, SPA_AUDIO_CHANNEL_FC,
                                                                 SPA_AUDIO_CHANNEL_RL, SPA_AUDIO_CHANNEL_RR };
static const enum spa_audio_channel PIPEWIRE_channel_map_6[] = { SPA_AUDIO_CHANNEL_FL, SPA_AUDIO_CHANNEL_FR, SPA_AUDIO_CHANNEL_FC,
                                                                 SPA_AUDIO_CHANNEL_LFE, SPA_AUDIO_CHANNEL_RL, SPA_AUDIO_CHANNEL_RR };
static const enum spa_audio_channel PIPEWIRE_channel_map_7[] = { SPA_AUDIO_CHANNEL_FL, SPA_AUDIO_CHANNEL_FR, SPA_AUDIO_CHANNEL_FC,
                                                                 SPA_AUDIO_CHANNEL_LFE, SPA_AUDIO_CHANNEL_RC, SPA_AUDIO_CHANNEL_RL,
                                                                 SPA_AUDIO_CHANNEL_RR };
static const enum spa_audio_channel PIPEWIRE_channel_map_8[] = { SPA_AUDIO_CHANNEL_FL, SPA_AUDIO_CHANNEL_FR, SPA_AUDIO_CHANNEL_FC,
                                                                 SPA_AUDIO_CHANNEL_LFE, SPA_AUDIO_CHANNEL_RL, SPA_AUDIO_CHANNEL_RR,
                                                                 SPA_AUDIO_CHANNEL_SL, SPA_AUDIO_CHANNEL_SR };

#define COPY_CHANNEL_MAP(c) SDL_memcpy(info->position, PIPEWIRE_channel_map_##c, sizeof(PIPEWIRE_channel_map_##c))

static void initialize_spa_info(const SDL_AudioSpec *spec, struct spa_audio_info_raw *info)
{
    info->channels = spec->channels;
    info->rate = spec->freq;

    switch (spec->channels) {
    case 1:
        COPY_CHANNEL_MAP(1);
        break;
    case 2:
        COPY_CHANNEL_MAP(2);
        break;
    case 3:
        COPY_CHANNEL_MAP(3);
        break;
    case 4:
        COPY_CHANNEL_MAP(4);
        break;
    case 5:
        COPY_CHANNEL_MAP(5);
        break;
    case 6:
        COPY_CHANNEL_MAP(6);
        break;
    case 7:
        COPY_CHANNEL_MAP(7);
        break;
    case 8:
        COPY_CHANNEL_MAP(8);
        break;
    }

    /* Pipewire natively supports all of SDL's sample formats */
    switch (spec->format) {
    case AUDIO_U8:
        info->format = SPA_AUDIO_FORMAT_U8;
        break;
    case AUDIO_S8:
        info->format = SPA_AUDIO_FORMAT_S8;
        break;
    case AUDIO_U16LSB:
        info->format = SPA_AUDIO_FORMAT_U16_LE;
        break;
    case AUDIO_S16LSB:
        info->format = SPA_AUDIO_FORMAT_S16_LE;
        break;
    case AUDIO_U16MSB:
        info->format = SPA_AUDIO_FORMAT_U16_BE;
        break;
    case AUDIO_S16MSB:
        info->format = SPA_AUDIO_FORMAT_S16_BE;
        break;
    case AUDIO_S32LSB:
        info->format = SPA_AUDIO_FORMAT_S32_LE;
        break;
    case AUDIO_S32MSB:
        info->format = SPA_AUDIO_FORMAT_S32_BE;
        break;
    case AUDIO_F32LSB:
        info->format = SPA_AUDIO_FORMAT_F32_LE;
        break;
    case AUDIO_F32MSB:
        info->format = SPA_AUDIO_FORMAT_F32_BE;
        break;
    }
}

static void output_callback(void *data)
{
    struct pw_buffer *pw_buf;
    struct spa_buffer *spa_buf;
    Uint8 *dst;

    _THIS = (SDL_AudioDevice *)data;
    struct pw_stream *stream = this->hidden->stream;

    /* Shutting down, don't do anything */
    if (SDL_AtomicGet(&this->shutdown)) {
        return;
    }

    /* See if a buffer is available */
    pw_buf = PIPEWIRE_pw_stream_dequeue_buffer(stream);
    if (!pw_buf) {
        return;
    }

    spa_buf = pw_buf->buffer;

    if (spa_buf->datas[0].data == NULL) {
        return;
    }

    /*
     * If the device is disabled, write silence to the stream buffer
     * and run the callback with the work buffer to keep the callback
     * firing regularly in case the audio is being used as a timer.
     */
    SDL_LockMutex(this->mixer_lock);
    if (!SDL_AtomicGet(&this->paused)) {
        if (SDL_AtomicGet(&this->enabled)) {
            dst = spa_buf->datas[0].data;
        } else {
            dst = this->work_buffer;
            SDL_memset(spa_buf->datas[0].data, this->spec.silence, this->spec.size);
        }

        if (!this->stream) {
            this->callbackspec.callback(this->callbackspec.userdata, dst, this->callbackspec.size);
        } else {
            int got;

            /* Fire the callback until we have enough to fill a buffer */
            while (SDL_AudioStreamAvailable(this->stream) < this->spec.size) {
                this->callbackspec.callback(this->callbackspec.userdata, this->work_buffer, this->callbackspec.size);
                SDL_AudioStreamPut(this->stream, this->work_buffer, this->callbackspec.size);
            }

            got = SDL_AudioStreamGet(this->stream, dst, this->spec.size);
            SDL_assert(got == this->spec.size);
        }
    } else {
        SDL_memset(spa_buf->datas[0].data, this->spec.silence, this->spec.size);
    }
    SDL_UnlockMutex(this->mixer_lock);

    spa_buf->datas[0].chunk->offset = 0;
    spa_buf->datas[0].chunk->stride = this->hidden->stride;
    spa_buf->datas[0].chunk->size = this->spec.size;

    PIPEWIRE_pw_stream_queue_buffer(stream, pw_buf);
}

static void input_callback(void *data)
{
    struct pw_buffer *pw_buf;
    struct spa_buffer *spa_buf;
    Uint8 *src;
    _THIS = (SDL_AudioDevice *)data;
    struct pw_stream *stream = this->hidden->stream;

    /* Shutting down, don't do anything */
    if (SDL_AtomicGet(&this->shutdown)) {
        return;
    }

    pw_buf = PIPEWIRE_pw_stream_dequeue_buffer(stream);
    if (!pw_buf) {
        return;
    }

    spa_buf = pw_buf->buffer;
    (src = (Uint8 *)spa_buf->datas[0].data);
    if (!src) {
        return;
    }

    if (!SDL_AtomicGet(&this->paused)) {
        /* Calculate the offset and data size */
        const Uint32 offset = SPA_MIN(spa_buf->datas[0].chunk->offset, spa_buf->datas[0].maxsize);
        const Uint32 size = SPA_MIN(spa_buf->datas[0].chunk->size, spa_buf->datas[0].maxsize - offset);

        src += offset;

        /* Fill the buffer with silence if the stream is disabled. */
        if (!SDL_AtomicGet(&this->enabled)) {
            SDL_memset(src, this->callbackspec.silence, size);
        }

        /* Pipewire can vary the latency, so buffer all incoming data */
        SDL_WriteToDataQueue(this->hidden->buffer, src, size);

        while (SDL_CountDataQueue(this->hidden->buffer) >= this->callbackspec.size) {
            SDL_ReadFromDataQueue(this->hidden->buffer, this->work_buffer, this->callbackspec.size);

            SDL_LockMutex(this->mixer_lock);
            this->callbackspec.callback(this->callbackspec.userdata, this->work_buffer, this->callbackspec.size);
            SDL_UnlockMutex(this->mixer_lock);
        }
    } else if (this->hidden->buffer) { /* Flush the buffer when paused */
        if (SDL_CountDataQueue(this->hidden->buffer) != 0) {
            SDL_ClearDataQueue(this->hidden->buffer, this->hidden->input_buffer_packet_size);
        }
    }

    PIPEWIRE_pw_stream_queue_buffer(stream, pw_buf);
}

static void stream_add_buffer_callback(void *data, struct pw_buffer *buffer)
{
    _THIS = data;

    if (this->iscapture == SDL_FALSE) {
        /*
         * Clamp the output spec samples and size to the max size of the Pipewire buffer.
         * If they exceed the maximum size of the Pipewire buffer, double buffering will be used.
         */
        if (this->spec.size > buffer->buffer->datas[0].maxsize) {
            this->spec.samples = buffer->buffer->datas[0].maxsize / this->hidden->stride;
            this->spec.size = buffer->buffer->datas[0].maxsize;
        }
    } else if (!this->hidden->buffer) {
        /*
         * The latency of source nodes can change, so buffering is always required.
         *
         * Ensure that the intermediate input buffer is large enough to hold the requested
         * application packet size or a full buffer of data from Pipewire, whichever is larger.
         *
         * A packet size of 2 periods should be more than is ever needed.
         */
        this->hidden->input_buffer_packet_size = SPA_MAX(this->spec.size, buffer->buffer->datas[0].maxsize) * 2;
        this->hidden->buffer = SDL_NewDataQueue(this->hidden->input_buffer_packet_size, this->hidden->input_buffer_packet_size);
    }

    this->hidden->stream_init_status |= PW_READY_FLAG_BUFFER_ADDED;
    PIPEWIRE_pw_thread_loop_signal(this->hidden->loop, false);
}

static void stream_state_changed_callback(void *data, enum pw_stream_state old, enum pw_stream_state state, const char *error)
{
    _THIS = data;

    if (state == PW_STREAM_STATE_STREAMING) {
        this->hidden->stream_init_status |= PW_READY_FLAG_STREAM_READY;
    }

    if (state == PW_STREAM_STATE_STREAMING || state == PW_STREAM_STATE_ERROR) {
        PIPEWIRE_pw_thread_loop_signal(this->hidden->loop, false);
    }
}

static const struct pw_stream_events stream_output_events = { PW_VERSION_STREAM_EVENTS,
                                                              .state_changed = stream_state_changed_callback,
                                                              .add_buffer = stream_add_buffer_callback,
                                                              .process = output_callback };
static const struct pw_stream_events stream_input_events = { PW_VERSION_STREAM_EVENTS,
                                                             .state_changed = stream_state_changed_callback,
                                                             .add_buffer = stream_add_buffer_callback,
                                                             .process = input_callback };

static int PIPEWIRE_OpenDevice(_THIS, const char *devname)
{
    /*
     * NOTE: The PW_STREAM_FLAG_RT_PROCESS flag can be set to call the stream
     * processing callback from the realtime thread.  However, it comes with some
     * caveats: no file IO, allocations, locking or other blocking operations
     * must occur in the mixer callback.  As this cannot be guaranteed when the
     * callback is in the calling application, this flag is omitted.
     */
    static const enum pw_stream_flags STREAM_FLAGS = PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS;

    char thread_name[PW_THREAD_NAME_BUFFER_LENGTH];
    Uint8 pod_buffer[PW_POD_BUFFER_LENGTH];
    struct spa_pod_builder b = SPA_POD_BUILDER_INIT(pod_buffer, sizeof(pod_buffer));
    struct spa_audio_info_raw spa_info = { 0 };
    const struct spa_pod *params = NULL;
    struct SDL_PrivateAudioData *priv;
    struct pw_properties *props;
    const char *app_name, *stream_name, *stream_role, *error;
    Uint32 node_id = !this->handle ? PW_ID_ANY : PW_HANDLE_TO_ID(this->handle);
    SDL_bool iscapture = this->iscapture;
    int res;

    /* Clamp the period size to sane values */
    const int min_period = PW_MIN_SAMPLES * SPA_MAX(this->spec.freq / PW_BASE_CLOCK_RATE, 1);

    /* Get the hints for the application name, stream name and role */
    app_name = SDL_GetHint(SDL_HINT_AUDIO_DEVICE_APP_NAME);
    if (!app_name || *app_name == '\0') {
        app_name = SDL_GetHint(SDL_HINT_APP_NAME);
        if (!app_name || *app_name == '\0') {
            app_name = "SDL Application";
        }
    }

    stream_name = SDL_GetHint(SDL_HINT_AUDIO_DEVICE_STREAM_NAME);
    if (!stream_name || *stream_name == '\0') {
        stream_name = "Audio Stream";
    }

    /*
     * 'Music' is the default used internally by Pipewire and it's modules,
     * but 'Game' seems more appropriate for the majority of SDL applications.
     */
    stream_role = SDL_GetHint(SDL_HINT_AUDIO_DEVICE_STREAM_ROLE);
    if (!stream_role || *stream_role == '\0') {
        stream_role = "Game";
    }

    /* Initialize the Pipewire stream info from the SDL audio spec */
    initialize_spa_info(&this->spec, &spa_info);
    params = spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat, &spa_info);
    if (!params) {
        return SDL_SetError("Pipewire: Failed to set audio format parameters");
    }

    priv = SDL_calloc(1, sizeof(struct SDL_PrivateAudioData));
    this->hidden = priv;
    if (!priv) {
        return SDL_OutOfMemory();
    }

    /* Size of a single audio frame in bytes */
    priv->stride = (SDL_AUDIO_BITSIZE(this->spec.format) >> 3) * this->spec.channels;

    if (this->spec.samples < min_period) {
        this->spec.samples = min_period;
        this->spec.size = this->spec.samples * priv->stride;
    }

    (void)SDL_snprintf(thread_name, sizeof(thread_name), "SDLAudio%c%ld", (iscapture) ? 'C' : 'P', (long)this->handle);
    priv->loop = PIPEWIRE_pw_thread_loop_new(thread_name, NULL);
    if (!priv->loop) {
        return SDL_SetError("Pipewire: Failed to create stream loop (%i)", errno);
    }

    /* Load the realtime module so Pipewire can set the loop thread to the appropriate priority. */
    props = PIPEWIRE_pw_properties_new(PW_KEY_CONFIG_NAME, "client-rt.conf", NULL);
    if (!props) {
        return SDL_SetError("Pipewire: Failed to create stream context properties (%i)", errno);
    }

    priv->context = PIPEWIRE_pw_context_new(PIPEWIRE_pw_thread_loop_get_loop(priv->loop), props, 0);
    if (!priv->context) {
        return SDL_SetError("Pipewire: Failed to create stream context (%i)", errno);
    }

    props = PIPEWIRE_pw_properties_new(NULL, NULL);
    if (!props) {
        return SDL_SetError("Pipewire: Failed to create stream properties (%i)", errno);
    }

    PIPEWIRE_pw_properties_set(props, PW_KEY_MEDIA_TYPE, "Audio");
    PIPEWIRE_pw_properties_set(props, PW_KEY_MEDIA_CATEGORY, iscapture ? "Capture" : "Playback");
    PIPEWIRE_pw_properties_set(props, PW_KEY_MEDIA_ROLE, stream_role);
    PIPEWIRE_pw_properties_set(props, PW_KEY_APP_NAME, app_name);
    PIPEWIRE_pw_properties_set(props, PW_KEY_NODE_NAME, stream_name);
    PIPEWIRE_pw_properties_set(props, PW_KEY_NODE_DESCRIPTION, stream_name);
    PIPEWIRE_pw_properties_setf(props, PW_KEY_NODE_LATENCY, "%u/%i", this->spec.samples, this->spec.freq);
    PIPEWIRE_pw_properties_setf(props, PW_KEY_NODE_RATE, "1/%u", this->spec.freq);
    PIPEWIRE_pw_properties_set(props, PW_KEY_NODE_ALWAYS_PROCESS, "true");

    /*
     * Pipewire 0.3.44 introduced PW_KEY_TARGET_OBJECT that takes either a path
     * (PW_KEY_NODE_NAME) or node serial number (PE_KEY_OBJECT_SERIAL) to connect
     * the stream to its target. The target_id parameter in pw_stream_connect() is
     * now deprecated and should always be PW_ID_ANY.
     */
    if (pipewire_version_at_least(0, 3, 44)) {
        if (node_id != PW_ID_ANY) {
            const struct io_node *node;

            PIPEWIRE_pw_thread_loop_lock(hotplug_loop);
            node = io_list_get_by_id(node_id);
            if (node) {
                PIPEWIRE_pw_properties_set(props, PW_KEY_TARGET_OBJECT, node->path);
            }
            PIPEWIRE_pw_thread_loop_unlock(hotplug_loop);

            node_id = PW_ID_ANY;
        }
    }

    /* Create the new stream */
    priv->stream = PIPEWIRE_pw_stream_new_simple(PIPEWIRE_pw_thread_loop_get_loop(priv->loop), stream_name, props,
                                                 iscapture ? &stream_input_events : &stream_output_events, this);
    if (!priv->stream) {
        return SDL_SetError("Pipewire: Failed to create stream (%i)", errno);
    }

    res = PIPEWIRE_pw_stream_connect(priv->stream, iscapture ? PW_DIRECTION_INPUT : PW_DIRECTION_OUTPUT, node_id, STREAM_FLAGS,
                                     &params, 1);
    if (res != 0) {
        return SDL_SetError("Pipewire: Failed to connect stream");
    }

    res = PIPEWIRE_pw_thread_loop_start(priv->loop);
    if (res != 0) {
        return SDL_SetError("Pipewire: Failed to start stream loop");
    }

    /* Wait until all init flags are set or the stream has failed. */
    PIPEWIRE_pw_thread_loop_lock(priv->loop);
    while (priv->stream_init_status != PW_READY_FLAG_ALL_BITS &&
           PIPEWIRE_pw_stream_get_state(priv->stream, NULL) != PW_STREAM_STATE_ERROR) {
        PIPEWIRE_pw_thread_loop_wait(priv->loop);
    }
    PIPEWIRE_pw_thread_loop_unlock(priv->loop);

    if (PIPEWIRE_pw_stream_get_state(priv->stream, &error) == PW_STREAM_STATE_ERROR) {
        return SDL_SetError("Pipewire: Stream error: %s", error);
    }

    /* If this is a capture stream, make sure the intermediate buffer was successfully allocated. */
    if (iscapture && !priv->buffer) {
        return SDL_SetError("Pipewire: Failed to allocate source buffer");
    }

    return 0;
}

static void PIPEWIRE_CloseDevice(_THIS)
{
    if (this->hidden->loop) {
        PIPEWIRE_pw_thread_loop_stop(this->hidden->loop);
    }

    if (this->hidden->stream) {
        PIPEWIRE_pw_stream_destroy(this->hidden->stream);
    }

    if (this->hidden->context) {
        PIPEWIRE_pw_context_destroy(this->hidden->context);
    }

    if (this->hidden->loop) {
        PIPEWIRE_pw_thread_loop_destroy(this->hidden->loop);
    }

    if (this->hidden->buffer) {
        SDL_FreeDataQueue(this->hidden->buffer);
    }

    SDL_free(this->hidden);
}

static int PIPEWIRE_GetDefaultAudioInfo(char **name, SDL_AudioSpec *spec, int iscapture)
{
    struct io_node *node;
    char *target;
    int ret = 0;

    PIPEWIRE_pw_thread_loop_lock(hotplug_loop);

    if (iscapture) {
        if (!pipewire_default_source_id) {
            ret = SDL_SetError("PipeWire could not find a default source");
            goto failed;
        }
        target = pipewire_default_source_id;
    } else {
        if (!pipewire_default_sink_id) {
            ret = SDL_SetError("PipeWire could not find a default sink");
            goto failed;
        }
        target = pipewire_default_sink_id;
    }

    node = io_list_get_by_path(target);
    if (!node) {
        ret = SDL_SetError("PipeWire device list is out of sync with defaults");
        goto failed;
    }

    if (name) {
        *name = SDL_strdup(node->name);
    }
    SDL_copyp(spec, &node->spec);

failed:
    PIPEWIRE_pw_thread_loop_unlock(hotplug_loop);
    return ret;
}

static void PIPEWIRE_Deinitialize(void)
{
    if (pipewire_initialized) {
        hotplug_loop_destroy();
        deinit_pipewire_library();
        pipewire_initialized = SDL_FALSE;
    }
}

static SDL_bool PIPEWIRE_Init(SDL_AudioDriverImpl *impl)
{
    if (!pipewire_initialized) {
        if (init_pipewire_library() < 0) {
            return SDL_FALSE;
        }

        pipewire_initialized = SDL_TRUE;

        if (hotplug_loop_init() < 0) {
            PIPEWIRE_Deinitialize();
            return SDL_FALSE;
        }
    }

    /* Set the function pointers */
    impl->DetectDevices = PIPEWIRE_DetectDevices;
    impl->OpenDevice = PIPEWIRE_OpenDevice;
    impl->CloseDevice = PIPEWIRE_CloseDevice;
    impl->Deinitialize = PIPEWIRE_Deinitialize;
    impl->GetDefaultAudioInfo = PIPEWIRE_GetDefaultAudioInfo;

    impl->HasCaptureSupport = SDL_TRUE;
    impl->ProvidesOwnCallbackThread = SDL_TRUE;
    impl->SupportsNonPow2Samples = SDL_TRUE;

    return SDL_TRUE;
}

AudioBootStrap PIPEWIRE_bootstrap = { "pipewire", "Pipewire", PIPEWIRE_Init, SDL_FALSE };

#endif /* SDL_AUDIO_DRIVER_PIPEWIRE */

/* vi: set ts=4 sw=4 expandtab: */
