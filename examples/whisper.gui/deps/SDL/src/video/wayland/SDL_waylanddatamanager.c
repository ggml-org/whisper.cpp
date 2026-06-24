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

#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <signal.h>

#include "SDL_stdinc.h"
#include "../../core/unix/SDL_poll.h"

#include "SDL_waylandvideo.h"
#include "SDL_waylanddatamanager.h"
#include "primary-selection-unstable-v1-client-protocol.h"

/* FIXME: This is arbitrary, but we want this to be less than a frame because
 * any longer can potentially spin an infinite loop of PumpEvents (!)
 */
#define PIPE_MS_TIMEOUT 14

static ssize_t write_pipe(int fd, const void *buffer, size_t total_length, size_t *pos)
{
    int ready = 0;
    ssize_t bytes_written = 0;
    ssize_t length = total_length - *pos;

    sigset_t sig_set;
    sigset_t old_sig_set;
    struct timespec zerotime = { 0 };

    ready = SDL_IOReady(fd, SDL_IOR_WRITE, PIPE_MS_TIMEOUT);

    sigemptyset(&sig_set);
    sigaddset(&sig_set, SIGPIPE);

#ifdef SDL_THREADS_DISABLED
    sigprocmask(SIG_BLOCK, &sig_set, &old_sig_set);
#else
    pthread_sigmask(SIG_BLOCK, &sig_set, &old_sig_set);
#endif

    if (ready == 0) {
        bytes_written = SDL_SetError("Pipe timeout");
    } else if (ready < 0) {
        bytes_written = SDL_SetError("Pipe select error");
    } else {
        if (length > 0) {
            bytes_written = write(fd, (Uint8 *)buffer + *pos, SDL_min(length, PIPE_BUF));
        }

        if (bytes_written > 0) {
            *pos += bytes_written;
        }
    }

    sigtimedwait(&sig_set, 0, &zerotime);

#ifdef SDL_THREADS_DISABLED
    sigprocmask(SIG_SETMASK, &old_sig_set, NULL);
#else
    pthread_sigmask(SIG_SETMASK, &old_sig_set, NULL);
#endif

    return bytes_written;
}

static ssize_t read_pipe(int fd, void **buffer, size_t *total_length, SDL_bool null_terminate)
{
    int ready = 0;
    void *output_buffer = NULL;
    char temp[PIPE_BUF];
    size_t new_buffer_length = 0;
    ssize_t bytes_read = 0;
    size_t pos = 0;

    ready = SDL_IOReady(fd, SDL_IOR_READ, PIPE_MS_TIMEOUT);

    if (ready == 0) {
        bytes_read = SDL_SetError("Pipe timeout");
    } else if (ready < 0) {
        bytes_read = SDL_SetError("Pipe select error");
    } else {
        bytes_read = read(fd, temp, sizeof(temp));
    }

    if (bytes_read > 0) {
        pos = *total_length;
        *total_length += bytes_read;

        if (null_terminate == SDL_TRUE) {
            new_buffer_length = *total_length + 1;
        } else {
            new_buffer_length = *total_length;
        }

        if (!*buffer) {
            output_buffer = SDL_malloc(new_buffer_length);
        } else {
            output_buffer = SDL_realloc(*buffer, new_buffer_length);
        }

        if (!output_buffer) {
            bytes_read = SDL_OutOfMemory();
        } else {
            SDL_memcpy((Uint8 *)output_buffer + pos, temp, bytes_read);

            if (null_terminate == SDL_TRUE) {
                SDL_memset((Uint8 *)output_buffer + (new_buffer_length - 1), 0, 1);
            }

            *buffer = output_buffer;
        }
    }

    return bytes_read;
}

#define MIME_LIST_SIZE 4

static const char *mime_conversion_list[MIME_LIST_SIZE][2] = {
    { "text/plain", TEXT_MIME },
    { "TEXT", TEXT_MIME },
    { "UTF8_STRING", TEXT_MIME },
    { "STRING", TEXT_MIME }
};

const char *Wayland_convert_mime_type(const char *mime_type)
{
    const char *found = mime_type;

    size_t index = 0;

    for (index = 0; index < MIME_LIST_SIZE; ++index) {
        if (SDL_strcmp(mime_conversion_list[index][0], mime_type) == 0) {
            found = mime_conversion_list[index][1];
            break;
        }
    }

    return found;
}

static SDL_MimeDataList *mime_data_list_find(struct wl_list *list,
                                             const char *mime_type)
{
    SDL_MimeDataList *found = NULL;

    SDL_MimeDataList *mime_list = NULL;
    wl_list_for_each (mime_list, list, link) {
        if (SDL_strcmp(mime_list->mime_type, mime_type) == 0) {
            found = mime_list;
            break;
        }
    }
    return found;
}

static int mime_data_list_add(struct wl_list *list,
                              const char *mime_type,
                              const void *buffer, size_t length)
{
    int status = 0;
    size_t mime_type_length = 0;
    SDL_MimeDataList *mime_data = NULL;
    void *internal_buffer = NULL;

    if (buffer) {
        internal_buffer = SDL_malloc(length);
        if (!internal_buffer) {
            return SDL_OutOfMemory();
        }
        SDL_memcpy(internal_buffer, buffer, length);
    }

    mime_data = mime_data_list_find(list, mime_type);

    if (!mime_data) {
        mime_data = SDL_calloc(1, sizeof(*mime_data));
        if (!mime_data) {
            status = SDL_OutOfMemory();
        } else {
            WAYLAND_wl_list_insert(list, &(mime_data->link));

            mime_type_length = SDL_strlen(mime_type) + 1;
            mime_data->mime_type = SDL_malloc(mime_type_length);
            if (!mime_data->mime_type) {
                status = SDL_OutOfMemory();
            } else {
                SDL_memcpy(mime_data->mime_type, mime_type, mime_type_length);
            }
        }
    }

    if (mime_data && buffer && length > 0) {
        if (mime_data->data) {
            SDL_free(mime_data->data);
        }
        mime_data->data = internal_buffer;
        mime_data->length = length;
    } else {
        SDL_free(internal_buffer);
    }

    return status;
}

static void mime_data_list_free(struct wl_list *list)
{
    SDL_MimeDataList *mime_data = NULL;
    SDL_MimeDataList *next = NULL;

    wl_list_for_each_safe(mime_data, next, list, link)
    {
        if (mime_data->data) {
            SDL_free(mime_data->data);
        }
        if (mime_data->mime_type) {
            SDL_free(mime_data->mime_type);
        }
        SDL_free(mime_data);
    }
}

static ssize_t Wayland_source_send(SDL_MimeDataList *mime_data, const char *mime_type, int fd)
{
    size_t written_bytes = 0;
    ssize_t status = 0;

    if (!mime_data || !mime_data->data) {
        status = SDL_SetError("Invalid mime type");
        close(fd);
    } else {
        while (write_pipe(fd,
                          mime_data->data,
                          mime_data->length,
                          &written_bytes) > 0) {
        }
        close(fd);
        status = written_bytes;
    }
    return status;
}

ssize_t Wayland_data_source_send(SDL_WaylandDataSource *source, const char *mime_type, int fd)
{
    SDL_MimeDataList *mime_data = NULL;

    mime_type = Wayland_convert_mime_type(mime_type);
    mime_data = mime_data_list_find(&source->mimes,
                                    mime_type);

    return Wayland_source_send(mime_data, mime_type, fd);
}

ssize_t Wayland_primary_selection_source_send(SDL_WaylandPrimarySelectionSource *source, const char *mime_type, int fd)
{
    SDL_MimeDataList *mime_data = NULL;

    mime_type = Wayland_convert_mime_type(mime_type);
    mime_data = mime_data_list_find(&source->mimes,
                                    mime_type);

    return Wayland_source_send(mime_data, mime_type, fd);
}

int Wayland_data_source_add_data(SDL_WaylandDataSource *source,
                                 const char *mime_type,
                                 const void *buffer,
                                 size_t length)
{
    return mime_data_list_add(&source->mimes, mime_type, buffer, length);
}

int Wayland_primary_selection_source_add_data(SDL_WaylandPrimarySelectionSource *source,
                                              const char *mime_type,
                                              const void *buffer,
                                              size_t length)
{
    return mime_data_list_add(&source->mimes, mime_type, buffer, length);
}

SDL_bool Wayland_data_source_has_mime(SDL_WaylandDataSource *source,
                                      const char *mime_type)
{
    SDL_bool found = SDL_FALSE;

    if (source) {
        found = mime_data_list_find(&source->mimes, mime_type) != NULL;
    }
    return found;
}

SDL_bool Wayland_primary_selection_source_has_mime(SDL_WaylandPrimarySelectionSource *source,
                                                   const char *mime_type)
{
    SDL_bool found = SDL_FALSE;

    if (source) {
        found = mime_data_list_find(&source->mimes, mime_type) != NULL;
    }
    return found;
}

static void *Wayland_source_get_data(SDL_MimeDataList *mime_data,
                                     size_t *length,
                                     SDL_bool null_terminate)
{
    void *buffer = NULL;

    if (mime_data && mime_data->length > 0) {
        size_t buffer_length = mime_data->length;

        if (null_terminate == SDL_TRUE) {
            ++buffer_length;
        }
        buffer = SDL_malloc(buffer_length);
        if (!buffer) {
            *length = SDL_OutOfMemory();
        } else {
            *length = mime_data->length;
            SDL_memcpy(buffer, mime_data->data, mime_data->length);
            if (null_terminate) {
                *((Uint8 *)buffer + mime_data->length) = 0;
            }
        }
    }

    return buffer;
}

void *Wayland_data_source_get_data(SDL_WaylandDataSource *source,
                                   size_t *length, const char *mime_type,
                                   SDL_bool null_terminate)
{
    SDL_MimeDataList *mime_data = NULL;
    void *buffer = NULL;
    *length = 0;

    if (!source) {
        SDL_SetError("Invalid data source");
    } else {
        mime_data = mime_data_list_find(&source->mimes, mime_type);
        buffer = Wayland_source_get_data(mime_data, length, null_terminate);
    }

    return buffer;
}

void *Wayland_primary_selection_source_get_data(SDL_WaylandPrimarySelectionSource *source,
                                                size_t *length, const char *mime_type,
                                                SDL_bool null_terminate)
{
    SDL_MimeDataList *mime_data = NULL;
    void *buffer = NULL;
    *length = 0;

    if (!source) {
        SDL_SetError("Invalid primary selection source");
    } else {
        mime_data = mime_data_list_find(&source->mimes, mime_type);
        buffer = Wayland_source_get_data(mime_data, length, null_terminate);
    }

    return buffer;
}

void Wayland_data_source_destroy(SDL_WaylandDataSource *source)
{
    if (source) {
        SDL_WaylandDataDevice *data_device = (SDL_WaylandDataDevice *)source->data_device;
        if (data_device && (data_device->selection_source == source)) {
            data_device->selection_source = NULL;
        }
        wl_data_source_destroy(source->source);
        mime_data_list_free(&source->mimes);
        SDL_free(source);
    }
}

void Wayland_primary_selection_source_destroy(SDL_WaylandPrimarySelectionSource *source)
{
    if (source) {
        SDL_WaylandPrimarySelectionDevice *primary_selection_device = (SDL_WaylandPrimarySelectionDevice *)source->primary_selection_device;
        if (primary_selection_device && (primary_selection_device->selection_source == source)) {
            primary_selection_device->selection_source = NULL;
        }
        zwp_primary_selection_source_v1_destroy(source->source);
        mime_data_list_free(&source->mimes);
        SDL_free(source);
    }
}

void *Wayland_data_offer_receive(SDL_WaylandDataOffer *offer,
                                 size_t *length, const char *mime_type,
                                 SDL_bool null_terminate)
{
    SDL_WaylandDataDevice *data_device = NULL;

    int pipefd[2];
    void *buffer = NULL;
    *length = 0;

    if (!offer) {
        SDL_SetError("Invalid data offer");
        return NULL;
    }
    data_device = offer->data_device;
    if (!data_device) {
        SDL_SetError("Data device not initialized");
    } else if (pipe2(pipefd, O_CLOEXEC | O_NONBLOCK) == -1) {
        SDL_SetError("Could not read pipe");
    } else {
        wl_data_offer_receive(offer->offer, mime_type, pipefd[1]);

        /* TODO: Needs pump and flush? */
        WAYLAND_wl_display_flush(data_device->video_data->display);

        close(pipefd[1]);

        while (read_pipe(pipefd[0], &buffer, length, null_terminate) > 0) {
        }
        close(pipefd[0]);
    }
    return buffer;
}

void *Wayland_primary_selection_offer_receive(SDL_WaylandPrimarySelectionOffer *offer,
                                              size_t *length, const char *mime_type,
                                              SDL_bool null_terminate)
{
    SDL_WaylandPrimarySelectionDevice *primary_selection_device = NULL;

    int pipefd[2];
    void *buffer = NULL;
    *length = 0;

    if (!offer) {
        SDL_SetError("Invalid data offer");
        return NULL;
    }
    primary_selection_device = offer->primary_selection_device;
    if (!primary_selection_device) {
        SDL_SetError("Primary selection device not initialized");
    } else if (pipe2(pipefd, O_CLOEXEC | O_NONBLOCK) == -1) {
        SDL_SetError("Could not read pipe");
    } else {
        zwp_primary_selection_offer_v1_receive(offer->offer, mime_type, pipefd[1]);

        /* TODO: Needs pump and flush? */
        WAYLAND_wl_display_flush(primary_selection_device->video_data->display);

        close(pipefd[1]);

        while (read_pipe(pipefd[0], &buffer, length, null_terminate) > 0) {
        }
        close(pipefd[0]);
    }
    return buffer;
}

int Wayland_data_offer_add_mime(SDL_WaylandDataOffer *offer,
                                const char *mime_type)
{
    return mime_data_list_add(&offer->mimes, mime_type, NULL, 0);
}

int Wayland_primary_selection_offer_add_mime(SDL_WaylandPrimarySelectionOffer *offer,
                                             const char *mime_type)
{
    return mime_data_list_add(&offer->mimes, mime_type, NULL, 0);
}

SDL_bool Wayland_data_offer_has_mime(SDL_WaylandDataOffer *offer,
                                     const char *mime_type)
{
    SDL_bool found = SDL_FALSE;

    if (offer) {
        found = mime_data_list_find(&offer->mimes, mime_type) != NULL;
    }
    return found;
}

SDL_bool Wayland_primary_selection_offer_has_mime(SDL_WaylandPrimarySelectionOffer *offer,
                                                  const char *mime_type)
{
    SDL_bool found = SDL_FALSE;

    if (offer) {
        found = mime_data_list_find(&offer->mimes, mime_type) != NULL;
    }
    return found;
}

void Wayland_data_offer_destroy(SDL_WaylandDataOffer *offer)
{
    if (offer) {
        wl_data_offer_destroy(offer->offer);
        mime_data_list_free(&offer->mimes);
        SDL_free(offer);
    }
}

void Wayland_primary_selection_offer_destroy(SDL_WaylandPrimarySelectionOffer *offer)
{
    if (offer) {
        zwp_primary_selection_offer_v1_destroy(offer->offer);
        mime_data_list_free(&offer->mimes);
        SDL_free(offer);
    }
}

int Wayland_data_device_clear_selection(SDL_WaylandDataDevice *data_device)
{
    int status = 0;

    if (!data_device || !data_device->data_device) {
        status = SDL_SetError("Invalid Data Device");
    } else if (data_device->selection_source) {
        wl_data_device_set_selection(data_device->data_device, NULL, 0);
        Wayland_data_source_destroy(data_device->selection_source);
        data_device->selection_source = NULL;
    }
    return status;
}

int Wayland_primary_selection_device_clear_selection(SDL_WaylandPrimarySelectionDevice *primary_selection_device)
{
    int status = 0;

    if (!primary_selection_device || !primary_selection_device->primary_selection_device) {
        status = SDL_SetError("Invalid Primary Selection Device");
    } else if (primary_selection_device->selection_source) {
        zwp_primary_selection_device_v1_set_selection(primary_selection_device->primary_selection_device,
                                                      NULL, 0);
        Wayland_primary_selection_source_destroy(primary_selection_device->selection_source);
        primary_selection_device->selection_source = NULL;
    }
    return status;
}

int Wayland_data_device_set_selection(SDL_WaylandDataDevice *data_device,
                                      SDL_WaylandDataSource *source)
{
    int status = 0;
    size_t num_offers = 0;
    size_t index = 0;

    if (!data_device) {
        status = SDL_SetError("Invalid Data Device");
    } else if (!source) {
        status = SDL_SetError("Invalid source");
    } else {
        SDL_MimeDataList *mime_data = NULL;

        wl_list_for_each (mime_data, &(source->mimes), link) {
            wl_data_source_offer(source->source,
                                 mime_data->mime_type);

            /* TODO - Improve system for multiple mime types to same data */
            for (index = 0; index < MIME_LIST_SIZE; ++index) {
                if (SDL_strcmp(mime_conversion_list[index][1], mime_data->mime_type) == 0) {
                    wl_data_source_offer(source->source,
                                         mime_conversion_list[index][0]);
                }
            }
            /* */

            ++num_offers;
        }

        if (num_offers == 0) {
            Wayland_data_device_clear_selection(data_device);
            status = SDL_SetError("No mime data");
        } else {
            /* Only set if there is a valid serial if not set it later */
            if (data_device->selection_serial != 0) {
                wl_data_device_set_selection(data_device->data_device,
                                             source->source,
                                             data_device->selection_serial);
            }
            if (data_device->selection_source) {
                Wayland_data_source_destroy(data_device->selection_source);
            }
            data_device->selection_source = source;
            source->data_device = data_device;
        }
    }

    return status;
}

int Wayland_primary_selection_device_set_selection(SDL_WaylandPrimarySelectionDevice *primary_selection_device,
                                                   SDL_WaylandPrimarySelectionSource *source)
{
    int status = 0;
    size_t num_offers = 0;
    size_t index = 0;

    if (!primary_selection_device) {
        status = SDL_SetError("Invalid Primary Selection Device");
    } else if (!source) {
        status = SDL_SetError("Invalid source");
    } else {
        SDL_MimeDataList *mime_data = NULL;

        wl_list_for_each (mime_data, &(source->mimes), link) {
            zwp_primary_selection_source_v1_offer(source->source,
                                                  mime_data->mime_type);

            /* TODO - Improve system for multiple mime types to same data */
            for (index = 0; index < MIME_LIST_SIZE; ++index) {
                if (SDL_strcmp(mime_conversion_list[index][1], mime_data->mime_type) == 0) {
                    zwp_primary_selection_source_v1_offer(source->source,
                                                          mime_conversion_list[index][0]);
                }
            }
            /* */

            ++num_offers;
        }

        if (num_offers == 0) {
            Wayland_primary_selection_device_clear_selection(primary_selection_device);
            status = SDL_SetError("No mime data");
        } else {
            /* Only set if there is a valid serial if not set it later */
            if (primary_selection_device->selection_serial != 0) {
                zwp_primary_selection_device_v1_set_selection(primary_selection_device->primary_selection_device,
                                                              source->source,
                                                              primary_selection_device->selection_serial);
            }
            if (primary_selection_device->selection_source) {
                Wayland_primary_selection_source_destroy(primary_selection_device->selection_source);
            }
            primary_selection_device->selection_source = source;
            source->primary_selection_device = primary_selection_device;
        }
    }

    return status;
}

int Wayland_data_device_set_serial(SDL_WaylandDataDevice *data_device,
                                   uint32_t serial)
{
    int status = -1;
    if (data_device) {
        status = 0;

        /* If there was no serial and there is a pending selection set it now. */
        if (data_device->selection_serial == 0 && data_device->selection_source) {
            wl_data_device_set_selection(data_device->data_device,
                                         data_device->selection_source->source,
                                         data_device->selection_serial);
        }

        data_device->selection_serial = serial;
    }

    return status;
}

int Wayland_primary_selection_device_set_serial(SDL_WaylandPrimarySelectionDevice *primary_selection_device,
                                                uint32_t serial)
{
    int status = -1;
    if (primary_selection_device) {
        status = 0;

        /* If there was no serial and there is a pending selection set it now. */
        if (primary_selection_device->selection_serial == 0 && primary_selection_device->selection_source) {
            zwp_primary_selection_device_v1_set_selection(primary_selection_device->primary_selection_device,
                                                          primary_selection_device->selection_source->source,
                                                          primary_selection_device->selection_serial);
        }

        primary_selection_device->selection_serial = serial;
    }

    return status;
}

#endif /* SDL_VIDEO_DRIVER_WAYLAND */

/* vi: set ts=4 sw=4 expandtab: */
