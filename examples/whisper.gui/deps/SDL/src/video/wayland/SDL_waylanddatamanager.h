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

#ifndef SDL_waylanddatamanager_h_
#define SDL_waylanddatamanager_h_

#include "SDL_waylandvideo.h"
#include "SDL_waylandwindow.h"

#define TEXT_MIME "text/plain;charset=utf-8"
#define FILE_MIME "text/uri-list"
#define FILE_PORTAL_MIME "application/vnd.portal.filetransfer"

typedef struct
{
    char *mime_type;
    void *data;
    size_t length;
    struct wl_list link;
} SDL_MimeDataList;

typedef struct
{
    struct wl_data_source *source;
    struct wl_list mimes;
    void *data_device;
} SDL_WaylandDataSource;

typedef struct
{
    struct zwp_primary_selection_source_v1 *source;
    struct wl_list mimes;
    void *primary_selection_device;
} SDL_WaylandPrimarySelectionSource;

typedef struct
{
    struct wl_data_offer *offer;
    struct wl_list mimes;
    void *data_device;
} SDL_WaylandDataOffer;

typedef struct
{
    struct zwp_primary_selection_offer_v1 *offer;
    struct wl_list mimes;
    void *primary_selection_device;
} SDL_WaylandPrimarySelectionOffer;

typedef struct
{
    struct wl_data_device *data_device;
    SDL_VideoData *video_data;

    /* Drag and Drop */
    uint32_t drag_serial;
    SDL_WaylandDataOffer *drag_offer;
    SDL_WaylandDataOffer *selection_offer;
    SDL_Window *dnd_window;

    /* Clipboard and Primary Selection */
    uint32_t selection_serial;
    SDL_WaylandDataSource *selection_source;
} SDL_WaylandDataDevice;

typedef struct
{
    struct zwp_primary_selection_device_v1 *primary_selection_device;
    SDL_VideoData *video_data;

    uint32_t selection_serial;
    SDL_WaylandPrimarySelectionSource *selection_source;
    SDL_WaylandPrimarySelectionOffer *selection_offer;
} SDL_WaylandPrimarySelectionDevice;

extern const char *Wayland_convert_mime_type(const char *mime_type);

/* Wayland Data Source / Primary Selection Source - (Sending) */
extern SDL_WaylandDataSource *Wayland_data_source_create(_THIS);
extern SDL_WaylandPrimarySelectionSource *Wayland_primary_selection_source_create(_THIS);
extern ssize_t Wayland_data_source_send(SDL_WaylandDataSource *source,
                                        const char *mime_type, int fd);
extern ssize_t Wayland_primary_selection_source_send(SDL_WaylandPrimarySelectionSource *source,
                                                     const char *mime_type, int fd);
extern int Wayland_data_source_add_data(SDL_WaylandDataSource *source,
                                        const char *mime_type,
                                        const void *buffer,
                                        size_t length);
extern int Wayland_primary_selection_source_add_data(SDL_WaylandPrimarySelectionSource *source,
                                                     const char *mime_type,
                                                     const void *buffer,
                                                     size_t length);
extern SDL_bool Wayland_data_source_has_mime(SDL_WaylandDataSource *source,
                                             const char *mime_type);
extern SDL_bool Wayland_primary_selection_source_has_mime(SDL_WaylandPrimarySelectionSource *source,
                                                          const char *mime_type);
extern void *Wayland_data_source_get_data(SDL_WaylandDataSource *source,
                                          size_t *length,
                                          const char *mime_type,
                                          SDL_bool null_terminate);
extern void *Wayland_primary_selection_source_get_data(SDL_WaylandPrimarySelectionSource *source,
                                                       size_t *length,
                                                       const char *mime_type,
                                                       SDL_bool null_terminate);
extern void Wayland_data_source_destroy(SDL_WaylandDataSource *source);
extern void Wayland_primary_selection_source_destroy(SDL_WaylandPrimarySelectionSource *source);

/* Wayland Data / Primary Selection Offer - (Receiving) */
extern void *Wayland_data_offer_receive(SDL_WaylandDataOffer *offer,
                                        size_t *length,
                                        const char *mime_type,
                                        SDL_bool null_terminate);
extern void *Wayland_primary_selection_offer_receive(SDL_WaylandPrimarySelectionOffer *offer,
                                                     size_t *length,
                                                     const char *mime_type,
                                                     SDL_bool null_terminate);
extern SDL_bool Wayland_data_offer_has_mime(SDL_WaylandDataOffer *offer,
                                            const char *mime_type);
extern SDL_bool Wayland_primary_selection_offer_has_mime(SDL_WaylandPrimarySelectionOffer *offer,
                                                         const char *mime_type);
extern int Wayland_data_offer_add_mime(SDL_WaylandDataOffer *offer,
                                       const char *mime_type);
extern int Wayland_primary_selection_offer_add_mime(SDL_WaylandPrimarySelectionOffer *offer,
                                                    const char *mime_type);
extern void Wayland_data_offer_destroy(SDL_WaylandDataOffer *offer);
extern void Wayland_primary_selection_offer_destroy(SDL_WaylandPrimarySelectionOffer *offer);

/* Clipboard / Primary Selection */
extern int Wayland_data_device_clear_selection(SDL_WaylandDataDevice *device);
extern int Wayland_primary_selection_device_clear_selection(SDL_WaylandPrimarySelectionDevice *device);
extern int Wayland_data_device_set_selection(SDL_WaylandDataDevice *device,
                                             SDL_WaylandDataSource *source);
extern int Wayland_primary_selection_device_set_selection(SDL_WaylandPrimarySelectionDevice *device,
                                                          SDL_WaylandPrimarySelectionSource *source);
extern int Wayland_data_device_set_serial(SDL_WaylandDataDevice *device,
                                          uint32_t serial);
extern int Wayland_primary_selection_device_set_serial(SDL_WaylandPrimarySelectionDevice *device,
                                                       uint32_t serial);
#endif /* SDL_waylanddatamanager_h_ */

/* vi: set ts=4 sw=4 expandtab: */
