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

#include "SDL_waylanddatamanager.h"
#include "SDL_waylandevents_c.h"
#include "SDL_waylandclipboard.h"

int Wayland_SetClipboardText(_THIS, const char *text)
{
    SDL_VideoData *video_data = NULL;
    SDL_WaylandDataDevice *data_device = NULL;

    int status = 0;

    if (!_this || !_this->driverdata) {
        status = SDL_SetError("Video driver uninitialized");
    } else {
        video_data = _this->driverdata;
        if (video_data->input && video_data->input->data_device) {
            data_device = video_data->input->data_device;
            if (text[0] != '\0') {
                SDL_WaylandDataSource *source = Wayland_data_source_create(_this);
                Wayland_data_source_add_data(source, TEXT_MIME, text,
                                             SDL_strlen(text));

                status = Wayland_data_device_set_selection(data_device, source);
                if (status != 0) {
                    Wayland_data_source_destroy(source);
                }
            } else {
                status = Wayland_data_device_clear_selection(data_device);
            }
        }
    }

    return status;
}

int Wayland_SetPrimarySelectionText(_THIS, const char *text)
{
    SDL_VideoData *video_data = NULL;
    SDL_WaylandPrimarySelectionDevice *primary_selection_device = NULL;

    int status = 0;

    if (!_this || !_this->driverdata) {
        status = SDL_SetError("Video driver uninitialized");
    } else {
        video_data = _this->driverdata;
        if (video_data->input && video_data->input->primary_selection_device) {
            primary_selection_device = video_data->input->primary_selection_device;
            if (text[0] != '\0') {
                SDL_WaylandPrimarySelectionSource *source = Wayland_primary_selection_source_create(_this);
                Wayland_primary_selection_source_add_data(source, TEXT_MIME, text,
                                                          SDL_strlen(text));

                status = Wayland_primary_selection_device_set_selection(primary_selection_device,
                                                                        source);
                if (status != 0) {
                    Wayland_primary_selection_source_destroy(source);
                }
            } else {
                status = Wayland_primary_selection_device_clear_selection(primary_selection_device);
            }
        }
    }

    return status;
}

char *Wayland_GetClipboardText(_THIS)
{
    SDL_VideoData *video_data = NULL;
    SDL_WaylandDataDevice *data_device = NULL;

    char *text = NULL;
    size_t length = 0;

    if (!_this || !_this->driverdata) {
        SDL_SetError("Video driver uninitialized");
    } else {
        video_data = _this->driverdata;
        if (video_data->input && video_data->input->data_device) {
            data_device = video_data->input->data_device;
            /* Prefer own selection, if not canceled */
            if (Wayland_data_source_has_mime(
                    data_device->selection_source, TEXT_MIME)) {
                text = Wayland_data_source_get_data(data_device->selection_source,
                                                    &length, TEXT_MIME, SDL_TRUE);
            } else if (Wayland_data_offer_has_mime(
                           data_device->selection_offer, TEXT_MIME)) {
                text = Wayland_data_offer_receive(data_device->selection_offer,
                                                  &length, TEXT_MIME, SDL_TRUE);
            }
        }
    }

    if (!text) {
        text = SDL_strdup("");
    }

    return text;
}

char *Wayland_GetPrimarySelectionText(_THIS)
{
    SDL_VideoData *video_data = NULL;
    SDL_WaylandPrimarySelectionDevice *primary_selection_device = NULL;

    char *text = NULL;
    size_t length = 0;

    if (!_this || !_this->driverdata) {
        SDL_SetError("Video driver uninitialized");
    } else {
        video_data = _this->driverdata;
        if (video_data->input && video_data->input->primary_selection_device) {
            primary_selection_device = video_data->input->primary_selection_device;
            /* Prefer own selection, if not canceled */
            if (Wayland_primary_selection_source_has_mime(
                    primary_selection_device->selection_source, TEXT_MIME)) {
                text = Wayland_primary_selection_source_get_data(primary_selection_device->selection_source,
                                                                 &length, TEXT_MIME, SDL_TRUE);
            } else if (Wayland_primary_selection_offer_has_mime(
                           primary_selection_device->selection_offer, TEXT_MIME)) {
                text = Wayland_primary_selection_offer_receive(primary_selection_device->selection_offer,
                                                               &length, TEXT_MIME, SDL_TRUE);
            }
        }
    }

    if (!text) {
        text = SDL_strdup("");
    }

    return text;
}

SDL_bool Wayland_HasClipboardText(_THIS)
{
    SDL_VideoData *video_data = NULL;
    SDL_WaylandDataDevice *data_device = NULL;

    SDL_bool result = SDL_FALSE;
    if (!_this || !_this->driverdata) {
        SDL_SetError("Video driver uninitialized");
    } else {
        video_data = _this->driverdata;
        if (video_data->input && video_data->input->data_device) {
            data_device = video_data->input->data_device;
            result = result ||
                     Wayland_data_source_has_mime(data_device->selection_source, TEXT_MIME) ||
                     Wayland_data_offer_has_mime(data_device->selection_offer, TEXT_MIME);
        }
    }
    return result;
}

SDL_bool Wayland_HasPrimarySelectionText(_THIS)
{
    SDL_VideoData *video_data = NULL;
    SDL_WaylandPrimarySelectionDevice *primary_selection_device = NULL;

    SDL_bool result = SDL_FALSE;
    if (!_this || !_this->driverdata) {
        SDL_SetError("Video driver uninitialized");
    } else {
        video_data = _this->driverdata;
        if (video_data->input && video_data->input->primary_selection_device) {
            primary_selection_device = video_data->input->primary_selection_device;
            result = result ||
                     Wayland_primary_selection_source_has_mime(
                         primary_selection_device->selection_source, TEXT_MIME) ||
                     Wayland_primary_selection_offer_has_mime(
                         primary_selection_device->selection_offer, TEXT_MIME);
        }
    }
    return result;
}

#endif /* SDL_VIDEO_DRIVER_WAYLAND */

/* vi: set ts=4 sw=4 expandtab: */
