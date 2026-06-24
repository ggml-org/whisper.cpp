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

#ifdef SDL_VIDEO_DRIVER_DIRECTFB

#include "SDL_DirectFB_video.h"
#include "SDL_DirectFB_window.h"

#include "../../events/SDL_windowevents_c.h"

#define COLOR_EXPAND(col) col.r, col.g, col.b, col.a

static DFB_Theme theme_std = {
    4, 4, 8, 8,
    {255, 200, 200, 200},
    24,
    {255, 0, 0, 255},
    16,
    {255, 255, 255, 255},
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    {255, 255, 0, 0},
    {255, 255, 255, 0},
};

static DFB_Theme theme_none = {
    0, 0, 0, 0,
    {0, 0, 0, 0},
    0,
    {0, 0, 0, 0},
    0,
    {0, 0, 0, 0},
    NULL
};

static void DrawTriangle(IDirectFBSurface * s, int down, int x, int y, int w)
{
    int x1, x2, x3;
    int y1, y2, y3;

    if (down) {
        x1 = x + w / 2;
        x2 = x;
        x3 = x + w;
        y1 = y + w;
        y2 = y;
        y3 = y;
    } else {
        x1 = x + w / 2;
        x2 = x;
        x3 = x + w;
        y1 = y;
        y2 = y + w;
        y3 = y + w;
    }
    s->FillTriangle(s, x1, y1, x2, y2, x3, y3);
}

static void LoadFont(_THIS, SDL_Window * window)
{
    SDL_DFB_DEVICEDATA(_this);
    SDL_DFB_WINDOWDATA(window);

    if (windata->font) {
        SDL_DFB_RELEASE(windata->font);
        windata->font = NULL;
        SDL_DFB_CHECK(windata->window_surface->SetFont(windata->window_surface, windata->font));
    }

    if (windata->theme.font)
    {
        DFBFontDescription fdesc;

        SDL_zero(fdesc);
        fdesc.flags = DFDESC_HEIGHT;
        fdesc.height = windata->theme.font_size;
        SDL_DFB_CHECK(devdata->
                      dfb->CreateFont(devdata->dfb, windata->theme.font,
                                      &fdesc, &windata->font));
        SDL_DFB_CHECK(windata->window_surface->SetFont(windata->window_surface, windata->font));
    }
}

static void DrawCraption(_THIS, IDirectFBSurface * s, int x, int y, char *text)
{
    DFBSurfaceTextFlags flags;

    flags = DSTF_CENTER | DSTF_TOP;

    s->DrawString(s, text, -1, x, y, flags);
}

void DirectFB_WM_RedrawLayout(_THIS, SDL_Window * window)
{
    SDL_DFB_WINDOWDATA(window);
    IDirectFBSurface *s = windata->window_surface;
    DFB_Theme *t = &windata->theme;
    int i;
    int d = (t->caption_size - t->font_size) / 2;
    int x, y, w;


    if (!windata->is_managed || (window->flags & SDL_WINDOW_FULLSCREEN))
        return;

    SDL_DFB_CHECK(s->SetSrcBlendFunction(s, DSBF_ONE));
    SDL_DFB_CHECK(s->SetDstBlendFunction(s, DSBF_ZERO));
    SDL_DFB_CHECK(s->SetDrawingFlags(s, DSDRAW_NOFX));
    SDL_DFB_CHECK(s->SetBlittingFlags(s, DSBLIT_NOFX));

    LoadFont(_this, window);
    /* s->SetDrawingFlags(s, DSDRAW_BLEND); */
    s->SetColor(s, COLOR_EXPAND(t->frame_color));
    /* top */
    for (i = 0; i < t->top_size; i++)
        s->DrawLine(s, 0, i, windata->size.w, i);
    /* bottom */
    for (i = windata->size.h - t->bottom_size; i < windata->size.h; i++)
        s->DrawLine(s, 0, i, windata->size.w, i);
    /* left */
    for (i = 0; i < t->left_size; i++)
        s->DrawLine(s, i, 0, i, windata->size.h);
    /* right */
    for (i = windata->size.w - t->right_size; i < windata->size.w; i++)
        s->DrawLine(s, i, 0, i, windata->size.h);
    /* Caption */
    s->SetColor(s, COLOR_EXPAND(t->caption_color));
    s->FillRectangle(s, t->left_size, t->top_size, windata->client.w,
                     t->caption_size);
    /* Close Button */
    w = t->caption_size;
    x = windata->size.w - t->right_size - w + d;
    y = t->top_size + d;
    s->SetColor(s, COLOR_EXPAND(t->close_color));
    DrawTriangle(s, 1, x, y, w - 2 * d);
    /* Max Button */
    s->SetColor(s, COLOR_EXPAND(t->max_color));
    DrawTriangle(s, window->flags & SDL_WINDOW_MAXIMIZED ? 1 : 0, x - w,
               y, w - 2 * d);

    /* Caption */
    if (*window->title) {
        s->SetColor(s, COLOR_EXPAND(t->font_color));
        DrawCraption(_this, s, (x - w) / 2, t->top_size + d, window->title);
    }
    /* Icon */
    if (windata->icon) {
        DFBRectangle dr;

        dr.x = t->left_size + d;
        dr.y = t->top_size + d;
        dr.w = w - 2 * d;
        dr.h = w - 2 * d;
        s->SetBlittingFlags(s, DSBLIT_BLEND_ALPHACHANNEL);

        s->StretchBlit(s, windata->icon, NULL, &dr);
    }
    windata->wm_needs_redraw = 0;
}

DFBResult DirectFB_WM_GetClientSize(_THIS, SDL_Window * window, int *cw, int *ch)
{
    SDL_DFB_WINDOWDATA(window);
    IDirectFBWindow *dfbwin = windata->dfbwin;

    SDL_DFB_CHECK(dfbwin->GetSize(dfbwin, cw, ch));
    dfbwin->GetSize(dfbwin, cw, ch);
    *cw -= windata->theme.left_size + windata->theme.right_size;
    *ch -=
        windata->theme.top_size + windata->theme.caption_size +
        windata->theme.bottom_size;
    return DFB_OK;
}

void DirectFB_WM_AdjustWindowLayout(SDL_Window * window, int flags, int w, int h)
{
    SDL_DFB_WINDOWDATA(window);

    if (!windata->is_managed)
        windata->theme = theme_none;
    else if (flags & SDL_WINDOW_BORDERLESS)
        /* desc.caps |= DWCAPS_NODECORATION;) */
        windata->theme = theme_none;
    else if (flags & SDL_WINDOW_FULLSCREEN) {
        windata->theme = theme_none;
    } else if (flags & SDL_WINDOW_MAXIMIZED) {
        windata->theme = theme_std;
        windata->theme.left_size = 0;
        windata->theme.right_size = 0;
        windata->theme.top_size = 0;
        windata->theme.bottom_size = 0;
    } else {
        windata->theme = theme_std;
    }

    windata->client.x = windata->theme.left_size;
    windata->client.y = windata->theme.top_size + windata->theme.caption_size;
    windata->client.w = w;
    windata->client.h = h;
    windata->size.w =
        w + windata->theme.left_size + windata->theme.right_size;
    windata->size.h =
        h + windata->theme.top_size +
        windata->theme.caption_size + windata->theme.bottom_size;
}


enum
{
    WM_POS_NONE = 0x00,
    WM_POS_CAPTION = 0x01,
    WM_POS_CLOSE = 0x02,
    WM_POS_MAX = 0x04,
    WM_POS_LEFT = 0x08,
    WM_POS_RIGHT = 0x10,
    WM_POS_TOP = 0x20,
    WM_POS_BOTTOM = 0x40,
};

static int WMIsClient(DFB_WindowData * p, int x, int y)
{
    x -= p->client.x;
    y -= p->client.y;
    if (x < 0 || y < 0)
        return 0;
    if (x >= p->client.w || y >= p->client.h)
        return 0;
    return 1;
}

static int WMPos(DFB_WindowData * p, int x, int y)
{
    int pos = WM_POS_NONE;

    if (!WMIsClient(p, x, y)) {
        if (y < p->theme.top_size) {
            pos |= WM_POS_TOP;
        } else if (y < p->client.y) {
            if (x <
                p->size.w - p->theme.right_size - 2 * p->theme.caption_size) {
                pos |= WM_POS_CAPTION;
            } else if (x <
                       p->size.w - p->theme.right_size -
                       p->theme.caption_size) {
                pos |= WM_POS_MAX;
            } else {
                pos |= WM_POS_CLOSE;
            }
        } else if (y >= p->size.h - p->theme.bottom_size) {
            pos |= WM_POS_BOTTOM;
        }
        if (x < p->theme.left_size) {
            pos |= WM_POS_LEFT;
        } else if (x >= p->size.w - p->theme.right_size) {
            pos |= WM_POS_RIGHT;
        }
    }
    return pos;
}

int DirectFB_WM_ProcessEvent(_THIS, SDL_Window * window, DFBWindowEvent * evt)
{
    SDL_DFB_WINDOWDATA(window);
    SDL_Window *grabbed_window = SDL_GetGrabbedWindow();
    IDirectFBWindow *dfbwin = windata->dfbwin;
    DFBWindowOptions wopts;

    if (!windata->is_managed)
        return 0;

    SDL_DFB_CHECK(dfbwin->GetOptions(dfbwin, &wopts));

    switch (evt->type) {
    case DWET_BUTTONDOWN:
        if (evt->buttons & DIBM_LEFT) {
            int pos = WMPos(windata, evt->x, evt->y);
            switch (pos) {
            case WM_POS_NONE:
                return 0;
            case WM_POS_CLOSE:
                windata->wm_grab = WM_POS_NONE;
                SDL_SendWindowEvent(window, SDL_WINDOWEVENT_CLOSE, 0,
                                    0);
                return 1;
            case WM_POS_MAX:
                windata->wm_grab = WM_POS_NONE;
                if (window->flags & SDL_WINDOW_MAXIMIZED) {
                    SDL_RestoreWindow(window);
                } else {
                    SDL_MaximizeWindow(window);
                }
                return 1;
            case WM_POS_CAPTION:
                if (!(wopts & DWOP_KEEP_STACKING)) {
                    DirectFB_RaiseWindow(_this, window);
                }
                if (window->flags & SDL_WINDOW_MAXIMIZED)
                    return 1;
                SDL_FALLTHROUGH;
            default:
                windata->wm_grab = pos;
                if (grabbed_window)
                    DirectFB_SetWindowMouseGrab(_this, grabbed_window, SDL_FALSE);
                DirectFB_SetWindowMouseGrab(_this, window, SDL_TRUE);
                windata->wm_lastx = evt->cx;
                windata->wm_lasty = evt->cy;
            }
        }
        return 1;
    case DWET_BUTTONUP:
        if (!windata->wm_grab)
            return 0;
        if (!(evt->buttons & DIBM_LEFT)) {
            if (windata->wm_grab & (WM_POS_RIGHT | WM_POS_BOTTOM)) {
                int dx = evt->cx - windata->wm_lastx;
                int dy = evt->cy - windata->wm_lasty;

                if (!(wopts & DWOP_KEEP_SIZE)) {
                    int cw, ch;
                    if ((windata->wm_grab & (WM_POS_BOTTOM | WM_POS_RIGHT)) == WM_POS_BOTTOM)
                        dx = 0;
                    else if ((windata->wm_grab & (WM_POS_BOTTOM | WM_POS_RIGHT)) == WM_POS_RIGHT)
                        dy = 0;
                    SDL_DFB_CHECK(dfbwin->GetSize(dfbwin, &cw, &ch));

                    /* necessary to trigger an event - ugly */
                    SDL_DFB_CHECK(dfbwin->DisableEvents(dfbwin, DWET_ALL));
                    SDL_DFB_CHECK(dfbwin->Resize(dfbwin, cw + dx + 1, ch + dy));
                    SDL_DFB_CHECK(dfbwin->EnableEvents(dfbwin, DWET_ALL));

                    SDL_DFB_CHECK(dfbwin->Resize(dfbwin, cw + dx, ch + dy));
                }
            }
            DirectFB_SetWindowMouseGrab(_this, window, SDL_FALSE);
            if (grabbed_window)
                DirectFB_SetWindowMouseGrab(_this, grabbed_window, SDL_TRUE);
            windata->wm_grab = WM_POS_NONE;
            return 1;
        }
        break;
    case DWET_MOTION:
        if (!windata->wm_grab)
            return 0;
        if (evt->buttons & DIBM_LEFT) {
            int dx = evt->cx - windata->wm_lastx;
            int dy = evt->cy - windata->wm_lasty;

            if (windata->wm_grab & WM_POS_CAPTION) {
                if (!(wopts & DWOP_KEEP_POSITION))
                    SDL_DFB_CHECK(dfbwin->Move(dfbwin, dx, dy));
            }
            if (windata->wm_grab & (WM_POS_RIGHT | WM_POS_BOTTOM)) {
                if (!(wopts & DWOP_KEEP_SIZE)) {
                    int cw, ch;

                    /* Make sure all events are disabled for this operation ! */
                    SDL_DFB_CHECK(dfbwin->DisableEvents(dfbwin, DWET_ALL));

                    if ((windata->wm_grab & (WM_POS_BOTTOM | WM_POS_RIGHT)) == WM_POS_BOTTOM)
                        dx = 0;
                    else if ((windata->wm_grab & (WM_POS_BOTTOM | WM_POS_RIGHT)) == WM_POS_RIGHT)
                        dy = 0;

                    SDL_DFB_CHECK(dfbwin->GetSize(dfbwin, &cw, &ch));
                    SDL_DFB_CHECK(dfbwin->Resize(dfbwin, cw + dx, ch + dy));

                    SDL_DFB_CHECK(dfbwin->EnableEvents(dfbwin, DWET_ALL));
                }
            }
            windata->wm_lastx = evt->cx;
            windata->wm_lasty = evt->cy;
            return 1;
        }
        break;
    case DWET_KEYDOWN:
        break;
    case DWET_KEYUP:
        break;
    default:
        ;
    }
    return 0;
}

#endif /* SDL_VIDEO_DRIVER_DIRECTFB */
