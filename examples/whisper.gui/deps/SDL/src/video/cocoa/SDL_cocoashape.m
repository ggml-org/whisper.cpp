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

#ifdef SDL_VIDEO_DRIVER_COCOA

#include "SDL_cocoavideo.h"
#include "SDL_shape.h"
#include "SDL_cocoashape.h"
#include "../SDL_sysvideo.h"

@implementation SDL_ShapeData
@end

@interface SDL_CocoaClosure : NSObject
    @property (nonatomic) NSView* view;
    @property (nonatomic) NSBezierPath* path;
    @property (nonatomic) SDL_Window* window;
@end

@implementation SDL_CocoaClosure
@end

SDL_WindowShaper *Cocoa_CreateShaper(SDL_Window* window)
{ @autoreleasepool
{
    SDL_WindowShaper* result;
    SDL_ShapeData* data;
    int resized_properly;
    SDL_WindowData* windata = (__bridge SDL_WindowData*)window->driverdata;

    result = (SDL_WindowShaper *)SDL_malloc(sizeof(SDL_WindowShaper));
    if (!result) {
        SDL_OutOfMemory();
        return NULL;
    }

    [windata.nswindow setOpaque:NO];

    [windata.nswindow setStyleMask:NSWindowStyleMaskBorderless];

    result->window = window;
    result->mode.mode = ShapeModeDefault;
    result->mode.parameters.binarizationCutoff = 1;
    result->userx = result->usery = 0;
    window->shaper = result;

    data = [[SDL_ShapeData alloc] init];
    data.context = [windata.nswindow graphicsContext];
    data.saved = SDL_FALSE;
    data.shape = NULL;

    /* TODO: There's no place to release this... */
    result->driverdata = (void*) CFBridgingRetain(data);

    resized_properly = Cocoa_ResizeWindowShape(window);
    SDL_assert(resized_properly == 0);
    return result;
}}

static void ConvertRects(SDL_ShapeTree* tree, void* closure)
{
    SDL_CocoaClosure* data = (__bridge SDL_CocoaClosure*)closure;
    if(tree->kind == OpaqueShape) {
        NSRect rect = NSMakeRect(tree->data.shape.x, data.window->h - tree->data.shape.y, tree->data.shape.w, tree->data.shape.h);
        [data.path appendBezierPathWithRect:[data.view convertRect:rect toView:nil]];
    }
}

int Cocoa_SetWindowShape(SDL_WindowShaper *shaper, SDL_Surface *shape, SDL_WindowShapeMode *shape_mode)
{ @autoreleasepool
{
    SDL_ShapeData* data = (__bridge SDL_ShapeData*)shaper->driverdata;
    SDL_WindowData* windata = (__bridge SDL_WindowData*)shaper->window->driverdata;
    SDL_CocoaClosure* closure;
    if(data.saved == SDL_TRUE) {
        [data.context restoreGraphicsState];
        data.saved = SDL_FALSE;
    }

    /*[data.context saveGraphicsState];*/
    /*data.saved = SDL_TRUE;*/
    [NSGraphicsContext setCurrentContext:data.context];

    [[NSColor clearColor] set];
    NSRectFill([windata.sdlContentView frame]);
    data.shape = SDL_CalculateShapeTree(*shape_mode, shape);

    closure = [[SDL_CocoaClosure alloc] init];

    closure.view = windata.sdlContentView;
    closure.path = [NSBezierPath bezierPath];
    closure.window = shaper->window;
    SDL_TraverseShapeTree(data.shape, &ConvertRects, (__bridge void*)closure);
    [closure.path addClip];

    return 0;
}}

int Cocoa_ResizeWindowShape(SDL_Window *window)
{ @autoreleasepool {
    SDL_ShapeData* data = (__bridge SDL_ShapeData*)window->shaper->driverdata;
    SDL_assert(data != NULL);
    return 0;
}}

#endif /* SDL_VIDEO_DRIVER_COCOA */

/* vi: set ts=4 sw=4 expandtab: */
