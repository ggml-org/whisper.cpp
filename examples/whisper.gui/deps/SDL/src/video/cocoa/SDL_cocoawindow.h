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

#ifndef SDL_cocoawindow_h_
#define SDL_cocoawindow_h_

#import <Cocoa/Cocoa.h>

#ifdef SDL_VIDEO_OPENGL_EGL
#include "../SDL_egl_c.h"
#endif

@class SDL_WindowData;

typedef enum
{
    PENDING_OPERATION_NONE,
    PENDING_OPERATION_ENTER_FULLSCREEN,
    PENDING_OPERATION_LEAVE_FULLSCREEN,
    PENDING_OPERATION_MINIMIZE
} PendingWindowOperation;

@interface Cocoa_WindowListener : NSResponder <NSWindowDelegate> {
    /* SDL_WindowData owns this Listener and has a strong reference to it.
     * To avoid reference cycles, we could have either a weak or an
     * unretained ref to the WindowData. */
    __weak SDL_WindowData *_data;
    BOOL observingVisible;
    BOOL wasCtrlLeft;
    BOOL wasVisible;
    BOOL isFullscreenSpace;
    BOOL inFullscreenTransition;
    PendingWindowOperation pendingWindowOperation;
    BOOL isMoving;
    NSInteger focusClickPending;
    int pendingWindowWarpX, pendingWindowWarpY;
    BOOL isDragAreaRunning;
    NSTimer *liveResizeTimer;
}

-(BOOL) isTouchFromTrackpad:(NSEvent *)theEvent;
-(void) listen:(SDL_WindowData *) data;
-(void) pauseVisibleObservation;
-(void) resumeVisibleObservation;
-(BOOL) setFullscreenSpace:(BOOL) state;
-(BOOL) isInFullscreenSpace;
-(BOOL) isInFullscreenSpaceTransition;
-(void) addPendingWindowOperation:(PendingWindowOperation) operation;
-(void) close;

-(BOOL) isMoving;
-(BOOL) isMovingOrFocusClickPending;
-(void) setFocusClickPending:(NSInteger) button;
-(void) clearFocusClickPending:(NSInteger) button;
-(void) setPendingMoveX:(int)x Y:(int)y;
-(void) windowDidFinishMoving;
-(void) onMovingOrFocusClickPendingStateCleared;

/* Window delegate functionality */
-(BOOL) windowShouldClose:(id) sender;
-(void) windowDidExpose:(NSNotification *) aNotification;
-(void) windowWillStartLiveResize:(NSNotification *)aNotification;
-(void) windowDidEndLiveResize:(NSNotification *)aNotification;
-(void) windowDidMove:(NSNotification *) aNotification;
-(void) windowDidResize:(NSNotification *) aNotification;
-(void) windowDidMiniaturize:(NSNotification *) aNotification;
-(void) windowDidDeminiaturize:(NSNotification *) aNotification;
-(void) windowDidBecomeKey:(NSNotification *) aNotification;
-(void) windowDidResignKey:(NSNotification *) aNotification;
-(void) windowDidChangeBackingProperties:(NSNotification *) aNotification;
-(void) windowDidChangeScreenProfile:(NSNotification *) aNotification;
-(void) windowDidChangeScreen:(NSNotification *) aNotification;
-(void) windowWillEnterFullScreen:(NSNotification *) aNotification;
-(void) windowDidEnterFullScreen:(NSNotification *) aNotification;
-(void) windowWillExitFullScreen:(NSNotification *) aNotification;
-(void) windowDidExitFullScreen:(NSNotification *) aNotification;
-(NSApplicationPresentationOptions)window:(NSWindow *)window willUseFullScreenPresentationOptions:(NSApplicationPresentationOptions)proposedOptions;

/* See if event is in a drag area, toggle on window dragging. */
-(BOOL) processHitTest:(NSEvent *)theEvent;

/* Window event handling */
-(void) mouseDown:(NSEvent *) theEvent;
-(void) rightMouseDown:(NSEvent *) theEvent;
-(void) otherMouseDown:(NSEvent *) theEvent;
-(void) mouseUp:(NSEvent *) theEvent;
-(void) rightMouseUp:(NSEvent *) theEvent;
-(void) otherMouseUp:(NSEvent *) theEvent;
-(void) mouseMoved:(NSEvent *) theEvent;
-(void) mouseDragged:(NSEvent *) theEvent;
-(void) rightMouseDragged:(NSEvent *) theEvent;
-(void) otherMouseDragged:(NSEvent *) theEvent;
-(void) scrollWheel:(NSEvent *) theEvent;
-(void) touchesBeganWithEvent:(NSEvent *) theEvent;
-(void) touchesMovedWithEvent:(NSEvent *) theEvent;
-(void) touchesEndedWithEvent:(NSEvent *) theEvent;
-(void) touchesCancelledWithEvent:(NSEvent *) theEvent;

/* Touch event handling */
-(void) handleTouches:(NSTouchPhase) phase withEvent:(NSEvent*) theEvent;

@end
/* *INDENT-ON* */

@class SDLOpenGLContext;
@class SDL_VideoData;

@interface SDL_WindowData : NSObject
    @property (nonatomic) SDL_Window *window;
    @property (nonatomic) NSWindow *nswindow;
    @property (nonatomic) NSView *sdlContentView;
    @property (nonatomic) NSMutableArray *nscontexts;
    @property (nonatomic) SDL_bool created;
    @property (nonatomic) SDL_bool inWindowFullscreenTransition;
    @property (nonatomic) NSInteger window_number;
    @property (nonatomic) NSInteger flash_request;
    @property (nonatomic) Cocoa_WindowListener *listener;
    @property (nonatomic) SDL_VideoData *videodata;
#ifdef SDL_VIDEO_OPENGL_EGL
    @property (nonatomic) EGLSurface egl_surface;
#endif
@end

extern int Cocoa_CreateWindow(_THIS, SDL_Window * window);
extern int Cocoa_CreateWindowFrom(_THIS, SDL_Window * window,
                                  const void *data);
extern void Cocoa_SetWindowTitle(_THIS, SDL_Window * window);
extern void Cocoa_SetWindowIcon(_THIS, SDL_Window * window, SDL_Surface * icon);
extern void Cocoa_SetWindowPosition(_THIS, SDL_Window * window);
extern void Cocoa_SetWindowSize(_THIS, SDL_Window * window);
extern void Cocoa_SetWindowMinimumSize(_THIS, SDL_Window * window);
extern void Cocoa_SetWindowMaximumSize(_THIS, SDL_Window * window);
extern void Cocoa_GetWindowSizeInPixels(_THIS, SDL_Window * window, int *w, int *h);
extern int Cocoa_SetWindowOpacity(_THIS, SDL_Window * window, float opacity);
extern void Cocoa_ShowWindow(_THIS, SDL_Window * window);
extern void Cocoa_HideWindow(_THIS, SDL_Window * window);
extern void Cocoa_RaiseWindow(_THIS, SDL_Window * window);
extern void Cocoa_MaximizeWindow(_THIS, SDL_Window * window);
extern void Cocoa_MinimizeWindow(_THIS, SDL_Window * window);
extern void Cocoa_RestoreWindow(_THIS, SDL_Window * window);
extern void Cocoa_SetWindowBordered(_THIS, SDL_Window * window, SDL_bool bordered);
extern void Cocoa_SetWindowResizable(_THIS, SDL_Window * window, SDL_bool resizable);
extern void Cocoa_SetWindowAlwaysOnTop(_THIS, SDL_Window * window, SDL_bool on_top);
extern void Cocoa_SetWindowFullscreen(_THIS, SDL_Window * window, SDL_VideoDisplay * display, SDL_bool fullscreen);
extern int Cocoa_SetWindowGammaRamp(_THIS, SDL_Window * window, const Uint16 * ramp);
extern void* Cocoa_GetWindowICCProfile(_THIS, SDL_Window * window, size_t * size);
extern int Cocoa_GetWindowDisplayIndex(_THIS, SDL_Window * window);
extern int Cocoa_GetWindowGammaRamp(_THIS, SDL_Window * window, Uint16 * ramp);
extern void Cocoa_SetWindowMouseRect(_THIS, SDL_Window * window);
extern void Cocoa_SetWindowMouseGrab(_THIS, SDL_Window * window, SDL_bool grabbed);
extern void Cocoa_DestroyWindow(_THIS, SDL_Window * window);
extern SDL_bool Cocoa_GetWindowWMInfo(_THIS, SDL_Window * window, struct SDL_SysWMinfo *info);
extern int Cocoa_SetWindowHitTest(SDL_Window *window, SDL_bool enabled);
extern void Cocoa_AcceptDragAndDrop(SDL_Window * window, SDL_bool accept);
extern int Cocoa_FlashWindow(_THIS, SDL_Window * window, SDL_FlashOperation operation);

#endif /* SDL_cocoawindow_h_ */

/* vi: set ts=4 sw=4 expandtab: */
