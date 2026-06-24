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
#include "../SDL_internal.h"

#ifndef SDL_shape_internals_h_
#define SDL_shape_internals_h_

#include "SDL_rect.h"
#include "SDL_shape.h"
#include "SDL_surface.h"

#include "begin_code.h"
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
/* *INDENT-OFF* */
extern "C" {
/* *INDENT-ON* */
#endif

struct SDL_ShapeTree;

typedef struct
{
    struct SDL_ShapeTree *upleft, *upright, *downleft, *downright;
} SDL_QuadTreeChildren;

typedef union
{
    SDL_QuadTreeChildren children;
    SDL_Rect shape;
} SDL_ShapeUnion;

typedef enum
{
    QuadShape,
    TransparentShape,
    OpaqueShape
} SDL_ShapeKind;

typedef struct SDL_ShapeTree
{
    SDL_ShapeKind kind;
    SDL_ShapeUnion data;
} SDL_ShapeTree;

typedef void (*SDL_TraversalFunction)(SDL_ShapeTree *, void *);

extern void SDL_CalculateShapeBitmap(SDL_WindowShapeMode mode, SDL_Surface *shape, Uint8 *bitmap, Uint8 ppb);
extern SDL_ShapeTree *SDL_CalculateShapeTree(SDL_WindowShapeMode mode, SDL_Surface *shape);
extern void SDL_TraverseShapeTree(SDL_ShapeTree *tree, SDL_TraversalFunction function, void *closure);
extern void SDL_FreeShapeTree(SDL_ShapeTree **shape_tree);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
/* *INDENT-OFF* */
}
/* *INDENT-ON* */
#endif
#include "close_code.h"

#endif
