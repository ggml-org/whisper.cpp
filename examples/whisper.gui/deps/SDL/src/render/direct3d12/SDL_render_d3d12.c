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

#include "SDL_render.h"
#include "SDL_system.h"

#if SDL_VIDEO_RENDER_D3D12

#define SDL_D3D12_NUM_BUFFERS        2
#define SDL_D3D12_NUM_VERTEX_BUFFERS 256
#define SDL_D3D12_MAX_NUM_TEXTURES   16384
#define SDL_D3D12_NUM_UPLOAD_BUFFERS 32

#include "../../core/windows/SDL_windows.h"
#include "../../video/windows/SDL_windowswindow.h"
#include "SDL_hints.h"
#include "SDL_loadso.h"
#include "SDL_syswm.h"
#include "../SDL_sysrender.h"
#include "../SDL_d3dmath.h"

#if defined(__XBOXONE__) || defined(__XBOXSERIES__)
#include "SDL_render_d3d12_xbox.h"
#ifndef D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
#define D3D12_TEXTURE_DATA_PITCH_ALIGNMENT 256
#endif
#else
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxgidebug.h>
#include <d3d12sdklayers.h>
#include <sdkddkver.h>
#endif

#include "SDL_shaders_d3d12.h"

#if defined(_MSC_VER) && !defined(__clang__)
#define SDL_COMPOSE_ERROR(str) __FUNCTION__ ", " str
#else
#define SDL_COMPOSE_ERROR(str) SDL_STRINGIFY_ARG(__FUNCTION__) ", " str
#endif

/* DXGI_PRESENT flags are removed on Xbox */
#if defined(__XBOXONE__) || defined(__XBOXSERIES__)
#define DXGI_PRESENT_ALLOW_TEARING 0
#endif

#ifdef __cplusplus
#define SAFE_RELEASE(X) \
    if (X) {            \
        (X)->Release(); \
        X = NULL;       \
    }
#define D3D_CALL(THIS, FUNC, ...)             (THIS)->FUNC(__VA_ARGS__)
#define D3D_CALL_RET(THIS, FUNC, RETVAL, ...) *(RETVAL) = (THIS)->FUNC(__VA_ARGS__)
#define D3D_GUID(X)                           (X)
#else
#define SAFE_RELEASE(X)          \
    if (X) {                     \
        (X)->lpVtbl->Release(X); \
        X = NULL;                \
    }
#define D3D_CALL(THIS, FUNC, ...)     (THIS)->lpVtbl->FUNC((THIS), ##__VA_ARGS__)
#define D3D_CALL_RET(THIS, FUNC, ...) (THIS)->lpVtbl->FUNC((THIS), ##__VA_ARGS__)
#define D3D_GUID(X)                   &(X)
#endif

/*
 * Older MS Windows SDK headers declare some d3d12 functions with the wrong function prototype.
 * - ID3D12Heap::GetDesc
 * - ID3D12Resource::GetDesc
 * - ID3D12DescriptorHeap::GetDesc
 * (and 9 more)+
 * This is fixed in SDKs since WDK_NTDDI_VERSION >= NTDDI_WIN10_FE (0x0A00000A)
 */

#if !(defined(__MINGW32__) || defined(__XBOXONE__) || defined(__XBOXSERIES__)) \
    && (WDK_NTDDI_VERSION < 0x0A00000A)

#define D3D_CALL_RET_ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(THIS, ...) do { \
        void (STDMETHODCALLTYPE * func)(ID3D12DescriptorHeap * This, D3D12_CPU_DESCRIPTOR_HANDLE * Handle) = \
            (void*)(THIS)->lpVtbl->GetCPUDescriptorHandleForHeapStart; \
        func((THIS), ##__VA_ARGS__); \
    } while (0)

#define D3D_CALL_RET_ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(THIS, ...) do { \
        void (STDMETHODCALLTYPE * func)(ID3D12DescriptorHeap * This, D3D12_GPU_DESCRIPTOR_HANDLE * Handle) = \
            (void*)(THIS)->lpVtbl->GetGPUDescriptorHandleForHeapStart; \
        func((THIS), ##__VA_ARGS__); \
    } while (0)

#define D3D_CALL_RET_ID3D12Resource_GetDesc(THIS, ...) do { \
        void (STDMETHODCALLTYPE * func)(ID3D12Resource * This, D3D12_RESOURCE_DESC * Desc) = \
            (void*)(THIS)->lpVtbl->GetDesc; \
        func((THIS), ##__VA_ARGS__); \
    } while (0)

#else

/*
 * MinGW has correct function prototypes in the vtables, but defines wrong functions
 * Xbox just needs these macros defined as used below (because CINTERFACE doesn't exist)
 */

#define D3D_CALL_RET_ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(THIS, ...) \
    D3D_CALL_RET(THIS, GetCPUDescriptorHandleForHeapStart, ##__VA_ARGS__);

#define D3D_CALL_RET_ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(THIS, ...) \
    D3D_CALL_RET(THIS, GetGPUDescriptorHandleForHeapStart, ##__VA_ARGS__);

#define D3D_CALL_RET_ID3D12Resource_GetDesc(THIS, ...) \
    D3D_CALL_RET(THIS, GetDesc, ##__VA_ARGS__);

#endif

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/* !!! FIXME: vertex buffer bandwidth could be lower; only use UV coords when
   !!! FIXME:  textures are needed. */

/* Vertex shader, common values */
typedef struct
{
    Float4X4 model;
    Float4X4 projectionAndView;
} VertexShaderConstants;

/* Per-vertex data */
typedef struct
{
    Float2 pos;
    Float2 tex;
    SDL_Color color;
} VertexPositionColor;

/* Per-texture data */
typedef struct
{
    ID3D12Resource *mainTexture;
    D3D12_CPU_DESCRIPTOR_HANDLE mainTextureResourceView;
    D3D12_RESOURCE_STATES mainResourceState;
    SIZE_T mainSRVIndex;
    D3D12_CPU_DESCRIPTOR_HANDLE mainTextureRenderTargetView;
    DXGI_FORMAT mainTextureFormat;
    ID3D12Resource *stagingBuffer;
    D3D12_RESOURCE_STATES stagingResourceState;
    D3D12_FILTER scaleMode;
#if SDL_HAVE_YUV
    /* YV12 texture support */
    SDL_bool yuv;
    ID3D12Resource *mainTextureU;
    D3D12_CPU_DESCRIPTOR_HANDLE mainTextureResourceViewU;
    D3D12_RESOURCE_STATES mainResourceStateU;
    SIZE_T mainSRVIndexU;
    ID3D12Resource *mainTextureV;
    D3D12_CPU_DESCRIPTOR_HANDLE mainTextureResourceViewV;
    D3D12_RESOURCE_STATES mainResourceStateV;
    SIZE_T mainSRVIndexV;

    /* NV12 texture support */
    SDL_bool nv12;
    ID3D12Resource *mainTextureNV;
    D3D12_CPU_DESCRIPTOR_HANDLE mainTextureResourceViewNV;
    D3D12_RESOURCE_STATES mainResourceStateNV;
    SIZE_T mainSRVIndexNV;

    Uint8 *pixels;
    int pitch;
#endif
    SDL_Rect lockedRect;
} D3D12_TextureData;

/* Pipeline State Object data */
typedef struct
{
    D3D12_Shader shader;
    SDL_BlendMode blendMode;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topology;
    DXGI_FORMAT rtvFormat;
    ID3D12PipelineState *pipelineState;
} D3D12_PipelineState;

/* Vertex Buffer */
typedef struct
{
    ID3D12Resource *resource;
    D3D12_VERTEX_BUFFER_VIEW view;
    size_t size;
} D3D12_VertexBuffer;

/* For SRV pool allocator */
typedef struct
{
    SIZE_T index;
    void *next;
} D3D12_SRVPoolNode;

/* Private renderer data */
typedef struct
{
    void *hDXGIMod;
    void *hD3D12Mod;
#if defined(__XBOXONE__) || defined(__XBOXSERIES__)
    UINT64 frameToken;
#else
    IDXGIFactory6 *dxgiFactory;
    IDXGIAdapter4 *dxgiAdapter;
    IDXGIDebug *dxgiDebug;
    IDXGISwapChain4 *swapChain;
#endif
    ID3D12Device1 *d3dDevice;
    ID3D12Debug *debugInterface;
    ID3D12CommandQueue *commandQueue;
    ID3D12GraphicsCommandList2 *commandList;
    DXGI_SWAP_EFFECT swapEffect;
    UINT swapFlags;

    /* Descriptor heaps */
    ID3D12DescriptorHeap *rtvDescriptorHeap;
    UINT rtvDescriptorSize;
    ID3D12DescriptorHeap *textureRTVDescriptorHeap;
    ID3D12DescriptorHeap *srvDescriptorHeap;
    UINT srvDescriptorSize;
    ID3D12DescriptorHeap *samplerDescriptorHeap;
    UINT samplerDescriptorSize;

    /* Data needed per backbuffer */
    ID3D12CommandAllocator *commandAllocators[SDL_D3D12_NUM_BUFFERS];
    ID3D12Resource *renderTargets[SDL_D3D12_NUM_BUFFERS];
    UINT64 fenceValue;
    int currentBackBufferIndex;

    /* Fences */
    ID3D12Fence *fence;
    HANDLE fenceEvent;

    /* Root signature and pipeline state data */
    ID3D12RootSignature *rootSignatures[NUM_ROOTSIGS];
    int pipelineStateCount;
    D3D12_PipelineState *pipelineStates;
    D3D12_PipelineState *currentPipelineState;

    D3D12_VertexBuffer vertexBuffers[SDL_D3D12_NUM_VERTEX_BUFFERS];
    D3D12_CPU_DESCRIPTOR_HANDLE nearestPixelSampler;
    D3D12_CPU_DESCRIPTOR_HANDLE linearSampler;

    /* Data for staging/allocating textures */
    ID3D12Resource *uploadBuffers[SDL_D3D12_NUM_UPLOAD_BUFFERS];
    int currentUploadBuffer;

    /* Pool allocator to handle reusing SRV heap indices */
    D3D12_SRVPoolNode *srvPoolHead;
    D3D12_SRVPoolNode srvPoolNodes[SDL_D3D12_MAX_NUM_TEXTURES];

    /* Vertex buffer constants */
    VertexShaderConstants vertexShaderConstantsData;

    /* Cached renderer properties */
    DXGI_MODE_ROTATION rotation;
    D3D12_TextureData *textureRenderTarget;
    D3D12_CPU_DESCRIPTOR_HANDLE currentRenderTargetView;
    D3D12_CPU_DESCRIPTOR_HANDLE currentShaderResource;
    D3D12_CPU_DESCRIPTOR_HANDLE currentSampler;
    SDL_bool cliprectDirty;
    SDL_bool currentCliprectEnabled;
    SDL_Rect currentCliprect;
    SDL_Rect currentViewport;
    int currentViewportRotation;
    SDL_bool viewportDirty;
    Float4X4 identity;
    int currentVertexBuffer;
    SDL_bool issueBatch;
} D3D12_RenderData;

/* Define D3D GUIDs here so we don't have to include uuid.lib. */

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#endif

static const GUID SDL_IID_IDXGIFactory6 = { 0xc1b6694f, 0xff09, 0x44a9, { 0xb0, 0x3c, 0x77, 0x90, 0x0a, 0x0a, 0x1d, 0x17 } };
static const GUID SDL_IID_IDXGIAdapter4 = { 0x3c8d99d1, 0x4fbf, 0x4181, { 0xa8, 0x2c, 0xaf, 0x66, 0xbf, 0x7b, 0xd2, 0x4e } };
static const GUID SDL_IID_IDXGIDevice1 = { 0x77db970f, 0x6276, 0x48ba, { 0xba, 0x28, 0x07, 0x01, 0x43, 0xb4, 0x39, 0x2c } };
static const GUID SDL_IID_ID3D12Device1 = { 0x77acce80, 0x638e, 0x4e65, { 0x88, 0x95, 0xc1, 0xf2, 0x33, 0x86, 0x86, 0x3e } };
static const GUID SDL_IID_IDXGISwapChain4 = { 0x3D585D5A, 0xBD4A, 0x489E, { 0xB1, 0xF4, 0x3D, 0xBC, 0xB6, 0x45, 0x2F, 0xFB } };
static const GUID SDL_IID_IDXGIDebug1 = { 0xc5a05f0c, 0x16f2, 0x4adf, { 0x9f, 0x4d, 0xa8, 0xc4, 0xd5, 0x8a, 0xc5, 0x50 } };
static const GUID SDL_IID_IDXGIInfoQueue = { 0xD67441C7, 0x672A, 0x476f, { 0x9E, 0x82, 0xCD, 0x55, 0xB4, 0x49, 0x49, 0xCE } };
static const GUID SDL_IID_ID3D12Debug = { 0x344488b7, 0x6846, 0x474b, { 0xb9, 0x89, 0xf0, 0x27, 0x44, 0x82, 0x45, 0xe0 } };
static const GUID SDL_DXGI_DEBUG_ALL = { 0xe48ae283, 0xda80, 0x490b, { 0x87, 0xe6, 0x43, 0xe9, 0xa9, 0xcf, 0xda, 0x8 } };
static const GUID SDL_IID_ID3D12CommandQueue = { 0x0ec870a6, 0x5d7e, 0x4c22, { 0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed } };
static const GUID SDL_IID_ID3D12DescriptorHeap = { 0x8efb471d, 0x616c, 0x4f49, { 0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51 } };
static const GUID SDL_IID_ID3D12CommandAllocator = { 0x6102dee4, 0xaf59, 0x4b09, { 0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24 } };
static const GUID SDL_IID_ID3D12GraphicsCommandList2 = { 0x38C3E585, 0xFF17, 0x412C, { 0x91, 0x50, 0x4F, 0xC6, 0xF9, 0xD7, 0x2A, 0x28 } };
static const GUID SDL_IID_ID3D12Fence = { 0x0a753dcf, 0xc4d8, 0x4b91, { 0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76 } };
static const GUID SDL_IID_ID3D12Resource = { 0x696442be, 0xa72e, 0x4059, { 0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad } };
static const GUID SDL_IID_ID3D12RootSignature = { 0xc54a6b66, 0x72df, 0x4ee8, { 0x8b, 0xe5, 0xa9, 0x46, 0xa1, 0x42, 0x92, 0x14 } };
static const GUID SDL_IID_ID3D12PipelineState = { 0x765a30f3, 0xf624, 0x4c6f, { 0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45 } };
static const GUID SDL_IID_ID3D12Heap = { 0x6b3b2502, 0x6e51, 0x45b3, { 0x90, 0xee, 0x98, 0x84, 0x26, 0x5e, 0x8d, 0xf3 } };
static const GUID SDL_IID_ID3D12InfoQueue = { 0x0742a90b, 0xc387, 0x483f, { 0xb9, 0x46, 0x30, 0xa7, 0xe4, 0xe6, 0x14, 0x58 } };

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

UINT D3D12_Align(UINT location, UINT alignment)
{
    return (location + (alignment - 1)) & ~(alignment - 1);
}

Uint32 D3D12_DXGIFormatToSDLPixelFormat(DXGI_FORMAT dxgiFormat)
{
    switch (dxgiFormat) {
    case DXGI_FORMAT_B8G8R8A8_UNORM:
        return SDL_PIXELFORMAT_ARGB8888;
    case DXGI_FORMAT_B8G8R8X8_UNORM:
        return SDL_PIXELFORMAT_RGB888;
    default:
        return SDL_PIXELFORMAT_UNKNOWN;
    }
}

static DXGI_FORMAT SDLPixelFormatToDXGIFormat(Uint32 sdlFormat)
{
    switch (sdlFormat) {
    case SDL_PIXELFORMAT_ARGB8888:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case SDL_PIXELFORMAT_RGB888:
        return DXGI_FORMAT_B8G8R8X8_UNORM;
    case SDL_PIXELFORMAT_YV12:
    case SDL_PIXELFORMAT_IYUV:
    case SDL_PIXELFORMAT_NV12: /* For the Y texture */
    case SDL_PIXELFORMAT_NV21: /* For the Y texture */
        return DXGI_FORMAT_R8_UNORM;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

static void D3D12_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture);

static void D3D12_ReleaseAll(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;
    SDL_Texture *texture = NULL;

    /* Release all textures */
    for (texture = renderer->textures; texture; texture = texture->next) {
        D3D12_DestroyTexture(renderer, texture);
    }

    /* Release/reset everything else */
    if (data) {
        int i;

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
        SAFE_RELEASE(data->dxgiFactory);
        SAFE_RELEASE(data->dxgiAdapter);
        SAFE_RELEASE(data->swapChain);
#endif
        SAFE_RELEASE(data->d3dDevice);
        SAFE_RELEASE(data->debugInterface);
        SAFE_RELEASE(data->commandQueue);
        SAFE_RELEASE(data->commandList);
        SAFE_RELEASE(data->rtvDescriptorHeap);
        SAFE_RELEASE(data->textureRTVDescriptorHeap);
        SAFE_RELEASE(data->srvDescriptorHeap);
        SAFE_RELEASE(data->samplerDescriptorHeap);
        SAFE_RELEASE(data->fence);

        for (i = 0; i < SDL_D3D12_NUM_BUFFERS; ++i) {
            SAFE_RELEASE(data->commandAllocators[i]);
            SAFE_RELEASE(data->renderTargets[i]);
        }

        if (data->pipelineStateCount > 0) {
            for (i = 0; i < data->pipelineStateCount; ++i) {
                SAFE_RELEASE(data->pipelineStates[i].pipelineState);
            }
            SDL_free(data->pipelineStates);
            data->pipelineStateCount = 0;
        }

        for (i = 0; i < NUM_ROOTSIGS; ++i) {
            SAFE_RELEASE(data->rootSignatures[i]);
        }

        for (i = 0; i < SDL_D3D12_NUM_VERTEX_BUFFERS; ++i) {
            SAFE_RELEASE(data->vertexBuffers[i].resource);
            data->vertexBuffers[i].size = 0;
        }

        data->swapEffect = (DXGI_SWAP_EFFECT)0;
        data->swapFlags = 0;
        data->currentRenderTargetView.ptr = 0;
        data->currentSampler.ptr = 0;

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
        /* Check for any leaks if in debug mode */
        if (data->dxgiDebug) {
            DXGI_DEBUG_RLO_FLAGS rloFlags = (DXGI_DEBUG_RLO_FLAGS)(DXGI_DEBUG_RLO_SUMMARY | DXGI_DEBUG_RLO_IGNORE_INTERNAL);
            D3D_CALL(data->dxgiDebug, ReportLiveObjects, SDL_DXGI_DEBUG_ALL, rloFlags);
            SAFE_RELEASE(data->dxgiDebug);
        }
#endif

        /* Unload the D3D libraries.  This should be done last, in order
         * to prevent IUnknown::Release() calls from crashing.
         */
        if (data->hD3D12Mod) {
            SDL_UnloadObject(data->hD3D12Mod);
            data->hD3D12Mod = NULL;
        }
        if (data->hDXGIMod) {
            SDL_UnloadObject(data->hDXGIMod);
            data->hDXGIMod = NULL;
        }
    }
}

static D3D12_GPU_DESCRIPTOR_HANDLE D3D12_CPUtoGPUHandle(ID3D12DescriptorHeap *heap, D3D12_CPU_DESCRIPTOR_HANDLE CPUHandle)
{
    D3D12_CPU_DESCRIPTOR_HANDLE CPUHeapStart;
    D3D12_GPU_DESCRIPTOR_HANDLE GPUHandle;
    SIZE_T offset;

    /* Calculate the correct offset into the heap */
    D3D_CALL_RET_ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(heap, &CPUHeapStart);
    offset = CPUHandle.ptr - CPUHeapStart.ptr;

    D3D_CALL_RET_ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(heap, &GPUHandle);
    GPUHandle.ptr += offset;

    return GPUHandle;
}

static void D3D12_WaitForGPU(D3D12_RenderData *data)
{
    if (data->commandQueue && data->fence && data->fenceEvent) {
        D3D_CALL(data->commandQueue, Signal, data->fence, data->fenceValue);
        if (D3D_CALL(data->fence, GetCompletedValue) < data->fenceValue) {
            D3D_CALL(data->fence, SetEventOnCompletion,
                     data->fenceValue,
                     data->fenceEvent);
            WaitForSingleObjectEx(data->fenceEvent, INFINITE, FALSE);
        }

        data->fenceValue++;
    }
}

static D3D12_CPU_DESCRIPTOR_HANDLE D3D12_GetCurrentRenderTargetView(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;
    D3D12_CPU_DESCRIPTOR_HANDLE rtvDescriptor;

    if (data->textureRenderTarget) {
        return data->textureRenderTarget->mainTextureRenderTargetView;
    }

    SDL_zero(rtvDescriptor);
    D3D_CALL_RET_ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(data->rtvDescriptorHeap, &rtvDescriptor);
    rtvDescriptor.ptr += data->currentBackBufferIndex * data->rtvDescriptorSize;
    return rtvDescriptor;
}

static void D3D12_TransitionResource(D3D12_RenderData *data,
                                     ID3D12Resource *resource,
                                     D3D12_RESOURCE_STATES beforeState,
                                     D3D12_RESOURCE_STATES afterState)
{
    D3D12_RESOURCE_BARRIER barrier;

    if (beforeState != afterState) {
        SDL_zero(barrier);
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.StateBefore = beforeState;
        barrier.Transition.StateAfter = afterState;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

        D3D_CALL(data->commandList, ResourceBarrier, 1, &barrier);
    }
}

static void D3D12_ResetCommandList(D3D12_RenderData *data)
{
    int i;
    ID3D12DescriptorHeap *rootDescriptorHeaps[] = { data->srvDescriptorHeap, data->samplerDescriptorHeap };
    ID3D12CommandAllocator *commandAllocator = data->commandAllocators[data->currentBackBufferIndex];

    D3D_CALL(commandAllocator, Reset);
    D3D_CALL(data->commandList, Reset, commandAllocator, NULL);
    data->currentPipelineState = NULL;
    data->currentVertexBuffer = 0;
    data->issueBatch = SDL_FALSE;
    data->cliprectDirty = SDL_TRUE;
    data->viewportDirty = SDL_TRUE;
    data->currentRenderTargetView.ptr = 0;

    /* Release any upload buffers that were inflight */
    for (i = 0; i < data->currentUploadBuffer; ++i) {
        SAFE_RELEASE(data->uploadBuffers[i]);
    }
    data->currentUploadBuffer = 0;

    D3D_CALL(data->commandList, SetDescriptorHeaps, 2, rootDescriptorHeaps);
}

static int D3D12_IssueBatch(D3D12_RenderData *data)
{
    HRESULT result = S_OK;

    /* Issue the command list */
    result = D3D_CALL(data->commandList, Close);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("D3D12_IssueBatch"), result);
        return result;
    }
    D3D_CALL(data->commandQueue, ExecuteCommandLists, 1, (ID3D12CommandList *const *)&data->commandList);

    D3D12_WaitForGPU(data);

    D3D12_ResetCommandList(data);

    return result;
}

static void D3D12_DestroyRenderer(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;
    D3D12_WaitForGPU(data);
    D3D12_ReleaseAll(renderer);
    if (data) {
        SDL_free(data);
    }
    SDL_free(renderer);
}

static int D3D12_GetOutputSize(SDL_Renderer *renderer, int *w, int *h)
{
    SDL_GetWindowSizeInPixels(renderer->window, w, h);
    return 0;
}

static D3D12_BLEND GetBlendFunc(SDL_BlendFactor factor)
{
    switch (factor) {
    case SDL_BLENDFACTOR_ZERO:
        return D3D12_BLEND_ZERO;
    case SDL_BLENDFACTOR_ONE:
        return D3D12_BLEND_ONE;
    case SDL_BLENDFACTOR_SRC_COLOR:
        return D3D12_BLEND_SRC_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_COLOR:
        return D3D12_BLEND_INV_SRC_COLOR;
    case SDL_BLENDFACTOR_SRC_ALPHA:
        return D3D12_BLEND_SRC_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA:
        return D3D12_BLEND_INV_SRC_ALPHA;
    case SDL_BLENDFACTOR_DST_COLOR:
        return D3D12_BLEND_DEST_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_COLOR:
        return D3D12_BLEND_INV_DEST_COLOR;
    case SDL_BLENDFACTOR_DST_ALPHA:
        return D3D12_BLEND_DEST_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_ALPHA:
        return D3D12_BLEND_INV_DEST_ALPHA;
    default:
        return (D3D12_BLEND)0;
    }
}

static D3D12_BLEND_OP GetBlendEquation(SDL_BlendOperation operation)
{
    switch (operation) {
    case SDL_BLENDOPERATION_ADD:
        return D3D12_BLEND_OP_ADD;
    case SDL_BLENDOPERATION_SUBTRACT:
        return D3D12_BLEND_OP_SUBTRACT;
    case SDL_BLENDOPERATION_REV_SUBTRACT:
        return D3D12_BLEND_OP_REV_SUBTRACT;
    case SDL_BLENDOPERATION_MINIMUM:
        return D3D12_BLEND_OP_MIN;
    case SDL_BLENDOPERATION_MAXIMUM:
        return D3D12_BLEND_OP_MAX;
    default:
        return (D3D12_BLEND_OP)0;
    }
}

static void D3D12_CreateBlendState(SDL_Renderer *renderer, SDL_BlendMode blendMode, D3D12_BLEND_DESC *outBlendDesc)
{
    SDL_BlendFactor srcColorFactor = SDL_GetBlendModeSrcColorFactor(blendMode);
    SDL_BlendFactor srcAlphaFactor = SDL_GetBlendModeSrcAlphaFactor(blendMode);
    SDL_BlendOperation colorOperation = SDL_GetBlendModeColorOperation(blendMode);
    SDL_BlendFactor dstColorFactor = SDL_GetBlendModeDstColorFactor(blendMode);
    SDL_BlendFactor dstAlphaFactor = SDL_GetBlendModeDstAlphaFactor(blendMode);
    SDL_BlendOperation alphaOperation = SDL_GetBlendModeAlphaOperation(blendMode);

    SDL_zerop(outBlendDesc);
    outBlendDesc->AlphaToCoverageEnable = FALSE;
    outBlendDesc->IndependentBlendEnable = FALSE;
    outBlendDesc->RenderTarget[0].BlendEnable = TRUE;
    outBlendDesc->RenderTarget[0].SrcBlend = GetBlendFunc(srcColorFactor);
    outBlendDesc->RenderTarget[0].DestBlend = GetBlendFunc(dstColorFactor);
    outBlendDesc->RenderTarget[0].BlendOp = GetBlendEquation(colorOperation);
    outBlendDesc->RenderTarget[0].SrcBlendAlpha = GetBlendFunc(srcAlphaFactor);
    outBlendDesc->RenderTarget[0].DestBlendAlpha = GetBlendFunc(dstAlphaFactor);
    outBlendDesc->RenderTarget[0].BlendOpAlpha = GetBlendEquation(alphaOperation);
    outBlendDesc->RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
}

static D3D12_PipelineState *D3D12_CreatePipelineState(SDL_Renderer *renderer,
                                                      D3D12_Shader shader,
                                                      SDL_BlendMode blendMode,
                                                      D3D12_PRIMITIVE_TOPOLOGY_TYPE topology,
                                                      DXGI_FORMAT rtvFormat)
{
    const D3D12_INPUT_ELEMENT_DESC vertexDesc[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipelineDesc;
    ID3D12PipelineState *pipelineState = NULL;
    D3D12_PipelineState *pipelineStates;
    HRESULT result = S_OK;

    SDL_zero(pipelineDesc);
    pipelineDesc.pRootSignature = data->rootSignatures[D3D12_GetRootSignatureType(shader)];
    D3D12_GetVertexShader(shader, &pipelineDesc.VS);
    D3D12_GetPixelShader(shader, &pipelineDesc.PS);
    D3D12_CreateBlendState(renderer, blendMode, &pipelineDesc.BlendState);
    pipelineDesc.SampleMask = 0xffffffff;

    pipelineDesc.RasterizerState.AntialiasedLineEnable = FALSE;
    pipelineDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pipelineDesc.RasterizerState.DepthBias = 0;
    pipelineDesc.RasterizerState.DepthBiasClamp = 0.0f;
    pipelineDesc.RasterizerState.DepthClipEnable = TRUE;
    pipelineDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pipelineDesc.RasterizerState.FrontCounterClockwise = FALSE;
    pipelineDesc.RasterizerState.MultisampleEnable = FALSE;
    pipelineDesc.RasterizerState.SlopeScaledDepthBias = 0.0f;

    pipelineDesc.InputLayout.pInputElementDescs = vertexDesc;
    pipelineDesc.InputLayout.NumElements = 3;

    pipelineDesc.PrimitiveTopologyType = topology;

    pipelineDesc.NumRenderTargets = 1;
    pipelineDesc.RTVFormats[0] = rtvFormat;
    pipelineDesc.SampleDesc.Count = 1;
    pipelineDesc.SampleDesc.Quality = 0;

    result = D3D_CALL(data->d3dDevice, CreateGraphicsPipelineState,
                      &pipelineDesc,
                      D3D_GUID(SDL_IID_ID3D12PipelineState),
                      (void **)&pipelineState);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateGraphicsPipelineState"), result);
        return NULL;
    }

    pipelineStates = (D3D12_PipelineState *)SDL_realloc(data->pipelineStates, (data->pipelineStateCount + 1) * sizeof(*pipelineStates));
    if (!pipelineStates) {
        SAFE_RELEASE(pipelineState);
        SDL_OutOfMemory();
        return NULL;
    }

    pipelineStates[data->pipelineStateCount].shader = shader;
    pipelineStates[data->pipelineStateCount].blendMode = blendMode;
    pipelineStates[data->pipelineStateCount].topology = topology;
    pipelineStates[data->pipelineStateCount].rtvFormat = rtvFormat;
    pipelineStates[data->pipelineStateCount].pipelineState = pipelineState;
    data->pipelineStates = pipelineStates;
    ++data->pipelineStateCount;

    return &pipelineStates[data->pipelineStateCount - 1];
}

static HRESULT D3D12_CreateVertexBuffer(D3D12_RenderData *data, size_t vbidx, size_t size)
{
    D3D12_HEAP_PROPERTIES vbufferHeapProps;
    D3D12_RESOURCE_DESC vbufferDesc;
    HRESULT result;

    SAFE_RELEASE(data->vertexBuffers[vbidx].resource);

    SDL_zero(vbufferHeapProps);
    vbufferHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
    vbufferHeapProps.CreationNodeMask = 1;
    vbufferHeapProps.VisibleNodeMask = 1;

    SDL_zero(vbufferDesc);
    vbufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    vbufferDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    vbufferDesc.Width = size;
    vbufferDesc.Height = 1;
    vbufferDesc.DepthOrArraySize = 1;
    vbufferDesc.MipLevels = 1;
    vbufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    vbufferDesc.SampleDesc.Count = 1;
    vbufferDesc.SampleDesc.Quality = 0;
    vbufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    vbufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    result = D3D_CALL(data->d3dDevice, CreateCommittedResource,
                      &vbufferHeapProps,
                      D3D12_HEAP_FLAG_NONE,
                      &vbufferDesc,
                      D3D12_RESOURCE_STATE_GENERIC_READ,
                      NULL,
                      D3D_GUID(SDL_IID_ID3D12Resource),
                      (void **)&data->vertexBuffers[vbidx].resource);

    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreatePlacedResource [vertex buffer]"), result);
        return result;
    }

    data->vertexBuffers[vbidx].view.BufferLocation = D3D_CALL(data->vertexBuffers[vbidx].resource, GetGPUVirtualAddress);
    data->vertexBuffers[vbidx].view.StrideInBytes = sizeof(VertexPositionColor);
    data->vertexBuffers[vbidx].size = size;

    return result;
}

/* Create resources that depend on the device. */
static HRESULT D3D12_CreateDeviceResources(SDL_Renderer *renderer)
{
#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    typedef HRESULT(WINAPI * PFN_CREATE_DXGI_FACTORY)(UINT flags, REFIID riid, void **ppFactory);
    PFN_CREATE_DXGI_FACTORY CreateDXGIFactoryFunc;
    PFN_D3D12_CREATE_DEVICE D3D12CreateDeviceFunc;
#endif
    typedef HANDLE(WINAPI * PFN_CREATE_EVENT_EX)(LPSECURITY_ATTRIBUTES lpEventAttributes, LPCWSTR lpName, DWORD dwFlags, DWORD dwDesiredAccess);
    PFN_CREATE_EVENT_EX CreateEventExFunc;

    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;
    ID3D12Device *d3dDevice = NULL;
    HRESULT result = S_OK;
    UINT creationFlags = 0;
    int i, j, k, l;
    SDL_bool createDebug = SDL_FALSE;

    D3D12_COMMAND_QUEUE_DESC queueDesc;
    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc;
    D3D12_SAMPLER_DESC samplerDesc;
    ID3D12DescriptorHeap *rootDescriptorHeaps[2];

    const SDL_BlendMode defaultBlendModes[] = {
        SDL_BLENDMODE_NONE,
        SDL_BLENDMODE_BLEND,
        SDL_BLENDMODE_ADD,
        SDL_BLENDMODE_MOD,
        SDL_BLENDMODE_MUL
    };
    const DXGI_FORMAT defaultRTVFormats[] = {
        DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_B8G8R8X8_UNORM,
        DXGI_FORMAT_R8_UNORM
    };

    /* See if we need debug interfaces */
    createDebug = SDL_GetHintBoolean(SDL_HINT_RENDER_DIRECT3D11_DEBUG, SDL_FALSE);

#ifdef __GDK__
    CreateEventExFunc = CreateEventExW;
#else
    /* CreateEventEx() arrived in Vista, so we need to load it with GetProcAddress for XP. */
    {
        HMODULE kernel32 = GetModuleHandle(TEXT("kernel32.dll"));
        CreateEventExFunc = NULL;
        if (kernel32) {
            CreateEventExFunc = (PFN_CREATE_EVENT_EX)GetProcAddress(kernel32, "CreateEventExW");
        }
    }
#endif
    if (!CreateEventExFunc) {
        result = E_FAIL;
        goto done;
    }

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    data->hDXGIMod = SDL_LoadObject("dxgi.dll");
    if (!data->hDXGIMod) {
        result = E_FAIL;
        goto done;
    }

    CreateDXGIFactoryFunc = (PFN_CREATE_DXGI_FACTORY)SDL_LoadFunction(data->hDXGIMod, "CreateDXGIFactory2");
    if (!CreateDXGIFactoryFunc) {
        result = E_FAIL;
        goto done;
    }

    data->hD3D12Mod = SDL_LoadObject("D3D12.dll");
    if (!data->hD3D12Mod) {
        result = E_FAIL;
        goto done;
    }

    D3D12CreateDeviceFunc = (PFN_D3D12_CREATE_DEVICE)SDL_LoadFunction(data->hD3D12Mod, "D3D12CreateDevice");
    if (!D3D12CreateDeviceFunc) {
        result = E_FAIL;
        goto done;
    }

    if (createDebug) {
        PFN_D3D12_GET_DEBUG_INTERFACE D3D12GetDebugInterfaceFunc;

        D3D12GetDebugInterfaceFunc = (PFN_D3D12_GET_DEBUG_INTERFACE)SDL_LoadFunction(data->hD3D12Mod, "D3D12GetDebugInterface");
        if (!D3D12GetDebugInterfaceFunc) {
            result = E_FAIL;
            goto done;
        }
        D3D12GetDebugInterfaceFunc(D3D_GUID(SDL_IID_ID3D12Debug), (void **)&data->debugInterface);
        D3D_CALL(data->debugInterface, EnableDebugLayer);
    }
#endif /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/

#if defined(__XBOXONE__) || defined(__XBOXSERIES__)
    result = D3D12_XBOX_CreateDevice(&d3dDevice, createDebug);
    if (FAILED(result)) {
        /* SDL Error is set by D3D12_XBOX_CreateDevice */
        goto done;
    }
#else
    if (createDebug) {
#ifdef __IDXGIInfoQueue_INTERFACE_DEFINED__
        IDXGIInfoQueue *dxgiInfoQueue = NULL;
        PFN_CREATE_DXGI_FACTORY DXGIGetDebugInterfaceFunc;

        /* If the debug hint is set, also create the DXGI factory in debug mode */
        DXGIGetDebugInterfaceFunc = (PFN_CREATE_DXGI_FACTORY)SDL_LoadFunction(data->hDXGIMod, "DXGIGetDebugInterface1");
        if (!DXGIGetDebugInterfaceFunc) {
            result = E_FAIL;
            goto done;
        }

        result = DXGIGetDebugInterfaceFunc(0, D3D_GUID(SDL_IID_IDXGIDebug1), (void **)&data->dxgiDebug);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("DXGIGetDebugInterface1"), result);
            goto done;
        }

        result = DXGIGetDebugInterfaceFunc(0, D3D_GUID(SDL_IID_IDXGIInfoQueue), (void **)&dxgiInfoQueue);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("DXGIGetDebugInterface1"), result);
            goto done;
        }

        D3D_CALL(dxgiInfoQueue, SetBreakOnSeverity, SDL_DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR, TRUE);
        D3D_CALL(dxgiInfoQueue, SetBreakOnSeverity, SDL_DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION, TRUE);
        SAFE_RELEASE(dxgiInfoQueue);
#endif /* __IDXGIInfoQueue_INTERFACE_DEFINED__ */
        creationFlags = DXGI_CREATE_FACTORY_DEBUG;
    }

    result = CreateDXGIFactoryFunc(creationFlags, D3D_GUID(SDL_IID_IDXGIFactory6), (void **)&data->dxgiFactory);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("CreateDXGIFactory"), result);
        goto done;
    }

    /* Prefer a high performance adapter if there are multiple choices */
    result = D3D_CALL(data->dxgiFactory, EnumAdapterByGpuPreference,
                      0,
                      DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                      D3D_GUID(SDL_IID_IDXGIAdapter4),
                      (void **)&data->dxgiAdapter);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("D3D12CreateDevice"), result);
        goto done;
    }

    result = D3D12CreateDeviceFunc((IUnknown *)data->dxgiAdapter,
                                   D3D_FEATURE_LEVEL_11_0, /* Request minimum feature level 11.0 for maximum compatibility */
                                   D3D_GUID(SDL_IID_ID3D12Device1),
                                   (void **)&d3dDevice);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("D3D12CreateDevice"), result);
        goto done;
    }

    /* Setup the info queue if in debug mode */
    if (createDebug) {
        ID3D12InfoQueue *infoQueue = NULL;
        D3D12_MESSAGE_SEVERITY severities[] = { D3D12_MESSAGE_SEVERITY_INFO };
        D3D12_INFO_QUEUE_FILTER filter;

        result = D3D_CALL(d3dDevice, QueryInterface, D3D_GUID(SDL_IID_ID3D12InfoQueue), (void **)&infoQueue);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device to ID3D12InfoQueue"), result);
            goto done;
        }

        SDL_zero(filter);
        filter.DenyList.NumSeverities = 1;
        filter.DenyList.pSeverityList = severities;
        D3D_CALL(infoQueue, PushStorageFilter, &filter);

        D3D_CALL(infoQueue, SetBreakOnSeverity, D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
        D3D_CALL(infoQueue, SetBreakOnSeverity, D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);

        SAFE_RELEASE(infoQueue);
    }
#endif /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/

    result = D3D_CALL(d3dDevice, QueryInterface, D3D_GUID(SDL_IID_ID3D12Device1), (void **)&data->d3dDevice);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device to ID3D12Device1"), result);
        goto done;
    }

    /* Create a command queue */
    SDL_zero(queueDesc);
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    result = D3D_CALL(data->d3dDevice, CreateCommandQueue,
                      &queueDesc,
                      D3D_GUID(SDL_IID_ID3D12CommandQueue),
                      (void **)&data->commandQueue);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommandQueue"), result);
        goto done;
    }

    /* Create the descriptor heaps for the render target view, texture SRVs, and samplers */
    SDL_zero(descriptorHeapDesc);
    descriptorHeapDesc.NumDescriptors = SDL_D3D12_NUM_BUFFERS;
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    result = D3D_CALL(data->d3dDevice, CreateDescriptorHeap,
                      &descriptorHeapDesc,
                      D3D_GUID(SDL_IID_ID3D12DescriptorHeap),
                      (void **)&data->rtvDescriptorHeap);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateDescriptorHeap [rtv]"), result);
        goto done;
    }
    data->rtvDescriptorSize = D3D_CALL(d3dDevice, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    descriptorHeapDesc.NumDescriptors = SDL_D3D12_MAX_NUM_TEXTURES;
    result = D3D_CALL(data->d3dDevice, CreateDescriptorHeap,
                      &descriptorHeapDesc,
                      D3D_GUID(SDL_IID_ID3D12DescriptorHeap),
                      (void **)&data->textureRTVDescriptorHeap);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateDescriptorHeap [texture rtv]"), result);
        goto done;
    }

    SDL_zero(descriptorHeapDesc);
    descriptorHeapDesc.NumDescriptors = SDL_D3D12_MAX_NUM_TEXTURES;
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    result = D3D_CALL(data->d3dDevice, CreateDescriptorHeap,
                      &descriptorHeapDesc,
                      D3D_GUID(SDL_IID_ID3D12DescriptorHeap),
                      (void **)&data->srvDescriptorHeap);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateDescriptorHeap  [srv]"), result);
        goto done;
    }
    rootDescriptorHeaps[0] = data->srvDescriptorHeap;
    data->srvDescriptorSize = D3D_CALL(d3dDevice, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    SDL_zero(descriptorHeapDesc);
    descriptorHeapDesc.NumDescriptors = 2;
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    result = D3D_CALL(data->d3dDevice, CreateDescriptorHeap,
                      &descriptorHeapDesc,
                      D3D_GUID(SDL_IID_ID3D12DescriptorHeap),
                      (void **)&data->samplerDescriptorHeap);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateDescriptorHeap  [sampler]"), result);
        goto done;
    }
    rootDescriptorHeaps[1] = data->samplerDescriptorHeap;
    data->samplerDescriptorSize = D3D_CALL(d3dDevice, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    /* Create a command allocator for each back buffer */
    for (i = 0; i < SDL_D3D12_NUM_BUFFERS; ++i) {
        result = D3D_CALL(data->d3dDevice, CreateCommandAllocator,
                          D3D12_COMMAND_LIST_TYPE_DIRECT,
                          D3D_GUID(SDL_IID_ID3D12CommandAllocator),
                          (void **)&data->commandAllocators[i]);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommandAllocator"), result);
            goto done;
        }
    }

    /* Create the command list */
    result = D3D_CALL(data->d3dDevice, CreateCommandList,
                      0,
                      D3D12_COMMAND_LIST_TYPE_DIRECT,
                      data->commandAllocators[0],
                      NULL,
                      D3D_GUID(SDL_IID_ID3D12GraphicsCommandList2),
                      (void **)&data->commandList);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommandList"), result);
        goto done;
    }

    /* Set the descriptor heaps to the correct initial value */
    D3D_CALL(data->commandList, SetDescriptorHeaps, 2, rootDescriptorHeaps);

    /* Create the fence and fence event */
    result = D3D_CALL(data->d3dDevice, CreateFence,
                      data->fenceValue,
                      D3D12_FENCE_FLAG_NONE,
                      D3D_GUID(SDL_IID_ID3D12Fence),
                      (void **)&data->fence);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateFence"), result);
        goto done;
    }

    data->fenceValue++;

    data->fenceEvent = CreateEventExFunc(NULL, NULL, 0, EVENT_MODIFY_STATE | SYNCHRONIZE);
    if (!data->fenceEvent) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("CreateEventEx"), result);
        goto done;
    }

    /* Create all the root signatures */
    for (i = 0; i < NUM_ROOTSIGS; ++i) {
        D3D12_SHADER_BYTECODE rootSigData;
        D3D12_GetRootSignatureData((D3D12_RootSignature)i, &rootSigData);
        result = D3D_CALL(data->d3dDevice, CreateRootSignature,
                          0,
                          rootSigData.pShaderBytecode,
                          rootSigData.BytecodeLength,
                          D3D_GUID(SDL_IID_ID3D12RootSignature),
                          (void **)&data->rootSignatures[i]);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateRootSignature"), result);
            goto done;
        }
    }

    /* Create all the default pipeline state objects
       (will add everything except custom blend states) */
    for (i = 0; i < NUM_SHADERS; ++i) {
        for (j = 0; j < SDL_arraysize(defaultBlendModes); ++j) {
            for (k = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT; k < D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH; ++k) {
                for (l = 0; l < SDL_arraysize(defaultRTVFormats); ++l) {
                    if (!D3D12_CreatePipelineState(renderer, (D3D12_Shader)i, defaultBlendModes[j], (D3D12_PRIMITIVE_TOPOLOGY_TYPE)k, defaultRTVFormats[l])) {
                        /* D3D12_CreatePipelineState will set the SDL error, if it fails */
                        goto done;
                    }
                }
            }
        }
    }

    /* Create default vertex buffers  */
    for (i = 0; i < SDL_D3D12_NUM_VERTEX_BUFFERS; ++i) {
        D3D12_CreateVertexBuffer(data, i, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
    }

    /* Create samplers to use when drawing textures: */
    SDL_zero(samplerDesc);
    samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    samplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplerDesc.MipLODBias = 0.0f;
    samplerDesc.MaxAnisotropy = 1;
    samplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    samplerDesc.MinLOD = 0.0f;
    samplerDesc.MaxLOD = D3D12_FLOAT32_MAX;
    D3D_CALL_RET_ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(data->samplerDescriptorHeap, &data->nearestPixelSampler);
    D3D_CALL(data->d3dDevice, CreateSampler, &samplerDesc, data->nearestPixelSampler);

    samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    data->linearSampler.ptr = data->nearestPixelSampler.ptr + data->samplerDescriptorSize;
    D3D_CALL(data->d3dDevice, CreateSampler, &samplerDesc, data->linearSampler);

    /* Initialize the pool allocator for SRVs */
    for (i = 0; i < SDL_D3D12_MAX_NUM_TEXTURES; ++i) {
        data->srvPoolNodes[i].index = (SIZE_T)i;
        if (i != SDL_D3D12_MAX_NUM_TEXTURES - 1) {
            data->srvPoolNodes[i].next = &data->srvPoolNodes[i + 1];
        }
    }
    data->srvPoolHead = &data->srvPoolNodes[0];
done:
    SAFE_RELEASE(d3dDevice);
    return result;
}

static DXGI_MODE_ROTATION D3D12_GetCurrentRotation()
{
    /* FIXME */
    return DXGI_MODE_ROTATION_IDENTITY;
}

static BOOL D3D12_IsDisplayRotated90Degrees(DXGI_MODE_ROTATION rotation)
{
    switch (rotation) {
    case DXGI_MODE_ROTATION_ROTATE90:
    case DXGI_MODE_ROTATION_ROTATE270:
        return TRUE;
    default:
        return FALSE;
    }
}

static int D3D12_GetRotationForCurrentRenderTarget(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;
    if (data->textureRenderTarget) {
        return DXGI_MODE_ROTATION_IDENTITY;
    } else {
        return data->rotation;
    }
}

static int D3D12_GetViewportAlignedD3DRect(SDL_Renderer *renderer, const SDL_Rect *sdlRect, D3D12_RECT *outRect, BOOL includeViewportOffset)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;
    const int rotation = D3D12_GetRotationForCurrentRenderTarget(renderer);
    const SDL_Rect *viewport = &data->currentViewport;

    switch (rotation) {
    case DXGI_MODE_ROTATION_IDENTITY:
        outRect->left = sdlRect->x;
        outRect->right = (LONG)sdlRect->x + sdlRect->w;
        outRect->top = sdlRect->y;
        outRect->bottom = (LONG)sdlRect->y + sdlRect->h;
        if (includeViewportOffset) {
            outRect->left += viewport->x;
            outRect->right += viewport->x;
            outRect->top += viewport->y;
            outRect->bottom += viewport->y;
        }
        break;
    case DXGI_MODE_ROTATION_ROTATE270:
        outRect->left = sdlRect->y;
        outRect->right = (LONG)sdlRect->y + sdlRect->h;
        outRect->top = viewport->w - sdlRect->x - sdlRect->w;
        outRect->bottom = viewport->w - sdlRect->x;
        break;
    case DXGI_MODE_ROTATION_ROTATE180:
        outRect->left = viewport->w - sdlRect->x - sdlRect->w;
        outRect->right = viewport->w - sdlRect->x;
        outRect->top = viewport->h - sdlRect->y - sdlRect->h;
        outRect->bottom = viewport->h - sdlRect->y;
        break;
    case DXGI_MODE_ROTATION_ROTATE90:
        outRect->left = viewport->h - sdlRect->y - sdlRect->h;
        outRect->right = viewport->h - sdlRect->y;
        outRect->top = sdlRect->x;
        outRect->bottom = (LONG)sdlRect->x + sdlRect->h;
        break;
    default:
        return SDL_SetError("The physical display is in an unknown or unsupported rotation");
    }
    return 0;
}

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
static HRESULT D3D12_CreateSwapChain(SDL_Renderer *renderer, int w, int h)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;
    IDXGISwapChain1* swapChain = NULL;
    HRESULT result = S_OK;
    SDL_SysWMinfo windowinfo;

    /* Create a swap chain using the same adapter as the existing Direct3D device. */
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc;
    SDL_zero(swapChainDesc);
    swapChainDesc.Width = w;
    swapChainDesc.Height = h;
    swapChainDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM; /* This is the most common swap chain format. */
    swapChainDesc.Stereo = FALSE;
    swapChainDesc.SampleDesc.Count = 1; /* Don't use multi-sampling. */
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferCount = 2; /* Use double-buffering to minimize latency. */
    if (WIN_IsWindows8OrGreater()) {
        swapChainDesc.Scaling = DXGI_SCALING_NONE;
    } else {
        swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
    }
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;               /* All Windows Store apps must use this SwapEffect. */
    swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT | /* To support SetMaximumFrameLatency */
                          DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;                  /* To support presenting with allow tearing on */

    SDL_VERSION(&windowinfo.version);
    if (!SDL_GetWindowWMInfo(renderer->window, &windowinfo) ||
        windowinfo.subsystem != SDL_SYSWM_WINDOWS)
    {
        SDL_SetError("Couldn't get window handle");
        result = E_FAIL;
        goto done;
    }

    result = D3D_CALL(data->dxgiFactory, CreateSwapChainForHwnd,
                      (IUnknown *)data->commandQueue,
                      windowinfo.info.win.window,
                      &swapChainDesc,
                      NULL,
                      NULL, /* Allow on all displays. */
                      &swapChain);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGIFactory2::CreateSwapChainForHwnd"), result);
        goto done;
    }

    D3D_CALL(data->dxgiFactory, MakeWindowAssociation, windowinfo.info.win.window, DXGI_MWA_NO_WINDOW_CHANGES);

    result = D3D_CALL(swapChain, QueryInterface, D3D_GUID(SDL_IID_IDXGISwapChain4), (void **)&data->swapChain);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain1::QueryInterface"), result);
        goto done;
    }

    /* Ensure that the swapchain does not queue more than one frame at a time. This both reduces latency
     * and ensures that the application will only render after each VSync, minimizing power consumption.
     */
    result = D3D_CALL(data->swapChain, SetMaximumFrameLatency, 1);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain4::SetMaximumFrameLatency"), result);
        goto done;
    }

    data->swapEffect = swapChainDesc.SwapEffect;
    data->swapFlags = swapChainDesc.Flags;

done:
    SAFE_RELEASE(swapChain);
    return result;
}
#endif

static HRESULT D3D12_UpdateForWindowSizeChange(SDL_Renderer *renderer);

HRESULT
D3D12_HandleDeviceLost(SDL_Renderer *renderer)
{
    HRESULT result = S_OK;

    D3D12_ReleaseAll(renderer);

    result = D3D12_CreateDeviceResources(renderer);
    if (FAILED(result)) {
        /* D3D12_CreateDeviceResources will set the SDL error */
        return result;
    }

    result = D3D12_UpdateForWindowSizeChange(renderer);
    if (FAILED(result)) {
        /* D3D12_UpdateForWindowSizeChange will set the SDL error */
        return result;
    }

    /* Let the application know that the device has been reset */
    {
        SDL_Event event;
        event.type = SDL_RENDER_DEVICE_RESET;
        SDL_PushEvent(&event);
    }

    return S_OK;
}

/* Initialize all resources that change when the window's size changes. */
static HRESULT D3D12_CreateWindowSizeDependentResources(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;
    HRESULT result = S_OK;
    int i, w, h;

    D3D12_RENDER_TARGET_VIEW_DESC rtvDesc;
    D3D12_CPU_DESCRIPTOR_HANDLE rtvDescriptor;

    /* Release resources in the current command list */
    D3D12_IssueBatch(data);
    D3D_CALL(data->commandList, OMSetRenderTargets, 0, NULL, FALSE, NULL);

    /* Release render targets */
    for (i = 0; i < SDL_D3D12_NUM_BUFFERS; ++i) {
        SAFE_RELEASE(data->renderTargets[i]);
    }

    /* The width and height of the swap chain must be based on the display's
     * non-rotated size.
     */
    SDL_GetWindowSizeInPixels(renderer->window, &w, &h);
    data->rotation = D3D12_GetCurrentRotation();
    if (D3D12_IsDisplayRotated90Degrees(data->rotation)) {
        int tmp = w;
        w = h;
        h = tmp;
    }

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    if (data->swapChain) {
        /* If the swap chain already exists, resize it. */
        result = D3D_CALL(data->swapChain, ResizeBuffers,
                          0,
                          w, h,
                          DXGI_FORMAT_UNKNOWN,
                          data->swapFlags);
        if (result == DXGI_ERROR_DEVICE_REMOVED) {
            /* If the device was removed for any reason, a new device and swap chain will need to be created. */
            D3D12_HandleDeviceLost(renderer);

            /* Everything is set up now. Do not continue execution of this method. HandleDeviceLost will reenter this method
             * and correctly set up the new device.
             */
            goto done;
        } else if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain::ResizeBuffers"), result);
            goto done;
        }
    } else {
        result = D3D12_CreateSwapChain(renderer, w, h);
        if (FAILED(result)) {
            goto done;
        }
    }

    /* Set the proper rotation for the swap chain. */
    if (WIN_IsWindows8OrGreater()) {
        if (data->swapEffect == DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL) {
            result = D3D_CALL(data->swapChain, SetRotation, data->rotation); /* NOLINT(clang-analyzer-core.NullDereference) */
            if (FAILED(result)) {
                WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain4::SetRotation"), result);
                goto done;
            }
        }
    }
#endif /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/

    /* Get each back buffer render target and create render target views */
    for (i = 0; i < SDL_D3D12_NUM_BUFFERS; ++i) {
#if defined(__XBOXONE__) || defined(__XBOXSERIES__)
        result = D3D12_XBOX_CreateBackBufferTarget(data->d3dDevice, renderer->window->w, renderer->window->h, (void **)&data->renderTargets[i]);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("D3D12_XBOX_CreateBackBufferTarget"), result);
            goto done;
        }
#else
        result = D3D_CALL(data->swapChain, GetBuffer, /* NOLINT(clang-analyzer-core.NullDereference) */
                          i,
                          D3D_GUID(SDL_IID_ID3D12Resource),
                          (void **)&data->renderTargets[i]);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain4::GetBuffer"), result);
            goto done;
        }
#endif

        SDL_zero(rtvDesc);
        rtvDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;

        SDL_zero(rtvDescriptor);
        D3D_CALL_RET_ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(data->rtvDescriptorHeap, &rtvDescriptor);
        rtvDescriptor.ptr += i * data->rtvDescriptorSize;
        D3D_CALL(data->d3dDevice, CreateRenderTargetView, data->renderTargets[i], &rtvDesc, rtvDescriptor);
    }

    /* Set back buffer index to current buffer */
#if defined(__XBOXONE__) || defined(__XBOXSERIES__)
    data->currentBackBufferIndex = 0;
#else
    data->currentBackBufferIndex = D3D_CALL(data->swapChain, GetCurrentBackBufferIndex);
#endif

    /* Set the swap chain target immediately, so that a target is always set
     * even before we get to SetDrawState. Without this it's possible to hit
     * null references in places like ReadPixels!
     */
    data->currentRenderTargetView = D3D12_GetCurrentRenderTargetView(renderer);
    D3D_CALL(data->commandList, OMSetRenderTargets, 1, &data->currentRenderTargetView, FALSE, NULL);
    D3D12_TransitionResource(data,
                             data->renderTargets[data->currentBackBufferIndex],
                             D3D12_RESOURCE_STATE_PRESENT,
                             D3D12_RESOURCE_STATE_RENDER_TARGET);

    data->viewportDirty = SDL_TRUE;

#if defined(__XBOXONE__) || defined(__XBOXSERIES__)
    D3D12_XBOX_StartFrame(data->d3dDevice, &data->frameToken);
#endif

done:
    return result;
}

/* This method is called when the window's size changes. */
static HRESULT D3D12_UpdateForWindowSizeChange(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;
    /* If the GPU has previous work, wait for it to be done first */
    D3D12_WaitForGPU(data);
    return D3D12_CreateWindowSizeDependentResources(renderer);
}

static void D3D12_WindowEvent(SDL_Renderer *renderer, const SDL_WindowEvent *event)
{
    if (event->event == SDL_WINDOWEVENT_SIZE_CHANGED) {
        D3D12_UpdateForWindowSizeChange(renderer);
    }
}

static SDL_bool D3D12_SupportsBlendMode(SDL_Renderer *renderer, SDL_BlendMode blendMode)
{
    SDL_BlendFactor srcColorFactor = SDL_GetBlendModeSrcColorFactor(blendMode);
    SDL_BlendFactor srcAlphaFactor = SDL_GetBlendModeSrcAlphaFactor(blendMode);
    SDL_BlendOperation colorOperation = SDL_GetBlendModeColorOperation(blendMode);
    SDL_BlendFactor dstColorFactor = SDL_GetBlendModeDstColorFactor(blendMode);
    SDL_BlendFactor dstAlphaFactor = SDL_GetBlendModeDstAlphaFactor(blendMode);
    SDL_BlendOperation alphaOperation = SDL_GetBlendModeAlphaOperation(blendMode);

    if (!GetBlendFunc(srcColorFactor) || !GetBlendFunc(srcAlphaFactor) ||
        !GetBlendEquation(colorOperation) ||
        !GetBlendFunc(dstColorFactor) || !GetBlendFunc(dstAlphaFactor) ||
        !GetBlendEquation(alphaOperation)) {
        return SDL_FALSE;
    }
    return SDL_TRUE;
}

static SIZE_T D3D12_GetAvailableSRVIndex(SDL_Renderer *renderer)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    if (rendererData->srvPoolHead) {
        SIZE_T index = rendererData->srvPoolHead->index;
        rendererData->srvPoolHead = (D3D12_SRVPoolNode *)(rendererData->srvPoolHead->next);
        return index;
    } else {
        SDL_SetError("[d3d12] Cannot allocate more than %d textures!", SDL_D3D12_MAX_NUM_TEXTURES);
        return SDL_D3D12_MAX_NUM_TEXTURES + 1;
    }
}

static void D3D12_FreeSRVIndex(SDL_Renderer *renderer, SIZE_T index)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    rendererData->srvPoolNodes[index].next = rendererData->srvPoolHead;
    rendererData->srvPoolHead = &rendererData->srvPoolNodes[index];
}

static int D3D12_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    D3D12_TextureData *textureData;
    HRESULT result;
    DXGI_FORMAT textureFormat = SDLPixelFormatToDXGIFormat(texture->format);
    D3D12_RESOURCE_DESC textureDesc;
    D3D12_HEAP_PROPERTIES heapProps;
    D3D12_SHADER_RESOURCE_VIEW_DESC resourceViewDesc;

    if (textureFormat == DXGI_FORMAT_UNKNOWN) {
        return SDL_SetError("%s, An unsupported SDL pixel format (0x%x) was specified", __FUNCTION__, texture->format);
    }

    textureData = (D3D12_TextureData *)SDL_calloc(1, sizeof(*textureData));
    if (!textureData) {
        SDL_OutOfMemory();
        return -1;
    }
    textureData->scaleMode = (texture->scaleMode == SDL_ScaleModeNearest) ? D3D12_FILTER_MIN_MAG_MIP_POINT : D3D12_FILTER_MIN_MAG_MIP_LINEAR;

    texture->driverdata = textureData;
    textureData->mainTextureFormat = textureFormat;

    SDL_zero(textureDesc);
    textureDesc.Width = texture->w;
    textureDesc.Height = texture->h;
    textureDesc.MipLevels = 1;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.Format = textureFormat;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    if (texture->access == SDL_TEXTUREACCESS_TARGET) {
        textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }

    SDL_zero(heapProps);
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    result = D3D_CALL(rendererData->d3dDevice, CreateCommittedResource,
                      &heapProps,
                      D3D12_HEAP_FLAG_NONE,
                      &textureDesc,
                      D3D12_RESOURCE_STATE_COPY_DEST,
                      NULL,
                      D3D_GUID(SDL_IID_ID3D12Resource),
                      (void **)&textureData->mainTexture);
    textureData->mainResourceState = D3D12_RESOURCE_STATE_COPY_DEST;
    if (FAILED(result)) {
        D3D12_DestroyTexture(renderer, texture);
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommittedResource [texture]"), result);
    }
#if SDL_HAVE_YUV
    if (texture->format == SDL_PIXELFORMAT_YV12 ||
        texture->format == SDL_PIXELFORMAT_IYUV) {
        textureData->yuv = SDL_TRUE;

        textureDesc.Width = (textureDesc.Width + 1) / 2;
        textureDesc.Height = (textureDesc.Height + 1) / 2;

        result = D3D_CALL(rendererData->d3dDevice, CreateCommittedResource,
                          &heapProps,
                          D3D12_HEAP_FLAG_NONE,
                          &textureDesc,
                          D3D12_RESOURCE_STATE_COPY_DEST,
                          NULL,
                          D3D_GUID(SDL_IID_ID3D12Resource),
                          (void **)&textureData->mainTextureU);
        textureData->mainResourceStateU = D3D12_RESOURCE_STATE_COPY_DEST;
        if (FAILED(result)) {
            D3D12_DestroyTexture(renderer, texture);
            return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommittedResource [texture]"), result);
        }

        result = D3D_CALL(rendererData->d3dDevice, CreateCommittedResource,
                          &heapProps,
                          D3D12_HEAP_FLAG_NONE,
                          &textureDesc,
                          D3D12_RESOURCE_STATE_COPY_DEST,
                          NULL,
                          D3D_GUID(SDL_IID_ID3D12Resource),
                          (void **)&textureData->mainTextureV);
        textureData->mainResourceStateV = D3D12_RESOURCE_STATE_COPY_DEST;
        if (FAILED(result)) {
            D3D12_DestroyTexture(renderer, texture);
            return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommittedResource [texture]"), result);
        }
    }

    if (texture->format == SDL_PIXELFORMAT_NV12 ||
        texture->format == SDL_PIXELFORMAT_NV21) {
        D3D12_RESOURCE_DESC nvTextureDesc = textureDesc;

        textureData->nv12 = SDL_TRUE;

        nvTextureDesc.Format = DXGI_FORMAT_R8G8_UNORM;
        nvTextureDesc.Width = (textureDesc.Width + 1) / 2;
        nvTextureDesc.Height = (textureDesc.Height + 1) / 2;

        result = D3D_CALL(rendererData->d3dDevice, CreateCommittedResource,
                          &heapProps,
                          D3D12_HEAP_FLAG_NONE,
                          &nvTextureDesc,
                          D3D12_RESOURCE_STATE_COPY_DEST,
                          NULL,
                          D3D_GUID(SDL_IID_ID3D12Resource),
                          (void **)&textureData->mainTextureNV);
        textureData->mainResourceStateNV = D3D12_RESOURCE_STATE_COPY_DEST;
        if (FAILED(result)) {
            D3D12_DestroyTexture(renderer, texture);
            return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateTexture2D"), result);
        }
    }
#endif /* SDL_HAVE_YUV */
    SDL_zero(resourceViewDesc);
    resourceViewDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    resourceViewDesc.Format = textureDesc.Format;
    resourceViewDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    resourceViewDesc.Texture2D.MostDetailedMip = 0;
    resourceViewDesc.Texture2D.MipLevels = textureDesc.MipLevels;

    textureData->mainSRVIndex = D3D12_GetAvailableSRVIndex(renderer);
    D3D_CALL_RET_ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(rendererData->srvDescriptorHeap, &textureData->mainTextureResourceView);
    textureData->mainTextureResourceView.ptr += textureData->mainSRVIndex * rendererData->srvDescriptorSize;

    D3D_CALL(rendererData->d3dDevice, CreateShaderResourceView,
             textureData->mainTexture,
             &resourceViewDesc,
             textureData->mainTextureResourceView);
#if SDL_HAVE_YUV
    if (textureData->yuv) {
        D3D_CALL_RET_ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(rendererData->srvDescriptorHeap, &textureData->mainTextureResourceViewU);
        textureData->mainSRVIndexU = D3D12_GetAvailableSRVIndex(renderer);
        textureData->mainTextureResourceViewU.ptr += textureData->mainSRVIndexU * rendererData->srvDescriptorSize;
        D3D_CALL(rendererData->d3dDevice, CreateShaderResourceView,
                 textureData->mainTextureU,
                 &resourceViewDesc,
                 textureData->mainTextureResourceViewU);

        D3D_CALL_RET_ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(rendererData->srvDescriptorHeap, &textureData->mainTextureResourceViewV);
        textureData->mainSRVIndexV = D3D12_GetAvailableSRVIndex(renderer);
        textureData->mainTextureResourceViewV.ptr += textureData->mainSRVIndexV * rendererData->srvDescriptorSize;
        D3D_CALL(rendererData->d3dDevice, CreateShaderResourceView,
                 textureData->mainTextureV,
                 &resourceViewDesc,
                 textureData->mainTextureResourceViewV);
    }

    if (textureData->nv12) {
        D3D12_SHADER_RESOURCE_VIEW_DESC nvResourceViewDesc = resourceViewDesc;

        nvResourceViewDesc.Format = DXGI_FORMAT_R8G8_UNORM;

        D3D_CALL_RET_ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(rendererData->srvDescriptorHeap, &textureData->mainTextureResourceViewNV);
        textureData->mainSRVIndexNV = D3D12_GetAvailableSRVIndex(renderer);
        textureData->mainTextureResourceViewNV.ptr += textureData->mainSRVIndexNV * rendererData->srvDescriptorSize;
        D3D_CALL(rendererData->d3dDevice, CreateShaderResourceView,
                 textureData->mainTextureNV,
                 &nvResourceViewDesc,
                 textureData->mainTextureResourceViewNV);
    }
#endif /* SDL_HAVE_YUV */

    if (texture->access & SDL_TEXTUREACCESS_TARGET) {
        D3D12_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
        SDL_zero(renderTargetViewDesc);
        renderTargetViewDesc.Format = textureDesc.Format;
        renderTargetViewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        renderTargetViewDesc.Texture2D.MipSlice = 0;

        D3D_CALL_RET_ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(rendererData->textureRTVDescriptorHeap, &textureData->mainTextureRenderTargetView);
        textureData->mainTextureRenderTargetView.ptr += textureData->mainSRVIndex * rendererData->rtvDescriptorSize;

        D3D_CALL(rendererData->d3dDevice, CreateRenderTargetView,
                 (ID3D12Resource *)textureData->mainTexture,
                 &renderTargetViewDesc,
                 textureData->mainTextureRenderTargetView);
    }

    return 0;
}

static void D3D12_DestroyTexture(SDL_Renderer *renderer,
                                 SDL_Texture *texture)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->driverdata;

    if (!textureData) {
        return;
    }

    /* Because SDL_DestroyTexture might be called while the data is in-flight, we need to issue the batch first
       Unfortunately, this means that deleting a lot of textures mid-frame will have poor performance. */
    D3D12_IssueBatch(rendererData);

    SAFE_RELEASE(textureData->mainTexture);
    SAFE_RELEASE(textureData->stagingBuffer);
    D3D12_FreeSRVIndex(renderer, textureData->mainSRVIndex);
#if SDL_HAVE_YUV
    SAFE_RELEASE(textureData->mainTextureU);
    SAFE_RELEASE(textureData->mainTextureV);
    if (textureData->yuv) {
        D3D12_FreeSRVIndex(renderer, textureData->mainSRVIndexU);
        D3D12_FreeSRVIndex(renderer, textureData->mainSRVIndexV);
    }
    SAFE_RELEASE(textureData->mainTextureNV);
    if (textureData->yuv) {
        D3D12_FreeSRVIndex(renderer, textureData->mainSRVIndexNV);
    }
    SDL_free(textureData->pixels);
#endif
    SDL_free(textureData);
    texture->driverdata = NULL;
}

static int D3D12_UpdateTextureInternal(D3D12_RenderData *rendererData, ID3D12Resource *texture, int bpp, int x, int y, int w, int h, const void *pixels, int pitch, D3D12_RESOURCE_STATES *resourceState)
{
    const Uint8 *src;
    Uint8 *dst;
    int row;
    UINT length;
    HRESULT result;
    D3D12_RESOURCE_DESC textureDesc;
    D3D12_RESOURCE_DESC uploadDesc;
    D3D12_HEAP_PROPERTIES heapProps;
    D3D12_SUBRESOURCE_FOOTPRINT pitchedDesc;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placedTextureDesc;
    D3D12_TEXTURE_COPY_LOCATION srcLocation;
    D3D12_TEXTURE_COPY_LOCATION dstLocation;
    BYTE *textureMemory;
    ID3D12Resource *uploadBuffer;

    /* Create an upload buffer, which will be used to write to the main texture. */
    SDL_zero(textureDesc);
    D3D_CALL_RET_ID3D12Resource_GetDesc(texture, &textureDesc);
    textureDesc.Width = w;
    textureDesc.Height = h;

    SDL_zero(uploadDesc);
    uploadDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    uploadDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    uploadDesc.Height = 1;
    uploadDesc.DepthOrArraySize = 1;
    uploadDesc.MipLevels = 1;
    uploadDesc.Format = DXGI_FORMAT_UNKNOWN;
    uploadDesc.SampleDesc.Count = 1;
    uploadDesc.SampleDesc.Quality = 0;
    uploadDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    uploadDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    /* Figure out how much we need to allocate for the upload buffer */
    D3D_CALL(rendererData->d3dDevice, GetCopyableFootprints,
             &textureDesc,
             0,
             1,
             0,
             NULL,
             NULL,
             NULL,
             &uploadDesc.Width);

    SDL_zero(heapProps);
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    /* Create the upload buffer */
    result = D3D_CALL(rendererData->d3dDevice, CreateCommittedResource,
                      &heapProps,
                      D3D12_HEAP_FLAG_NONE,
                      &uploadDesc,
                      D3D12_RESOURCE_STATE_GENERIC_READ,
                      NULL,
                      D3D_GUID(SDL_IID_ID3D12Resource),
                      (void **)&rendererData->uploadBuffers[rendererData->currentUploadBuffer]);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommittedResource [create upload buffer]"), result);
    }

    /* Get a write-only pointer to data in the upload buffer: */
    uploadBuffer = rendererData->uploadBuffers[rendererData->currentUploadBuffer];
    result = D3D_CALL(uploadBuffer, Map,
                      0,
                      NULL,
                      (void **)&textureMemory);
    if (FAILED(result)) {
        SAFE_RELEASE(rendererData->uploadBuffers[rendererData->currentUploadBuffer]);
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Resource::Map [map staging texture]"), result);
    }

    SDL_zero(pitchedDesc);
    pitchedDesc.Format = textureDesc.Format;
    pitchedDesc.Width = w;
    pitchedDesc.Height = h;
    pitchedDesc.Depth = 1;
    pitchedDesc.RowPitch = D3D12_Align(w * bpp, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);

    SDL_zero(placedTextureDesc);
    placedTextureDesc.Offset = 0;
    placedTextureDesc.Footprint = pitchedDesc;

    src = (const Uint8 *)pixels;
    dst = textureMemory;
    length = w * bpp;
    if (length == pitch && length == pitchedDesc.RowPitch) {
        SDL_memcpy(dst, src, (size_t)length * h);
    } else {
        if (length > (UINT)pitch) {
            length = pitch;
        }
        if (length > pitchedDesc.RowPitch) {
            length = pitchedDesc.RowPitch;
        }
        for (row = 0; row < h; ++row) {
            SDL_memcpy(dst, src, length);
            src += pitch;
            dst += pitchedDesc.RowPitch;
        }
    }

    /* Commit the changes back to the upload buffer: */
    D3D_CALL(uploadBuffer, Unmap, 0, NULL);

    /* Make sure the destination is in the correct resource state */
    D3D12_TransitionResource(rendererData, texture, *resourceState, D3D12_RESOURCE_STATE_COPY_DEST);
    *resourceState = D3D12_RESOURCE_STATE_COPY_DEST;

    SDL_zero(dstLocation);
    dstLocation.pResource = texture;
    dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLocation.SubresourceIndex = 0;

    SDL_zero(srcLocation);
    srcLocation.pResource = rendererData->uploadBuffers[rendererData->currentUploadBuffer];
    srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLocation.PlacedFootprint = placedTextureDesc;

    D3D_CALL(rendererData->commandList, CopyTextureRegion,
             &dstLocation,
             x,
             y,
             0,
             &srcLocation,
             NULL);

    /* Transition the texture to be shader accessible */
    D3D12_TransitionResource(rendererData, texture, *resourceState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    *resourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    rendererData->currentUploadBuffer++;
    /* If we've used up all the upload buffers, we need to issue the batch */
    if (rendererData->currentUploadBuffer == SDL_D3D12_NUM_UPLOAD_BUFFERS) {
        D3D12_IssueBatch(rendererData);
    }

    return 0;
}

static int D3D12_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                               const SDL_Rect *rect, const void *srcPixels,
                               int srcPitch)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->driverdata;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    if (D3D12_UpdateTextureInternal(rendererData, textureData->mainTexture, SDL_BYTESPERPIXEL(texture->format), rect->x, rect->y, rect->w, rect->h, srcPixels, srcPitch, &textureData->mainResourceState) < 0) {
        return -1;
    }
#if SDL_HAVE_YUV
    if (textureData->yuv) {
        /* Skip to the correct offset into the next texture */
        srcPixels = (const void *)((const Uint8 *)srcPixels + rect->h * srcPitch);

        if (D3D12_UpdateTextureInternal(rendererData, texture->format == SDL_PIXELFORMAT_YV12 ? textureData->mainTextureV : textureData->mainTextureU, SDL_BYTESPERPIXEL(texture->format), rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2, srcPixels, (srcPitch + 1) / 2, texture->format == SDL_PIXELFORMAT_YV12 ? &textureData->mainResourceStateV : &textureData->mainResourceStateU) < 0) {
            return -1;
        }

        /* Skip to the correct offset into the next texture */
        srcPixels = (const void *)((const Uint8 *)srcPixels + ((rect->h + 1) / 2) * ((srcPitch + 1) / 2));
        if (D3D12_UpdateTextureInternal(rendererData, texture->format == SDL_PIXELFORMAT_YV12 ? textureData->mainTextureU : textureData->mainTextureV, SDL_BYTESPERPIXEL(texture->format), rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2, srcPixels, (srcPitch + 1) / 2, texture->format == SDL_PIXELFORMAT_YV12 ? &textureData->mainResourceStateU : &textureData->mainResourceStateV) < 0) {
            return -1;
        }
    }

    if (textureData->nv12) {
        /* Skip to the correct offset into the next texture */
        srcPixels = (const void *)((const Uint8 *)srcPixels + rect->h * srcPitch);

        if (D3D12_UpdateTextureInternal(rendererData, textureData->mainTextureNV, 2, rect->x / 2, rect->y / 2, ((rect->w + 1) / 2), (rect->h + 1) / 2, srcPixels, 2 * ((srcPitch + 1) / 2), &textureData->mainResourceStateNV) < 0) {
            return -1;
        }
    }
#endif /* SDL_HAVE_YUV */
    return 0;
}

#if SDL_HAVE_YUV
static int D3D12_UpdateTextureYUV(SDL_Renderer *renderer, SDL_Texture *texture,
                                  const SDL_Rect *rect,
                                  const Uint8 *Yplane, int Ypitch,
                                  const Uint8 *Uplane, int Upitch,
                                  const Uint8 *Vplane, int Vpitch)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->driverdata;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    if (D3D12_UpdateTextureInternal(rendererData, textureData->mainTexture, SDL_BYTESPERPIXEL(texture->format), rect->x, rect->y, rect->w, rect->h, Yplane, Ypitch, &textureData->mainResourceState) < 0) {
        return -1;
    }
    if (D3D12_UpdateTextureInternal(rendererData, textureData->mainTextureU, SDL_BYTESPERPIXEL(texture->format), rect->x / 2, rect->y / 2, rect->w / 2, rect->h / 2, Uplane, Upitch, &textureData->mainResourceStateU) < 0) {
        return -1;
    }
    if (D3D12_UpdateTextureInternal(rendererData, textureData->mainTextureV, SDL_BYTESPERPIXEL(texture->format), rect->x / 2, rect->y / 2, rect->w / 2, rect->h / 2, Vplane, Vpitch, &textureData->mainResourceStateV) < 0) {
        return -1;
    }
    return 0;
}

static int D3D12_UpdateTextureNV(SDL_Renderer *renderer, SDL_Texture *texture,
                                 const SDL_Rect *rect,
                                 const Uint8 *Yplane, int Ypitch,
                                 const Uint8 *UVplane, int UVpitch)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->driverdata;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    if (D3D12_UpdateTextureInternal(rendererData, textureData->mainTexture, SDL_BYTESPERPIXEL(texture->format), rect->x, rect->y, rect->w, rect->h, Yplane, Ypitch, &textureData->mainResourceState) < 0) {
        return -1;
    }

    if (D3D12_UpdateTextureInternal(rendererData, textureData->mainTextureNV, 2, rect->x / 2, rect->y / 2, ((rect->w + 1) / 2), (rect->h + 1) / 2, UVplane, UVpitch, &textureData->mainResourceStateNV) < 0) {
        return -1;
    }
    return 0;
}
#endif

static int D3D12_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                             const SDL_Rect *rect, void **pixels, int *pitch)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->driverdata;
    HRESULT result = S_OK;

    D3D12_RESOURCE_DESC textureDesc;
    D3D12_RESOURCE_DESC uploadDesc;
    D3D12_HEAP_PROPERTIES heapProps;
    D3D12_SUBRESOURCE_FOOTPRINT pitchedDesc;
    BYTE *textureMemory;
    int bpp;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }
#if SDL_HAVE_YUV
    if (textureData->yuv || textureData->nv12) {
        /* It's more efficient to upload directly... */
        if (!textureData->pixels) {
            textureData->pitch = texture->w;
            textureData->pixels = (Uint8 *)SDL_malloc((texture->h * textureData->pitch * 3) / 2);
            if (!textureData->pixels) {
                return SDL_OutOfMemory();
            }
        }
        textureData->lockedRect = *rect;
        *pixels =
            (void *)(textureData->pixels + rect->y * textureData->pitch +
                     rect->x * SDL_BYTESPERPIXEL(texture->format));
        *pitch = textureData->pitch;
        return 0;
    }
#endif
    if (textureData->stagingBuffer) {
        return SDL_SetError("texture is already locked");
    }

    /* Create an upload buffer, which will be used to write to the main texture. */
    SDL_zero(textureDesc);
    D3D_CALL_RET_ID3D12Resource_GetDesc(textureData->mainTexture, &textureDesc);
    textureDesc.Width = rect->w;
    textureDesc.Height = rect->h;

    SDL_zero(uploadDesc);
    uploadDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    uploadDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    uploadDesc.Height = 1;
    uploadDesc.DepthOrArraySize = 1;
    uploadDesc.MipLevels = 1;
    uploadDesc.Format = DXGI_FORMAT_UNKNOWN;
    uploadDesc.SampleDesc.Count = 1;
    uploadDesc.SampleDesc.Quality = 0;
    uploadDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    uploadDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    /* Figure out how much we need to allocate for the upload buffer */
    D3D_CALL(rendererData->d3dDevice, GetCopyableFootprints,
             &textureDesc,
             0,
             1,
             0,
             NULL,
             NULL,
             NULL,
             &uploadDesc.Width);

    SDL_zero(heapProps);
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    /* Create the upload buffer */
    result = D3D_CALL(rendererData->d3dDevice, CreateCommittedResource,
                      &heapProps,
                      D3D12_HEAP_FLAG_NONE,
                      &uploadDesc,
                      D3D12_RESOURCE_STATE_GENERIC_READ,
                      NULL,
                      D3D_GUID(SDL_IID_ID3D12Resource),
                      (void **)&textureData->stagingBuffer);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommittedResource [create upload buffer]"), result);
    }

    /* Get a write-only pointer to data in the upload buffer: */
    result = D3D_CALL(textureData->stagingBuffer, Map,
                      0,
                      NULL,
                      (void **)&textureMemory);
    if (FAILED(result)) {
        SAFE_RELEASE(rendererData->uploadBuffers[rendererData->currentUploadBuffer]);
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Resource::Map [map staging texture]"), result);
    }

    SDL_zero(pitchedDesc);
    pitchedDesc.Format = textureDesc.Format;
    pitchedDesc.Width = rect->w;
    pitchedDesc.Height = rect->h;
    pitchedDesc.Depth = 1;
    if (pitchedDesc.Format == DXGI_FORMAT_R8_UNORM) {
        bpp = 1;
    } else {
        bpp = 4;
    }
    pitchedDesc.RowPitch = D3D12_Align(rect->w * bpp, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);

    /* Make note of where the staging texture will be written to
     * (on a call to SDL_UnlockTexture):
     */
    textureData->lockedRect = *rect;

    /* Make sure the caller has information on the texture's pixel buffer,
     * then return:
     */
    *pixels = textureMemory;
    *pitch = pitchedDesc.RowPitch;
    return 0;
}

static void D3D12_UnlockTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->driverdata;

    D3D12_RESOURCE_DESC textureDesc;
    D3D12_SUBRESOURCE_FOOTPRINT pitchedDesc;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placedTextureDesc;
    D3D12_TEXTURE_COPY_LOCATION srcLocation;
    D3D12_TEXTURE_COPY_LOCATION dstLocation;
    int bpp;

    if (!textureData) {
        return;
    }
#if SDL_HAVE_YUV
    if (textureData->yuv || textureData->nv12) {
        const SDL_Rect *rect = &textureData->lockedRect;
        void *pixels =
            (void *)(textureData->pixels + rect->y * textureData->pitch +
                     rect->x * SDL_BYTESPERPIXEL(texture->format));
        D3D12_UpdateTexture(renderer, texture, rect, pixels, textureData->pitch);
        return;
    }
#endif
    /* Commit the pixel buffer's changes back to the staging texture: */
    D3D_CALL(textureData->stagingBuffer, Unmap, 0, NULL);

    SDL_zero(textureDesc);
    D3D_CALL_RET_ID3D12Resource_GetDesc(textureData->mainTexture, &textureDesc);
    textureDesc.Width = textureData->lockedRect.w;
    textureDesc.Height = textureData->lockedRect.h;

    SDL_zero(pitchedDesc);
    pitchedDesc.Format = textureDesc.Format;
    pitchedDesc.Width = (UINT)textureDesc.Width;
    pitchedDesc.Height = textureDesc.Height;
    pitchedDesc.Depth = 1;
    if (pitchedDesc.Format == DXGI_FORMAT_R8_UNORM) {
        bpp = 1;
    } else {
        bpp = 4;
    }
    pitchedDesc.RowPitch = D3D12_Align(textureData->lockedRect.w * bpp, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);

    SDL_zero(placedTextureDesc);
    placedTextureDesc.Offset = 0;
    placedTextureDesc.Footprint = pitchedDesc;

    D3D12_TransitionResource(rendererData, textureData->mainTexture, textureData->mainResourceState, D3D12_RESOURCE_STATE_COPY_DEST);
    textureData->mainResourceState = D3D12_RESOURCE_STATE_COPY_DEST;

    SDL_zero(dstLocation);
    dstLocation.pResource = textureData->mainTexture;
    dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLocation.SubresourceIndex = 0;

    SDL_zero(srcLocation);
    srcLocation.pResource = textureData->stagingBuffer;
    srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLocation.PlacedFootprint = placedTextureDesc;

    D3D_CALL(rendererData->commandList, CopyTextureRegion,
             &dstLocation,
             textureData->lockedRect.x,
             textureData->lockedRect.y,
             0,
             &srcLocation,
             NULL);

    /* Transition the texture to be shader accessible */
    D3D12_TransitionResource(rendererData, textureData->mainTexture, textureData->mainResourceState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    textureData->mainResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    /* Execute the command list before releasing the staging buffer */
    D3D12_IssueBatch(rendererData);
    SAFE_RELEASE(textureData->stagingBuffer);
}

static void D3D12_SetTextureScaleMode(SDL_Renderer *renderer, SDL_Texture *texture, SDL_ScaleMode scaleMode)
{
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->driverdata;

    if (!textureData) {
        return;
    }

    textureData->scaleMode = (scaleMode == SDL_ScaleModeNearest) ? D3D12_FILTER_MIN_MAG_MIP_POINT : D3D12_FILTER_MIN_MAG_MIP_LINEAR;
}

static int D3D12_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    D3D12_TextureData *textureData = NULL;

    if (!texture) {
        if (rendererData->textureRenderTarget) {
            D3D12_TransitionResource(rendererData,
                                     rendererData->textureRenderTarget->mainTexture,
                                     rendererData->textureRenderTarget->mainResourceState,
                                     D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            rendererData->textureRenderTarget->mainResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        }
        rendererData->textureRenderTarget = NULL;
        return 0;
    }

    textureData = (D3D12_TextureData *)texture->driverdata;

    if (!textureData->mainTextureRenderTargetView.ptr) {
        return SDL_SetError("specified texture is not a render target");
    }

    rendererData->textureRenderTarget = textureData;
    D3D12_TransitionResource(rendererData,
                             rendererData->textureRenderTarget->mainTexture,
                             rendererData->textureRenderTarget->mainResourceState,
                             D3D12_RESOURCE_STATE_RENDER_TARGET);
    rendererData->textureRenderTarget->mainResourceState = D3D12_RESOURCE_STATE_RENDER_TARGET;

    return 0;
}

static int D3D12_QueueSetViewport(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    return 0; /* nothing to do in this backend. */
}

static int D3D12_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    VertexPositionColor *verts = (VertexPositionColor *)SDL_AllocateRenderVertices(renderer, count * sizeof(VertexPositionColor), 0, &cmd->data.draw.first);
    int i;
    SDL_Color color;
    color.r = cmd->data.draw.r;
    color.g = cmd->data.draw.g;
    color.b = cmd->data.draw.b;
    color.a = cmd->data.draw.a;

    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = count;

    for (i = 0; i < count; i++) {
        verts->pos.x = points[i].x + 0.5f;
        verts->pos.y = points[i].y + 0.5f;
        verts->tex.x = 0.0f;
        verts->tex.y = 0.0f;
        verts->color = color;
        verts++;
    }

    return 0;
}

static int D3D12_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                               const float *xy, int xy_stride, const SDL_Color *color, int color_stride, const float *uv, int uv_stride,
                               int num_vertices, const void *indices, int num_indices, int size_indices,
                               float scale_x, float scale_y)
{
    int i;
    int count = indices ? num_indices : num_vertices;
    VertexPositionColor *verts = (VertexPositionColor *)SDL_AllocateRenderVertices(renderer, count * sizeof(VertexPositionColor), 0, &cmd->data.draw.first);

    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = count;
    size_indices = indices ? size_indices : 0;

    for (i = 0; i < count; i++) {
        int j;
        float *xy_;
        if (size_indices == 4) {
            j = ((const Uint32 *)indices)[i];
        } else if (size_indices == 2) {
            j = ((const Uint16 *)indices)[i];
        } else if (size_indices == 1) {
            j = ((const Uint8 *)indices)[i];
        } else {
            j = i;
        }

        xy_ = (float *)((char *)xy + j * xy_stride);

        verts->pos.x = xy_[0] * scale_x;
        verts->pos.y = xy_[1] * scale_y;
        verts->color = *(SDL_Color *)((char *)color + j * color_stride);

        if (texture) {
            float *uv_ = (float *)((char *)uv + j * uv_stride);
            verts->tex.x = uv_[0];
            verts->tex.y = uv_[1];
        } else {
            verts->tex.x = 0.0f;
            verts->tex.y = 0.0f;
        }

        verts += 1;
    }
    return 0;
}

static int D3D12_UpdateVertexBuffer(SDL_Renderer *renderer,
                                    const void *vertexData, size_t dataSizeInBytes)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    HRESULT result = S_OK;
    const int vbidx = rendererData->currentVertexBuffer;
    UINT8 *vertexBufferData = NULL;
    D3D12_RANGE range;
    ID3D12Resource *vertexBuffer;

    range.Begin = 0;
    range.End = 0;

    if (dataSizeInBytes == 0) {
        return 0; /* nothing to do. */
    }

    if (rendererData->issueBatch) {
        if (FAILED(D3D12_IssueBatch(rendererData))) {
            SDL_SetError("Failed to issue intermediate batch");
            return E_FAIL;
        }
    }

    /* If the existing vertex buffer isn't big enough, we need to recreate a big enough one */
    if (dataSizeInBytes > rendererData->vertexBuffers[vbidx].size) {
        D3D12_CreateVertexBuffer(rendererData, vbidx, dataSizeInBytes);
    }

    vertexBuffer = rendererData->vertexBuffers[vbidx].resource;
    result = D3D_CALL(vertexBuffer, Map, 0, &range, (void **)&vertexBufferData);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Resource::Map [vertex buffer]"), result);
    }
    SDL_memcpy(vertexBufferData, vertexData, dataSizeInBytes);
    D3D_CALL(vertexBuffer, Unmap, 0, NULL);

    rendererData->vertexBuffers[vbidx].view.SizeInBytes = (UINT)dataSizeInBytes;

    D3D_CALL(rendererData->commandList, IASetVertexBuffers, 0, 1, &rendererData->vertexBuffers[vbidx].view);

    rendererData->currentVertexBuffer++;
    if (rendererData->currentVertexBuffer >= SDL_D3D12_NUM_VERTEX_BUFFERS) {
        rendererData->currentVertexBuffer = 0;
        rendererData->issueBatch = SDL_TRUE;
    }

    return S_OK;
}

static int D3D12_UpdateViewport(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;
    const SDL_Rect *viewport = &data->currentViewport;
    Float4X4 projection;
    Float4X4 view;
    SDL_FRect orientationAlignedViewport;
    BOOL swapDimensions;
    D3D12_VIEWPORT d3dviewport;
    const int rotation = D3D12_GetRotationForCurrentRenderTarget(renderer);

    if (viewport->w == 0 || viewport->h == 0) {
        /* If the viewport is empty, assume that it is because
         * SDL_CreateRenderer is calling it, and will call it again later
         * with a non-empty viewport.
         */
        /* SDL_Log("%s, no viewport was set!\n", __FUNCTION__); */
        return -1;
    }

    /* Make sure the SDL viewport gets rotated to that of the physical display's rotation.
     * Keep in mind here that the Y-axis will be been inverted (from Direct3D's
     * default coordinate system) so rotations will be done in the opposite
     * direction of the DXGI_MODE_ROTATION enumeration.
     */
    switch (rotation) {
        case DXGI_MODE_ROTATION_IDENTITY:
            projection = MatrixIdentity();
            break;
        case DXGI_MODE_ROTATION_ROTATE270:
            projection = MatrixRotationZ(SDL_static_cast(float, M_PI * 0.5f));
            break;
        case DXGI_MODE_ROTATION_ROTATE180:
            projection = MatrixRotationZ(SDL_static_cast(float, M_PI));
            break;
        case DXGI_MODE_ROTATION_ROTATE90:
            projection = MatrixRotationZ(SDL_static_cast(float, -M_PI * 0.5f));
            break;
        default:
            return SDL_SetError("An unknown DisplayOrientation is being used");
    }

    /* Update the view matrix */
    SDL_zero(view);
    view.m[0][0] = 2.0f / viewport->w;
    view.m[1][1] = -2.0f / viewport->h;
    view.m[2][2] = 1.0f;
    view.m[3][0] = -1.0f;
    view.m[3][1] = 1.0f;
    view.m[3][3] = 1.0f;

    /* Combine the projection + view matrix together now, as both only get
     * set here (as of this writing, on Dec 26, 2013).  When done, store it
     * for eventual transfer to the GPU.
     */
    data->vertexShaderConstantsData.projectionAndView = MatrixMultiply(
        view,
        projection);

    /* Update the Direct3D viewport, which seems to be aligned to the
     * swap buffer's coordinate space, which is always in either
     * a landscape mode, for all Windows 8/RT devices, or a portrait mode,
     * for Windows Phone devices.
     */
    swapDimensions = D3D12_IsDisplayRotated90Degrees((DXGI_MODE_ROTATION)rotation);
    if (swapDimensions) {
        orientationAlignedViewport.x = (float)viewport->y;
        orientationAlignedViewport.y = (float)viewport->x;
        orientationAlignedViewport.w = (float)viewport->h;
        orientationAlignedViewport.h = (float)viewport->w;
    } else {
        orientationAlignedViewport.x = (float)viewport->x;
        orientationAlignedViewport.y = (float)viewport->y;
        orientationAlignedViewport.w = (float)viewport->w;
        orientationAlignedViewport.h = (float)viewport->h;
    }

    d3dviewport.TopLeftX = orientationAlignedViewport.x;
    d3dviewport.TopLeftY = orientationAlignedViewport.y;
    d3dviewport.Width = orientationAlignedViewport.w;
    d3dviewport.Height = orientationAlignedViewport.h;
    d3dviewport.MinDepth = 0.0f;
    d3dviewport.MaxDepth = 1.0f;
    /* SDL_Log("%s: D3D viewport = {%f,%f,%f,%f}\n", __FUNCTION__, d3dviewport.TopLeftX, d3dviewport.TopLeftY, d3dviewport.Width, d3dviewport.Height); */
    D3D_CALL(data->commandList, RSSetViewports, 1, &d3dviewport);

    data->viewportDirty = SDL_FALSE;

    return 0;
}

static int D3D12_SetDrawState(SDL_Renderer *renderer, const SDL_RenderCommand *cmd, D3D12_Shader shader,
                              D3D12_PRIMITIVE_TOPOLOGY_TYPE topology,
                              const int numShaderResources, D3D12_CPU_DESCRIPTOR_HANDLE *shaderResources,
                              D3D12_CPU_DESCRIPTOR_HANDLE *sampler, const Float4X4 *matrix)

{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    const Float4X4 *newmatrix = matrix ? matrix : &rendererData->identity;
    D3D12_CPU_DESCRIPTOR_HANDLE renderTargetView = D3D12_GetCurrentRenderTargetView(renderer);
    const SDL_BlendMode blendMode = cmd->data.draw.blend;
    SDL_bool updateSubresource = SDL_FALSE;
    int i;
    D3D12_CPU_DESCRIPTOR_HANDLE firstShaderResource;
    DXGI_FORMAT rtvFormat = DXGI_FORMAT_B8G8R8A8_UNORM;

    if (rendererData->textureRenderTarget) {
        rtvFormat = rendererData->textureRenderTarget->mainTextureFormat;
    }

    /* See if we need to change the pipeline state */
    if (!rendererData->currentPipelineState ||
        rendererData->currentPipelineState->shader != shader ||
        rendererData->currentPipelineState->blendMode != blendMode ||
        rendererData->currentPipelineState->topology != topology ||
        rendererData->currentPipelineState->rtvFormat != rtvFormat) {

        /* Find the matching pipeline.
           NOTE: Although it may seem inefficient to linearly search through ~450 pipelines
           to find the correct one, in profiling this doesn't come up at all.
           It's unlikely that using a hash table would affect performance a measurable amount unless
           it's a degenerate case that's chaning the pipline state dozens of times per frame.
        */
        rendererData->currentPipelineState = NULL;
        for (i = 0; i < rendererData->pipelineStateCount; ++i) {
            D3D12_PipelineState *candidatePiplineState = &rendererData->pipelineStates[i];
            if (candidatePiplineState->shader == shader &&
                candidatePiplineState->blendMode == blendMode &&
                candidatePiplineState->topology == topology &&
                candidatePiplineState->rtvFormat == rtvFormat) {
                rendererData->currentPipelineState = candidatePiplineState;
                break;
            }
        }

        /* If we didn't find a match, create a new one -- it must mean the blend mode is non-standard */
        if (!rendererData->currentPipelineState) {
            rendererData->currentPipelineState = D3D12_CreatePipelineState(renderer, shader, blendMode, topology, rtvFormat);
        }

        if (!rendererData->currentPipelineState) {
            return SDL_SetError("[direct3d12] Unable to create required pipeline state");
        }

        D3D_CALL(rendererData->commandList, SetPipelineState, rendererData->currentPipelineState->pipelineState);
        D3D_CALL(rendererData->commandList, SetGraphicsRootSignature,
                 rendererData->rootSignatures[D3D12_GetRootSignatureType(rendererData->currentPipelineState->shader)]);
        /* When we change these we will need to re-upload the constant buffer and reset any descriptors */
        updateSubresource = SDL_TRUE;
        rendererData->currentSampler.ptr = 0;
        rendererData->currentShaderResource.ptr = 0;
    }

    if (renderTargetView.ptr != rendererData->currentRenderTargetView.ptr) {
        D3D_CALL(rendererData->commandList, OMSetRenderTargets, 1, &renderTargetView, FALSE, NULL);
        rendererData->currentRenderTargetView = renderTargetView;
    }

    if (rendererData->viewportDirty) {
        if (D3D12_UpdateViewport(renderer) == 0) {
            /* vertexShaderConstantsData.projectionAndView has changed */
            updateSubresource = SDL_TRUE;
        }
    }

    if (rendererData->cliprectDirty) {
        D3D12_RECT scissorRect;
        if (D3D12_GetViewportAlignedD3DRect(renderer, &rendererData->currentCliprect, &scissorRect, TRUE) != 0) {
            /* D3D12_GetViewportAlignedD3DRect will have set the SDL error */
            return -1;
        }
        D3D_CALL(rendererData->commandList, RSSetScissorRects, 1, &scissorRect);
        rendererData->cliprectDirty = SDL_FALSE;
    }

    if (numShaderResources > 0) {
        firstShaderResource = shaderResources[0];
    } else {
        firstShaderResource.ptr = 0;
    }
    if (firstShaderResource.ptr != rendererData->currentShaderResource.ptr) {
        for (i = 0; i < numShaderResources; ++i) {
            D3D12_GPU_DESCRIPTOR_HANDLE GPUHandle = D3D12_CPUtoGPUHandle(rendererData->srvDescriptorHeap, shaderResources[i]);
            D3D_CALL(rendererData->commandList, SetGraphicsRootDescriptorTable, i + 1, GPUHandle);
        }
        rendererData->currentShaderResource.ptr = firstShaderResource.ptr;
    }

    if (sampler && sampler->ptr != rendererData->currentSampler.ptr) {
        D3D12_GPU_DESCRIPTOR_HANDLE GPUHandle = D3D12_CPUtoGPUHandle(rendererData->samplerDescriptorHeap, *sampler);
        UINT tableIndex = 0;

        /* Figure out the correct sampler descriptor table index based on the type of shader */
        switch (shader) {
        case SHADER_RGB:
            tableIndex = 2;
            break;
#if SDL_HAVE_YUV
        case SHADER_YUV_JPEG:
        case SHADER_YUV_BT601:
        case SHADER_YUV_BT709:
            tableIndex = 4;
            break;
        case SHADER_NV12_JPEG:
        case SHADER_NV12_BT601:
        case SHADER_NV12_BT709:
        case SHADER_NV21_JPEG:
        case SHADER_NV21_BT601:
        case SHADER_NV21_BT709:
            tableIndex = 3;
            break;
#endif
        default:
            return SDL_SetError("[direct3d12] Trying to set a sampler for a shader which doesn't have one");
            break;
        }

        D3D_CALL(rendererData->commandList, SetGraphicsRootDescriptorTable, tableIndex, GPUHandle);
        rendererData->currentSampler = *sampler;
    }

    if (updateSubresource == SDL_TRUE || SDL_memcmp(&rendererData->vertexShaderConstantsData.model, newmatrix, sizeof(*newmatrix)) != 0) {
        SDL_memcpy(&rendererData->vertexShaderConstantsData.model, newmatrix, sizeof(*newmatrix));
        D3D_CALL(rendererData->commandList, SetGraphicsRoot32BitConstants,
                 0,
                 32,
                 &rendererData->vertexShaderConstantsData,
                 0);
    }

    return 0;
}

static int D3D12_SetCopyState(SDL_Renderer *renderer, const SDL_RenderCommand *cmd, const Float4X4 *matrix)
{
    SDL_Texture *texture = cmd->data.draw.texture;
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->driverdata;
    D3D12_CPU_DESCRIPTOR_HANDLE *textureSampler;

    switch (textureData->scaleMode) {
    case D3D12_FILTER_MIN_MAG_MIP_POINT:
        textureSampler = &rendererData->nearestPixelSampler;
        break;
    case D3D12_FILTER_MIN_MAG_MIP_LINEAR:
        textureSampler = &rendererData->linearSampler;
        break;
    default:
        return SDL_SetError("Unknown scale mode: %d\n", textureData->scaleMode);
    }
#if SDL_HAVE_YUV
    if (textureData->yuv) {
        D3D12_CPU_DESCRIPTOR_HANDLE shaderResources[] = {
            textureData->mainTextureResourceView,
            textureData->mainTextureResourceViewU,
            textureData->mainTextureResourceViewV
        };
        D3D12_Shader shader;

        switch (SDL_GetYUVConversionModeForResolution(texture->w, texture->h)) {
        case SDL_YUV_CONVERSION_JPEG:
            shader = SHADER_YUV_JPEG;
            break;
        case SDL_YUV_CONVERSION_BT601:
            shader = SHADER_YUV_BT601;
            break;
        case SDL_YUV_CONVERSION_BT709:
            shader = SHADER_YUV_BT709;
            break;
        default:
            return SDL_SetError("Unsupported YUV conversion mode");
        }

        /* Make sure each texture is in the correct state to be accessed by the pixel shader. */
        D3D12_TransitionResource(rendererData, textureData->mainTexture, textureData->mainResourceState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        textureData->mainResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_TransitionResource(rendererData, textureData->mainTextureU, textureData->mainResourceStateU, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        textureData->mainResourceStateU = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_TransitionResource(rendererData, textureData->mainTextureV, textureData->mainResourceStateV, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        textureData->mainResourceStateV = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

        return D3D12_SetDrawState(renderer, cmd, shader, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, SDL_arraysize(shaderResources), shaderResources,
                                  textureSampler, matrix);
    } else if (textureData->nv12) {
        D3D12_CPU_DESCRIPTOR_HANDLE shaderResources[] = {
            textureData->mainTextureResourceView,
            textureData->mainTextureResourceViewNV,
        };
        D3D12_Shader shader;

        switch (SDL_GetYUVConversionModeForResolution(texture->w, texture->h)) {
        case SDL_YUV_CONVERSION_JPEG:
            shader = texture->format == SDL_PIXELFORMAT_NV12 ? SHADER_NV12_JPEG : SHADER_NV21_JPEG;
            break;
        case SDL_YUV_CONVERSION_BT601:
            shader = texture->format == SDL_PIXELFORMAT_NV12 ? SHADER_NV12_BT601 : SHADER_NV21_BT601;
            break;
        case SDL_YUV_CONVERSION_BT709:
            shader = texture->format == SDL_PIXELFORMAT_NV12 ? SHADER_NV12_BT709 : SHADER_NV21_BT709;
            break;
        default:
            return SDL_SetError("Unsupported YUV conversion mode");
        }

        /* Make sure each texture is in the correct state to be accessed by the pixel shader. */
        D3D12_TransitionResource(rendererData, textureData->mainTexture, textureData->mainResourceState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        textureData->mainResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_TransitionResource(rendererData, textureData->mainTextureNV, textureData->mainResourceStateNV, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        textureData->mainResourceStateNV = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

        return D3D12_SetDrawState(renderer, cmd, shader, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, SDL_arraysize(shaderResources), shaderResources,
                                  textureSampler, matrix);
    }
#endif /* SDL_HAVE_YUV */
    D3D12_TransitionResource(rendererData, textureData->mainTexture, textureData->mainResourceState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    textureData->mainResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    return D3D12_SetDrawState(renderer, cmd, SHADER_RGB, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, 1, &textureData->mainTextureResourceView,
                              textureSampler, matrix);
}

static void D3D12_DrawPrimitives(SDL_Renderer *renderer, D3D12_PRIMITIVE_TOPOLOGY primitiveTopology, const size_t vertexStart, const size_t vertexCount)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    D3D_CALL(rendererData->commandList, IASetPrimitiveTopology, primitiveTopology);
    D3D_CALL(rendererData->commandList, DrawInstanced, (UINT)vertexCount, 1, (UINT)vertexStart, 0);
}

static int D3D12_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->driverdata;
    const int viewportRotation = D3D12_GetRotationForCurrentRenderTarget(renderer);

    if (rendererData->currentViewportRotation != viewportRotation) {
        rendererData->currentViewportRotation = viewportRotation;
        rendererData->viewportDirty = SDL_TRUE;
    }

    if (D3D12_UpdateVertexBuffer(renderer, vertices, vertsize) < 0) {
        return -1;
    }

    while (cmd) {
        switch (cmd->command) {
        case SDL_RENDERCMD_SETDRAWCOLOR:
        {
            break; /* this isn't currently used in this render backend. */
        }

        case SDL_RENDERCMD_SETVIEWPORT:
        {
            SDL_Rect *viewport = &rendererData->currentViewport;
            if (SDL_memcmp(viewport, &cmd->data.viewport.rect, sizeof(cmd->data.viewport.rect)) != 0) {
                SDL_copyp(viewport, &cmd->data.viewport.rect);
                rendererData->viewportDirty = SDL_TRUE;
                rendererData->cliprectDirty = SDL_TRUE;
            }
            break;
        }

        case SDL_RENDERCMD_SETCLIPRECT:
        {
            const SDL_Rect *rect = &cmd->data.cliprect.rect;
            SDL_Rect viewport_cliprect;
            if (rendererData->currentCliprectEnabled != cmd->data.cliprect.enabled) {
                rendererData->currentCliprectEnabled = cmd->data.cliprect.enabled;
                rendererData->cliprectDirty = SDL_TRUE;
            }
            if (!rendererData->currentCliprectEnabled) {
                /* If the clip rect is disabled, then the scissor rect should be the whole viewport,
                   since direct3d12 doesn't allow disabling the scissor rectangle */
                viewport_cliprect.x = 0;
                viewport_cliprect.y = 0;
                viewport_cliprect.w = rendererData->currentViewport.w;
                viewport_cliprect.h = rendererData->currentViewport.h;
                rect = &viewport_cliprect;
            }
            if (SDL_memcmp(&rendererData->currentCliprect, rect, sizeof(*rect)) != 0) {
                SDL_copyp(&rendererData->currentCliprect, rect);
                rendererData->cliprectDirty = SDL_TRUE;
            }
            break;
        }

        case SDL_RENDERCMD_CLEAR:
        {
            const float colorRGBA[] = {
                (cmd->data.color.r / 255.0f),
                (cmd->data.color.g / 255.0f),
                (cmd->data.color.b / 255.0f),
                (cmd->data.color.a / 255.0f)
            };

            D3D12_CPU_DESCRIPTOR_HANDLE rtvDescriptor = D3D12_GetCurrentRenderTargetView(renderer);
            D3D_CALL(rendererData->commandList, ClearRenderTargetView, rtvDescriptor, colorRGBA, 0, NULL);
            break;
        }

        case SDL_RENDERCMD_DRAW_POINTS:
        {
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            const size_t start = first / sizeof(VertexPositionColor);
            D3D12_SetDrawState(renderer, cmd, SHADER_SOLID, D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT, 0, NULL, NULL, NULL);
            D3D12_DrawPrimitives(renderer, D3D_PRIMITIVE_TOPOLOGY_POINTLIST, start, count);
            break;
        }

        case SDL_RENDERCMD_DRAW_LINES:
        {
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            const size_t start = first / sizeof(VertexPositionColor);
            const VertexPositionColor *verts = (VertexPositionColor *)(((Uint8 *)vertices) + first);
            D3D12_SetDrawState(renderer, cmd, SHADER_SOLID, D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE, 0, NULL, NULL, NULL);
            D3D12_DrawPrimitives(renderer, D3D_PRIMITIVE_TOPOLOGY_LINESTRIP, start, count);
            if (verts[0].pos.x != verts[count - 1].pos.x || verts[0].pos.y != verts[count - 1].pos.y) {
                D3D12_DrawPrimitives(renderer, D3D_PRIMITIVE_TOPOLOGY_POINTLIST, start + (count - 1), 1);
            }
            break;
        }

        case SDL_RENDERCMD_FILL_RECTS: /* unused */
            break;

        case SDL_RENDERCMD_COPY: /* unused */
            break;

        case SDL_RENDERCMD_COPY_EX: /* unused */
            break;

        case SDL_RENDERCMD_GEOMETRY:
        {
            SDL_Texture *texture = cmd->data.draw.texture;
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            const size_t start = first / sizeof(VertexPositionColor);

            if (texture) {
                D3D12_SetCopyState(renderer, cmd, NULL);
            } else {
                D3D12_SetDrawState(renderer, cmd, SHADER_SOLID, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, 0, NULL, NULL, NULL);
            }

            D3D12_DrawPrimitives(renderer, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, start, count);
            break;
        }

        case SDL_RENDERCMD_NO_OP:
            break;
        }

        cmd = cmd->next;
    }

    return 0;
}

static int D3D12_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect,
                                  Uint32 format, void *pixels, int pitch)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;
    ID3D12Resource *backBuffer = NULL;
    ID3D12Resource *readbackBuffer = NULL;
    HRESULT result;
    int status = -1;
    D3D12_RESOURCE_DESC textureDesc;
    D3D12_RESOURCE_DESC readbackDesc;
    D3D12_HEAP_PROPERTIES heapProps;
    D3D12_RECT srcRect = { 0, 0, 0, 0 };
    D3D12_BOX srcBox;
    D3D12_TEXTURE_COPY_LOCATION dstLocation;
    D3D12_TEXTURE_COPY_LOCATION srcLocation;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placedTextureDesc;
    D3D12_SUBRESOURCE_FOOTPRINT pitchedDesc;
    BYTE *textureMemory;
    int bpp;

    if (data->textureRenderTarget) {
        backBuffer = data->textureRenderTarget->mainTexture;
    } else {
        backBuffer = data->renderTargets[data->currentBackBufferIndex];
    }

    /* Create a staging texture to copy the screen's data to: */
    SDL_zero(textureDesc);
    D3D_CALL_RET_ID3D12Resource_GetDesc(backBuffer, &textureDesc);
    textureDesc.Width = rect->w;
    textureDesc.Height = rect->h;

    SDL_zero(readbackDesc);
    readbackDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    readbackDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    readbackDesc.Height = 1;
    readbackDesc.DepthOrArraySize = 1;
    readbackDesc.MipLevels = 1;
    readbackDesc.Format = DXGI_FORMAT_UNKNOWN;
    readbackDesc.SampleDesc.Count = 1;
    readbackDesc.SampleDesc.Quality = 0;
    readbackDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    readbackDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    /* Figure out how much we need to allocate for the upload buffer */
    D3D_CALL(data->d3dDevice, GetCopyableFootprints,
             &textureDesc,
             0,
             1,
             0,
             NULL,
             NULL,
             NULL,
             &readbackDesc.Width);

    SDL_zero(heapProps);
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    result = D3D_CALL(data->d3dDevice, CreateCommittedResource,
                      &heapProps,
                      D3D12_HEAP_FLAG_NONE,
                      &readbackDesc,
                      D3D12_RESOURCE_STATE_COPY_DEST,
                      NULL,
                      D3D_GUID(SDL_IID_ID3D12Resource),
                      (void **)&readbackBuffer);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateTexture2D [create staging texture]"), result);
        goto done;
    }

    /* Transition the render target to be copyable from */
    D3D12_TransitionResource(data, backBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);

    /* Copy the desired portion of the back buffer to the staging texture: */
    if (D3D12_GetViewportAlignedD3DRect(renderer, rect, &srcRect, FALSE) != 0) {
        /* D3D12_GetViewportAlignedD3DRect will have set the SDL error */
        goto done;
    }
    srcBox.left = srcRect.left;
    srcBox.right = srcRect.right;
    srcBox.top = srcRect.top;
    srcBox.bottom = srcRect.bottom;
    srcBox.front = 0;
    srcBox.back = 1;

    /* Issue the copy texture region */
    SDL_zero(pitchedDesc);
    pitchedDesc.Format = textureDesc.Format;
    pitchedDesc.Width = (UINT)textureDesc.Width;
    pitchedDesc.Height = textureDesc.Height;
    pitchedDesc.Depth = 1;
    if (pitchedDesc.Format == DXGI_FORMAT_R8_UNORM) {
        bpp = 1;
    } else {
        bpp = 4;
    }
    pitchedDesc.RowPitch = D3D12_Align(pitchedDesc.Width * bpp, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);

    SDL_zero(placedTextureDesc);
    placedTextureDesc.Offset = 0;
    placedTextureDesc.Footprint = pitchedDesc;

    SDL_zero(dstLocation);
    dstLocation.pResource = readbackBuffer;
    dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLocation.PlacedFootprint = placedTextureDesc;

    SDL_zero(srcLocation);
    srcLocation.pResource = backBuffer;
    srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLocation.SubresourceIndex = 0;

    D3D_CALL(data->commandList, CopyTextureRegion,
             &dstLocation,
             0, 0, 0,
             &srcLocation,
             &srcBox);

    /* We need to issue the command list for the copy to finish */
    D3D12_IssueBatch(data);

    /* Transition the render target back to a render target */
    D3D12_TransitionResource(data, backBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);

    /* Map the staging texture's data to CPU-accessible memory: */
    result = D3D_CALL(readbackBuffer, Map,
                      0,
                      NULL,
                      (void **)&textureMemory);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Resource::Map [map staging texture]"), result);
        goto done;
    }

    /* Copy the data into the desired buffer, converting pixels to the
     * desired format at the same time:
     */
    status = SDL_ConvertPixels(
        rect->w, rect->h,
        D3D12_DXGIFormatToSDLPixelFormat(textureDesc.Format),
        textureMemory,
        pitchedDesc.RowPitch,
        format,
        pixels,
        pitch);

    /* Unmap the texture: */
    D3D_CALL(readbackBuffer, Unmap, 0, NULL);

done:
    SAFE_RELEASE(readbackBuffer);
    return status;
}

static int D3D12_RenderPresent(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;
#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    UINT syncInterval;
    UINT presentFlags;
#endif
    HRESULT result;

    /* Transition the render target to present state */
    D3D12_TransitionResource(data,
                             data->renderTargets[data->currentBackBufferIndex],
                             D3D12_RESOURCE_STATE_RENDER_TARGET,
                             D3D12_RESOURCE_STATE_PRESENT);

    /* Issue the command list */
    result = D3D_CALL(data->commandList, Close);
    D3D_CALL(data->commandQueue, ExecuteCommandLists, 1, (ID3D12CommandList *const *)&data->commandList);

#if defined(__XBOXONE__) || defined(__XBOXSERIES__)
    result = D3D12_XBOX_PresentFrame(data->commandQueue, data->frameToken, data->renderTargets[data->currentBackBufferIndex]);
#else
    if (renderer->info.flags & SDL_RENDERER_PRESENTVSYNC) {
        syncInterval = 1;
        presentFlags = 0;
    } else {
        syncInterval = 0;
        presentFlags = DXGI_PRESENT_ALLOW_TEARING;
    }

    /* The application may optionally specify "dirty" or "scroll"
     * rects to improve efficiency in certain scenarios.
     */
    result = D3D_CALL(data->swapChain, Present, syncInterval, presentFlags);
#endif

    if (FAILED(result) && result != DXGI_ERROR_WAS_STILL_DRAWING) {
        /* If the device was removed either by a disconnect or a driver upgrade, we
         * must recreate all device resources.
         */
        if (result == DXGI_ERROR_DEVICE_REMOVED) {
            D3D12_HandleDeviceLost(renderer);
        } else if (result == DXGI_ERROR_INVALID_CALL) {
            /* We probably went through a fullscreen <-> windowed transition */
            D3D12_CreateWindowSizeDependentResources(renderer);
        } else {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain::Present"), result);
        }
        return -1;
    } else {
        /* Wait for the GPU and move to the next frame */
        result = D3D_CALL(data->commandQueue, Signal, data->fence, data->fenceValue);

        if (D3D_CALL(data->fence, GetCompletedValue) < data->fenceValue) {
            result = D3D_CALL(data->fence, SetEventOnCompletion,
                              data->fenceValue,
                              data->fenceEvent);
            WaitForSingleObjectEx(data->fenceEvent, INFINITE, FALSE);
        }

        data->fenceValue++;
#if defined(__XBOXONE__) || defined(__XBOXSERIES__)
        data->currentBackBufferIndex++;
        data->currentBackBufferIndex %= SDL_D3D12_NUM_BUFFERS;
#else
        data->currentBackBufferIndex = D3D_CALL(data->swapChain, GetCurrentBackBufferIndex);
#endif

        /* Reset the command allocator and command list, and transition back to render target */
        D3D12_ResetCommandList(data);
        D3D12_TransitionResource(data,
                                 data->renderTargets[data->currentBackBufferIndex],
                                 D3D12_RESOURCE_STATE_PRESENT,
                                 D3D12_RESOURCE_STATE_RENDER_TARGET);

#if defined(__XBOXONE__) || defined(__XBOXSERIES__)
        D3D12_XBOX_StartFrame(data->d3dDevice, &data->frameToken);
#endif
        return 0;
    }
}

static int D3D12_SetVSync(SDL_Renderer *renderer, const int vsync)
{
    if (vsync) {
        renderer->info.flags |= SDL_RENDERER_PRESENTVSYNC;
    } else {
        renderer->info.flags &= ~SDL_RENDERER_PRESENTVSYNC;
    }
    return 0;
}

SDL_Renderer *D3D12_CreateRenderer(SDL_Window *window, Uint32 flags)
{
    SDL_Renderer *renderer;
    D3D12_RenderData *data;

    renderer = (SDL_Renderer *)SDL_calloc(1, sizeof(*renderer));
    if (!renderer) {
        SDL_OutOfMemory();
        return NULL;
    }

    data = (D3D12_RenderData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        SDL_free(renderer);
        SDL_OutOfMemory();
        return NULL;
    }

    data->identity = MatrixIdentity();

    renderer->WindowEvent = D3D12_WindowEvent;
    renderer->GetOutputSize = D3D12_GetOutputSize;
    renderer->SupportsBlendMode = D3D12_SupportsBlendMode;
    renderer->CreateTexture = D3D12_CreateTexture;
    renderer->UpdateTexture = D3D12_UpdateTexture;
#if SDL_HAVE_YUV
    renderer->UpdateTextureYUV = D3D12_UpdateTextureYUV;
    renderer->UpdateTextureNV = D3D12_UpdateTextureNV;
#endif
    renderer->LockTexture = D3D12_LockTexture;
    renderer->UnlockTexture = D3D12_UnlockTexture;
    renderer->SetTextureScaleMode = D3D12_SetTextureScaleMode;
    renderer->SetRenderTarget = D3D12_SetRenderTarget;
    renderer->QueueSetViewport = D3D12_QueueSetViewport;
    renderer->QueueSetDrawColor = D3D12_QueueSetViewport; /* SetViewport and SetDrawColor are (currently) no-ops. */
    renderer->QueueDrawPoints = D3D12_QueueDrawPoints;
    renderer->QueueDrawLines = D3D12_QueueDrawPoints; /* lines and points queue vertices the same way. */
    renderer->QueueGeometry = D3D12_QueueGeometry;
    renderer->RunCommandQueue = D3D12_RunCommandQueue;
    renderer->RenderReadPixels = D3D12_RenderReadPixels;
    renderer->RenderPresent = D3D12_RenderPresent;
    renderer->DestroyTexture = D3D12_DestroyTexture;
    renderer->DestroyRenderer = D3D12_DestroyRenderer;
    renderer->info = D3D12_RenderDriver.info;
    renderer->info.flags = (SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE);
    renderer->driverdata = data;

    if (flags & SDL_RENDERER_PRESENTVSYNC) {
        renderer->info.flags |= SDL_RENDERER_PRESENTVSYNC;
    }
    renderer->SetVSync = D3D12_SetVSync;

    /* HACK: make sure the SDL_Renderer references the SDL_Window data now, in
     * order to give init functions access to the underlying window handle:
     */
    renderer->window = window;

    /* Initialize Direct3D resources */
    if (FAILED(D3D12_CreateDeviceResources(renderer))) {
        D3D12_DestroyRenderer(renderer);
        return NULL;
    }
    if (FAILED(D3D12_CreateWindowSizeDependentResources(renderer))) {
        D3D12_DestroyRenderer(renderer);
        return NULL;
    }

    return renderer;
}

SDL_RenderDriver D3D12_RenderDriver = {
    D3D12_CreateRenderer,
    {
        "direct3d12",
        (
            SDL_RENDERER_ACCELERATED |
            SDL_RENDERER_PRESENTVSYNC |
            SDL_RENDERER_TARGETTEXTURE), /* flags.  see SDL_RendererFlags */
        6,                               /* num_texture_formats */
        {                                /* texture_formats */
          SDL_PIXELFORMAT_ARGB8888,
          SDL_PIXELFORMAT_RGB888,
          SDL_PIXELFORMAT_YV12,
          SDL_PIXELFORMAT_IYUV,
          SDL_PIXELFORMAT_NV12,
          SDL_PIXELFORMAT_NV21 },
        16384, /* max_texture_width */
        16384  /* max_texture_height */
    }
};

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif

#endif /* SDL_VIDEO_RENDER_D3D12 */

#if defined(__WIN32__) || defined(__GDK__)
#ifdef __cplusplus
extern "C"
#endif
    /* This function needs to always exist on Windows, for the Dynamic API. */
    ID3D12Device *
    SDL_RenderGetD3D12Device(SDL_Renderer *renderer)
{
    ID3D12Device *device = NULL;

#if SDL_VIDEO_RENDER_D3D12
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->driverdata;

    /* Make sure that this is a D3D renderer */
    if (renderer->DestroyRenderer != D3D12_DestroyRenderer) {
        SDL_SetError("Renderer is not a D3D12 renderer");
        return NULL;
    }

    device = (ID3D12Device *)data->d3dDevice;
    if (device) {
        D3D_CALL(device, AddRef);
    }
#endif /* SDL_VIDEO_RENDER_D3D12 */

    return device;
}
#endif /* defined(__WIN32__) || defined(__GDK__) */

/* vi: set ts=4 sw=4 expandtab: */
