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

/* Functions for audio drivers to perform runtime conversion of audio format */

#include "SDL.h"
#include "SDL_audio.h"
#include "SDL_audio_c.h"

#include "SDL_loadso.h"
#include "../SDL_dataqueue.h"
#include "SDL_cpuinfo.h"

#define DEBUG_AUDIOSTREAM 0

#ifdef __ARM_NEON
#define HAVE_NEON_INTRINSICS 1
#endif

#ifdef __SSE__
#define HAVE_SSE_INTRINSICS 1
#endif

#ifdef __SSE3__
#define HAVE_SSE3_INTRINSICS 1
#endif

#if defined(HAVE_IMMINTRIN_H) && !defined(SDL_DISABLE_IMMINTRIN_H)
#define HAVE_AVX_INTRINSICS 1
#endif
#if defined __clang__
#if (!__has_attribute(target))
#undef HAVE_AVX_INTRINSICS
#endif
#if (defined(_MSC_VER) || defined(__SCE__)) && !defined(__AVX__)
#undef HAVE_AVX_INTRINSICS
#endif
#elif defined __GNUC__
#if (__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)
#undef HAVE_AVX_INTRINSICS
#endif
#endif

/*
 * CHANNEL LAYOUTS AS SDL EXPECTS THEM:
 *
 * (Even if the platform expects something else later, that
 * SDL will swizzle between the app and the platform).
 *
 * Abbreviations:
 * - FRONT=single mono speaker
 * - FL=front left speaker
 * - FR=front right speaker
 * - FC=front center speaker
 * - BL=back left speaker
 * - BR=back right speaker
 * - SR=surround right speaker
 * - SL=surround left speaker
 * - BC=back center speaker
 * - LFE=low-frequency speaker
 *
 * These are listed in the order they are laid out in
 * memory, so "FL+FR" means "the front left speaker is
 * layed out in memory first, then the front right, then
 * it repeats for the next audio frame".
 *
 * 1 channel (mono) layout: FRONT
 * 2 channels (stereo) layout: FL+FR
 * 3 channels (2.1) layout: FL+FR+LFE
 * 4 channels (quad) layout: FL+FR+BL+BR
 * 5 channels (4.1) layout: FL+FR+LFE+BL+BR
 * 6 channels (5.1) layout: FL+FR+FC+LFE+BL+BR
 * 7 channels (6.1) layout: FL+FR+FC+LFE+BC+SL+SR
 * 8 channels (7.1) layout: FL+FR+FC+LFE+BL+BR+SL+SR
 */

#ifdef HAVE_SSE3_INTRINSICS
/* Convert from stereo to mono. Average left and right. */
static void SDLCALL SDL_ConvertStereoToMono_SSE3(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const __m128 divby2 = _mm_set1_ps(0.5f);
    float *dst = (float *)cvt->buf;
    const float *src = dst;
    int i = cvt->len_cvt / 8;

    LOG_DEBUG_CONVERT("stereo", "mono (using SSE3)");
    SDL_assert(format == AUDIO_F32SYS);

    /* Do SSE blocks as long as we have 16 bytes available.
       Just use unaligned load/stores, if the memory at runtime is
       aligned it'll be just as fast on modern processors */
    while (i >= 4) { /* 4 * float32 */
        _mm_storeu_ps(dst, _mm_mul_ps(_mm_hadd_ps(_mm_loadu_ps(src), _mm_loadu_ps(src + 4)), divby2));
        i -= 4;
        src += 8;
        dst += 4;
    }

    /* Finish off any leftovers with scalar operations. */
    while (i) {
        *dst = (src[0] + src[1]) * 0.5f;
        dst++;
        i--;
        src += 2;
    }

    cvt->len_cvt /= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, format);
    }
}
#endif

#ifdef HAVE_SSE_INTRINSICS
/* Convert from mono to stereo. Duplicate to stereo left and right. */
static void SDLCALL SDL_ConvertMonoToStereo_SSE(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    float *dst = ((float *)(cvt->buf + (cvt->len_cvt * 2))) - 8;
    const float *src = ((const float *)(cvt->buf + cvt->len_cvt)) - 4;
    int i = cvt->len_cvt / sizeof(float);

    LOG_DEBUG_CONVERT("mono", "stereo (using SSE)");
    SDL_assert(format == AUDIO_F32SYS);

    /* Do SSE blocks as long as we have 16 bytes available.
       Just use unaligned load/stores, if the memory at runtime is
       aligned it'll be just as fast on modern processors */
    /* convert backwards, since output is growing in-place. */
    while (i >= 4) {                                           /* 4 * float32 */
        const __m128 input = _mm_loadu_ps(src);                /* A B C D */
        _mm_storeu_ps(dst, _mm_unpacklo_ps(input, input));     /* A A B B */
        _mm_storeu_ps(dst + 4, _mm_unpackhi_ps(input, input)); /* C C D D */
        i -= 4;
        src -= 4;
        dst -= 8;
    }

    /* Finish off any leftovers with scalar operations. */
    src += 3;
    dst += 6;   /* adjust for smaller buffers. */
    while (i) { /* convert backwards, since output is growing in-place. */
        const float srcFC = src[0];
        dst[1] /* FR */ = srcFC;
        dst[0] /* FL */ = srcFC;
        i--;
        src--;
        dst -= 2;
    }

    cvt->len_cvt *= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, format);
    }
}
#endif

/* Include the autogenerated channel converters... */
#include "SDL_audio_channel_converters.h"

/* SDL's resampler uses a "bandlimited interpolation" algorithm:
     https://ccrma.stanford.edu/~jos/resample/ */

#include "SDL_audio_resampler_filter.h"

static Sint32 ResamplerPadding(const Sint32 inrate, const Sint32 outrate)
{
    /* This function uses integer arithmetics to avoid precision loss caused
     * by large floating point numbers. Sint32 is needed for the large number
     * multiplication. The integers are assumed to be non-negative so that
     * division rounds by truncation. */
    if (inrate == outrate) {
        return 0;
    }
    if (inrate > outrate) {
        return (RESAMPLER_SAMPLES_PER_ZERO_CROSSING * inrate + outrate - 1) / outrate;
    }
    return RESAMPLER_SAMPLES_PER_ZERO_CROSSING;
}

/* lpadding and rpadding are expected to be buffers of (ResamplePadding(inrate, outrate) * chans * sizeof(float)) bytes. */
static int SDL_ResampleAudio(const int chans, const int inrate, const int outrate,
                             const float *lpadding, const float *rpadding,
                             const float *inbuf, const int inbuflen,
                             float *outbuf, const int outbuflen)
{
    /* This function uses integer arithmetics to avoid precision loss caused
     * by large floating point numbers. For some operations, Sint32 or Sint64
     * are needed for the large number multiplications. The input integers are
     * assumed to be non-negative so that division rounds by truncation and
     * modulo is always non-negative. Note that the operator order is important
     * for these integer divisions. */
    const int paddinglen = ResamplerPadding(inrate, outrate);
    const int framelen = chans * (int)sizeof(float);
    const int inframes = inbuflen / framelen;
    /* outbuflen isn't total to write, it's total available. */
    const int wantedoutframes = (int)((Sint64)inframes * outrate / inrate);
    const int maxoutframes = outbuflen / framelen;
    const int outframes = SDL_min(wantedoutframes, maxoutframes);
    float *dst = outbuf;
    int i, j, chan;

    for (i = 0; i < outframes; i++) {
        const int srcindex = (int)((Sint64)i * inrate / outrate);
        /* Calculating the following way avoids subtraction or modulo of large
         * floats which have low result precision.
         *   interpolation1
         * = (i / outrate * inrate) - floor(i / outrate * inrate)
         * = mod(i / outrate * inrate, 1)
         * = mod(i * inrate, outrate) / outrate */
        const int srcfraction = ((Sint64)i) * inrate % outrate;
        const float interpolation1 = ((float)srcfraction) / ((float)outrate);
        const int filterindex1 = ((Sint32)srcfraction) * RESAMPLER_SAMPLES_PER_ZERO_CROSSING / outrate;
        const float interpolation2 = 1.0f - interpolation1;
        const int filterindex2 = ((Sint32)(outrate - srcfraction)) * RESAMPLER_SAMPLES_PER_ZERO_CROSSING / outrate;

        for (chan = 0; chan < chans; chan++) {
            float outsample = 0.0f;

            /* do this twice to calculate the sample, once for the "left wing" and then same for the right. */
            for (j = 0; (filterindex1 + (j * RESAMPLER_SAMPLES_PER_ZERO_CROSSING)) < RESAMPLER_FILTER_SIZE; j++) {
                const int filt_ind = filterindex1 + j * RESAMPLER_SAMPLES_PER_ZERO_CROSSING;
                const int srcframe = srcindex - j;
                /* !!! FIXME: we can bubble this conditional out of here by doing a pre loop. */
                const float insample = (srcframe < 0) ? lpadding[((paddinglen + srcframe) * chans) + chan] : inbuf[(srcframe * chans) + chan];
                outsample += (float) (insample * (ResamplerFilter[filt_ind] + (interpolation1 * ResamplerFilterDifference[filt_ind])));
            }

            /* Do the right wing! */
            for (j = 0; (filterindex2 + (j * RESAMPLER_SAMPLES_PER_ZERO_CROSSING)) < RESAMPLER_FILTER_SIZE; j++) {
                const int filt_ind = filterindex2 + j * RESAMPLER_SAMPLES_PER_ZERO_CROSSING;
                const int srcframe = srcindex + 1 + j;
                /* !!! FIXME: we can bubble this conditional out of here by doing a post loop. */
                const float insample = (srcframe >= inframes) ? rpadding[((srcframe - inframes) * chans) + chan] : inbuf[(srcframe * chans) + chan];
                outsample += (float) (insample * (ResamplerFilter[filt_ind] + (interpolation2 * ResamplerFilterDifference[filt_ind])));
            }

            *(dst++) = outsample;
        }
    }

    return outframes * chans * sizeof(float);
}

int SDL_ConvertAudio(SDL_AudioCVT *cvt)
{
    /* !!! FIXME: (cvt) should be const; stack-copy it here. */
    /* !!! FIXME: (actually, we can't...len_cvt needs to be updated. Grr.) */

    /* Make sure there's data to convert */
    if (!cvt->buf) {
        return SDL_SetError("No buffer allocated for conversion");
    }

    /* Return okay if no conversion is necessary */
    cvt->len_cvt = cvt->len;
    if (cvt->filters[0] == NULL) {
        return 0;
    }

    /* Set up the conversion and go! */
    cvt->filter_index = 0;
    cvt->filters[0](cvt, cvt->src_format);
    return 0;
}

static void SDLCALL SDL_Convert_Byteswap(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
#if DEBUG_CONVERT
    SDL_Log("SDL_AUDIO_CONVERT: Converting byte order\n");
#endif

    switch (SDL_AUDIO_BITSIZE(format)) {
#define CASESWAP(b)                                            \
    case b:                                                    \
    {                                                          \
        Uint##b *ptr = (Uint##b *)cvt->buf;                    \
        int i;                                                 \
        for (i = cvt->len_cvt / sizeof(*ptr); i; --i, ++ptr) { \
            *ptr = SDL_Swap##b(*ptr);                          \
        }                                                      \
        break;                                                 \
    }

        CASESWAP(16);
        CASESWAP(32);
        CASESWAP(64);

#undef CASESWAP

    default:
        SDL_assert(!"unhandled byteswap datatype!");
        break;
    }

    if (cvt->filters[++cvt->filter_index]) {
        /* flip endian flag for data. */
        if (format & SDL_AUDIO_MASK_ENDIAN) {
            format &= ~SDL_AUDIO_MASK_ENDIAN;
        } else {
            format |= SDL_AUDIO_MASK_ENDIAN;
        }
        cvt->filters[cvt->filter_index](cvt, format);
    }
}

static int SDL_AddAudioCVTFilter(SDL_AudioCVT *cvt, SDL_AudioFilter filter)
{
    if (cvt->filter_index >= SDL_AUDIOCVT_MAX_FILTERS) {
        return SDL_SetError("Too many filters needed for conversion, exceeded maximum of %d", SDL_AUDIOCVT_MAX_FILTERS);
    }
    SDL_assert(filter != NULL);
    cvt->filters[cvt->filter_index++] = filter;
    cvt->filters[cvt->filter_index] = NULL; /* Moving terminator */
    return 0;
}

static int SDL_BuildAudioTypeCVTToFloat(SDL_AudioCVT *cvt, const SDL_AudioFormat src_fmt)
{
    int retval = 0; /* 0 == no conversion necessary. */

    if ((SDL_AUDIO_ISBIGENDIAN(src_fmt) != 0) == (SDL_BYTEORDER == SDL_LIL_ENDIAN) && SDL_AUDIO_BITSIZE(src_fmt) > 8) {
        if (SDL_AddAudioCVTFilter(cvt, SDL_Convert_Byteswap) < 0) {
            return -1;
        }
        retval = 1; /* added a converter. */
    }

    if (!SDL_AUDIO_ISFLOAT(src_fmt)) {
        const Uint16 src_bitsize = SDL_AUDIO_BITSIZE(src_fmt);
        const Uint16 dst_bitsize = 32;
        SDL_AudioFilter filter = NULL;

        switch (src_fmt & ~SDL_AUDIO_MASK_ENDIAN) {
        case AUDIO_S8:
            filter = SDL_Convert_S8_to_F32;
            break;
        case AUDIO_U8:
            filter = SDL_Convert_U8_to_F32;
            break;
        case AUDIO_S16:
            filter = SDL_Convert_S16_to_F32;
            break;
        case AUDIO_U16:
            filter = SDL_Convert_U16_to_F32;
            break;
        case AUDIO_S32:
            filter = SDL_Convert_S32_to_F32;
            break;
        default:
            SDL_assert(!"Unexpected audio format!");
            break;
        }

        if (!filter) {
            return SDL_SetError("No conversion from source format to float available");
        }

        if (SDL_AddAudioCVTFilter(cvt, filter) < 0) {
            return -1;
        }
        if (src_bitsize < dst_bitsize) {
            const int mult = (dst_bitsize / src_bitsize);
            cvt->len_mult *= mult;
            cvt->len_ratio *= mult;
        } else if (src_bitsize > dst_bitsize) {
            const int div = (src_bitsize / dst_bitsize);
            cvt->len_ratio /= div;
        }

        retval = 1; /* added a converter. */
    }

    return retval;
}

static int SDL_BuildAudioTypeCVTFromFloat(SDL_AudioCVT *cvt, const SDL_AudioFormat dst_fmt)
{
    int retval = 0; /* 0 == no conversion necessary. */

    if (!SDL_AUDIO_ISFLOAT(dst_fmt)) {
        const Uint16 dst_bitsize = SDL_AUDIO_BITSIZE(dst_fmt);
        const Uint16 src_bitsize = 32;
        SDL_AudioFilter filter = NULL;
        switch (dst_fmt & ~SDL_AUDIO_MASK_ENDIAN) {
        case AUDIO_S8:
            filter = SDL_Convert_F32_to_S8;
            break;
        case AUDIO_U8:
            filter = SDL_Convert_F32_to_U8;
            break;
        case AUDIO_S16:
            filter = SDL_Convert_F32_to_S16;
            break;
        case AUDIO_U16:
            filter = SDL_Convert_F32_to_U16;
            break;
        case AUDIO_S32:
            filter = SDL_Convert_F32_to_S32;
            break;
        default:
            SDL_assert(!"Unexpected audio format!");
            break;
        }

        if (!filter) {
            return SDL_SetError("No conversion from float to format 0x%.4x available", dst_fmt);
        }

        if (SDL_AddAudioCVTFilter(cvt, filter) < 0) {
            return -1;
        }
        if (src_bitsize < dst_bitsize) {
            const int mult = (dst_bitsize / src_bitsize);
            cvt->len_mult *= mult;
            cvt->len_ratio *= mult;
        } else if (src_bitsize > dst_bitsize) {
            const int div = (src_bitsize / dst_bitsize);
            cvt->len_ratio /= div;
        }
        retval = 1; /* added a converter. */
    }

    if ((SDL_AUDIO_ISBIGENDIAN(dst_fmt) != 0) == (SDL_BYTEORDER == SDL_LIL_ENDIAN) && SDL_AUDIO_BITSIZE(dst_fmt) > 8) {
        if (SDL_AddAudioCVTFilter(cvt, SDL_Convert_Byteswap) < 0) {
            return -1;
        }
        retval = 1; /* added a converter. */
    }

    return retval;
}

#ifdef HAVE_LIBSAMPLERATE_H

static void SDL_ResampleCVT_SRC(SDL_AudioCVT *cvt, const int chans, const SDL_AudioFormat format)
{
    const float *src = (const float *)cvt->buf;
    const int srclen = cvt->len_cvt;
    float *dst = (float *)(cvt->buf + srclen);
    const int dstlen = (cvt->len * cvt->len_mult) - srclen;
    const int framelen = sizeof(float) * chans;
    int result = 0;
    SRC_DATA data;

    SDL_zero(data);

    data.data_in = (float *)src; /* Older versions of libsamplerate had a non-const pointer, but didn't write to it */
    data.input_frames = srclen / framelen;

    data.data_out = dst;
    data.output_frames = dstlen / framelen;

    data.src_ratio = cvt->rate_incr;

    result = SRC_src_simple(&data, SRC_converter, chans); /* Simple API converts the whole buffer at once.  No need for initialization. */
/* !!! FIXME: Handle library failures? */
#if DEBUG_CONVERT
    if (result != 0) {
        SDL_Log("src_simple() failed: %s", SRC_src_strerror(result));
    }
#else
    (void)result;
#endif

    cvt->len_cvt = data.output_frames_gen * framelen;

    SDL_memmove(cvt->buf, dst, cvt->len_cvt);

    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, format);
    }
}

#endif /* HAVE_LIBSAMPLERATE_H */

static void SDL_ResampleCVT(SDL_AudioCVT *cvt, const int chans, const SDL_AudioFormat format)
{
    /* !!! FIXME in 2.1: there are ten slots in the filter list, and the theoretical maximum we use is six (seven with NULL terminator).
       !!! FIXME in 2.1:   We need to store data for this resampler, because the cvt structure doesn't store the original sample rates,
       !!! FIXME in 2.1:   so we steal the ninth and tenth slot.  :( */
    const int inrate = (int)(size_t)cvt->filters[SDL_AUDIOCVT_MAX_FILTERS - 1];
    const int outrate = (int)(size_t)cvt->filters[SDL_AUDIOCVT_MAX_FILTERS];
    const float *src = (const float *)cvt->buf;
    const int srclen = cvt->len_cvt;
    /*float *dst = (float *) cvt->buf;
    const int dstlen = (cvt->len * cvt->len_mult);*/
    /* !!! FIXME: remove this if we can get the resampler to work in-place again. */
    float *dst = (float *)(cvt->buf + srclen);
    const int dstlen = (cvt->len * cvt->len_mult) - srclen;
    const int requestedpadding = ResamplerPadding(inrate, outrate);
    int paddingsamples;
    float *padding;

    if (requestedpadding < SDL_MAX_SINT32 / chans) {
        paddingsamples = requestedpadding * chans;
    } else {
        paddingsamples = 0;
    }
    SDL_assert(format == AUDIO_F32SYS);

    /* we keep no streaming state here, so pad with silence on both ends. */
    padding = (float *)SDL_calloc(paddingsamples ? paddingsamples : 1, sizeof(float));
    if (!padding) {
        SDL_OutOfMemory();
        return;
    }

    cvt->len_cvt = SDL_ResampleAudio(chans, inrate, outrate, padding, padding, src, srclen, dst, dstlen);

    SDL_free(padding);

    SDL_memmove(cvt->buf, dst, cvt->len_cvt); /* !!! FIXME: remove this if we can get the resampler to work in-place again. */

    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, format);
    }
}

/* !!! FIXME: We only have this macro salsa because SDL_AudioCVT doesn't
   !!! FIXME:  store channel info, so we have to have function entry
   !!! FIXME:  points for each supported channel count and multiple
   !!! FIXME:  vs arbitrary. When we rev the ABI, clean this up. */
#define RESAMPLER_FUNCS(chans)                                              \
    static void SDLCALL                                                     \
        SDL_ResampleCVT_c##chans(SDL_AudioCVT *cvt, SDL_AudioFormat format) \
    {                                                                       \
        SDL_ResampleCVT(cvt, chans, format);                                \
    }
RESAMPLER_FUNCS(1)
RESAMPLER_FUNCS(2)
RESAMPLER_FUNCS(4)
RESAMPLER_FUNCS(6)
RESAMPLER_FUNCS(8)
#undef RESAMPLER_FUNCS

#ifdef HAVE_LIBSAMPLERATE_H
#define RESAMPLER_FUNCS(chans)                                                  \
    static void SDLCALL                                                         \
        SDL_ResampleCVT_SRC_c##chans(SDL_AudioCVT *cvt, SDL_AudioFormat format) \
    {                                                                           \
        SDL_ResampleCVT_SRC(cvt, chans, format);                                \
    }
RESAMPLER_FUNCS(1)
RESAMPLER_FUNCS(2)
RESAMPLER_FUNCS(4)
RESAMPLER_FUNCS(6)
RESAMPLER_FUNCS(8)
#undef RESAMPLER_FUNCS
#endif /* HAVE_LIBSAMPLERATE_H */

static SDL_AudioFilter ChooseCVTResampler(const int dst_channels)
{
#ifdef HAVE_LIBSAMPLERATE_H
    if (SRC_available) {
        switch (dst_channels) {
        case 1:
            return SDL_ResampleCVT_SRC_c1;
        case 2:
            return SDL_ResampleCVT_SRC_c2;
        case 4:
            return SDL_ResampleCVT_SRC_c4;
        case 6:
            return SDL_ResampleCVT_SRC_c6;
        case 8:
            return SDL_ResampleCVT_SRC_c8;
        default:
            break;
        }
    }
#endif /* HAVE_LIBSAMPLERATE_H */

    switch (dst_channels) {
    case 1:
        return SDL_ResampleCVT_c1;
    case 2:
        return SDL_ResampleCVT_c2;
    case 4:
        return SDL_ResampleCVT_c4;
    case 6:
        return SDL_ResampleCVT_c6;
    case 8:
        return SDL_ResampleCVT_c8;
    default:
        break;
    }

    return NULL;
}

static int SDL_BuildAudioResampleCVT(SDL_AudioCVT *cvt, const int dst_channels,
                                     const int src_rate, const int dst_rate)
{
    SDL_AudioFilter filter;

    if (src_rate == dst_rate) {
        return 0; /* no conversion necessary. */
    }

    filter = ChooseCVTResampler(dst_channels);
    if (!filter) {
        return SDL_SetError("No conversion available for these rates");
    }

    /* Update (cvt) with filter details... */
    if (SDL_AddAudioCVTFilter(cvt, filter) < 0) {
        return -1;
    }

    /* !!! FIXME in 2.1: there are ten slots in the filter list, and the theoretical maximum we use is six (seven with NULL terminator).
       !!! FIXME in 2.1:   We need to store data for this resampler, because the cvt structure doesn't store the original sample rates,
       !!! FIXME in 2.1:   so we steal the ninth and tenth slot.  :( */
    if (cvt->filter_index >= (SDL_AUDIOCVT_MAX_FILTERS - 2)) {
        return SDL_SetError("Too many filters needed for conversion, exceeded maximum of %d", SDL_AUDIOCVT_MAX_FILTERS - 2);
    }
    cvt->filters[SDL_AUDIOCVT_MAX_FILTERS - 1] = (SDL_AudioFilter)(uintptr_t)src_rate;
    cvt->filters[SDL_AUDIOCVT_MAX_FILTERS] = (SDL_AudioFilter)(uintptr_t)dst_rate;

    if (src_rate < dst_rate) {
        const double mult = ((double)dst_rate) / ((double)src_rate);
        cvt->len_mult *= (int)SDL_ceil(mult);
        cvt->len_ratio *= mult;
    } else {
        cvt->len_ratio /= ((double)src_rate) / ((double)dst_rate);
    }

    /* !!! FIXME: remove this if we can get the resampler to work in-place again. */
    /* the buffer is big enough to hold the destination now, but
       we need it large enough to hold a separate scratch buffer. */
    cvt->len_mult *= 2;

    return 1; /* added a converter. */
}

static SDL_bool SDL_SupportedAudioFormat(const SDL_AudioFormat fmt)
{
    switch (fmt) {
    case AUDIO_U8:
    case AUDIO_S8:
    case AUDIO_U16LSB:
    case AUDIO_S16LSB:
    case AUDIO_U16MSB:
    case AUDIO_S16MSB:
    case AUDIO_S32LSB:
    case AUDIO_S32MSB:
    case AUDIO_F32LSB:
    case AUDIO_F32MSB:
        return SDL_TRUE; /* supported. */

    default:
        break;
    }

    return SDL_FALSE; /* unsupported. */
}

static SDL_bool SDL_SupportedChannelCount(const int channels)
{
    return ((channels >= 1) && (channels <= 8)) ? SDL_TRUE : SDL_FALSE;
}

/* Creates a set of audio filters to convert from one format to another.
   Returns 0 if no conversion is needed, 1 if the audio filter is set up,
   or -1 if an error like invalid parameter, unsupported format, etc. occurred.
*/

int SDL_BuildAudioCVT(SDL_AudioCVT *cvt,
                      SDL_AudioFormat src_format, Uint8 src_channels, int src_rate,
                      SDL_AudioFormat dst_format, Uint8 dst_channels, int dst_rate)
{
    SDL_AudioFilter channel_converter = NULL;

    /* Sanity check target pointer */
    if (!cvt) {
        return SDL_InvalidParamError("cvt");
    }

    /* Make sure we zero out the audio conversion before error checking */
    SDL_zerop(cvt);

    if (!SDL_SupportedAudioFormat(src_format)) {
        return SDL_SetError("Invalid source format");
    }
    if (!SDL_SupportedAudioFormat(dst_format)) {
        return SDL_SetError("Invalid destination format");
    }
    if (!SDL_SupportedChannelCount(src_channels)) {
        return SDL_SetError("Invalid source channels");
    }
    if (!SDL_SupportedChannelCount(dst_channels)) {
        return SDL_SetError("Invalid destination channels");
    }
    if (src_rate <= 0) {
        return SDL_SetError("Source rate is equal to or less than zero");
    }
    if (dst_rate <= 0) {
        return SDL_SetError("Destination rate is equal to or less than zero");
    }
    if (src_rate >= SDL_MAX_SINT32 / RESAMPLER_SAMPLES_PER_ZERO_CROSSING) {
        return SDL_SetError("Source rate is too high");
    }
    if (dst_rate >= SDL_MAX_SINT32 / RESAMPLER_SAMPLES_PER_ZERO_CROSSING) {
        return SDL_SetError("Destination rate is too high");
    }

#if DEBUG_CONVERT
    SDL_Log("SDL_AUDIO_CONVERT: Build format %04x->%04x, channels %u->%u, rate %d->%d\n",
            src_format, dst_format, src_channels, dst_channels, src_rate, dst_rate);
#endif

    /* Start off with no conversion necessary */
    cvt->src_format = src_format;
    cvt->dst_format = dst_format;
    cvt->needed = 0;
    cvt->filter_index = 0;
    SDL_zeroa(cvt->filters);
    cvt->len_mult = 1;
    cvt->len_ratio = 1.0;
    cvt->rate_incr = ((double)dst_rate) / ((double)src_rate);

    /* Make sure we've chosen audio conversion functions (SIMD, scalar, etc.) */
    SDL_ChooseAudioConverters();

    /* Type conversion goes like this now:
        - byteswap to CPU native format first if necessary.
        - convert to native Float32 if necessary.
        - resample and change channel count if necessary.
        - convert to final data format.
        - byteswap back to foreign format if necessary.

       The expectation is we can process data faster in float32
       (possibly with SIMD), and making several passes over the same
       buffer is likely to be CPU cache-friendly, avoiding the
       biggest performance hit in modern times. Previously we had
       (script-generated) custom converters for every data type and
       it was a bloat on SDL compile times and final library size. */

    /* see if we can skip float conversion entirely. */
    if (src_rate == dst_rate && src_channels == dst_channels) {
        if (src_format == dst_format) {
            return 0;
        }

        /* just a byteswap needed? */
        if ((src_format & ~SDL_AUDIO_MASK_ENDIAN) == (dst_format & ~SDL_AUDIO_MASK_ENDIAN)) {
            if (SDL_AUDIO_BITSIZE(dst_format) == 8) {
                return 0;
            }
            if (SDL_AddAudioCVTFilter(cvt, SDL_Convert_Byteswap) < 0) {
                return -1;
            }
            cvt->needed = 1;
            return 1;
        }
    }

    /* Convert data types, if necessary. Updates (cvt). */
    if (SDL_BuildAudioTypeCVTToFloat(cvt, src_format) < 0) {
        return -1; /* shouldn't happen, but just in case... */
    }

    /* Channel conversion */

    /* SDL_SupportedChannelCount should have caught these asserts, or we added a new format and forgot to update the table. */
    SDL_assert(src_channels <= SDL_arraysize(channel_converters));
    SDL_assert(dst_channels <= SDL_arraysize(channel_converters[0]));

    channel_converter = channel_converters[src_channels - 1][dst_channels - 1];
    if ((!channel_converter) != (src_channels == dst_channels)) {
        /* All combinations of supported channel counts should have been handled by now, but let's be defensive */
        return SDL_SetError("Invalid channel combination");
    } else if (channel_converter != NULL) {
        /* swap in some SIMD versions for a few of these. */
        if (channel_converter == SDL_ConvertStereoToMono) {
            SDL_AudioFilter filter = NULL;
#ifdef HAVE_SSE3_INTRINSICS
            if (!filter && SDL_HasSSE3()) {
                filter = SDL_ConvertStereoToMono_SSE3;
            }
#endif
            if (filter) {
                channel_converter = filter;
            }
        } else if (channel_converter == SDL_ConvertMonoToStereo) {
            SDL_AudioFilter filter = NULL;
#ifdef HAVE_SSE_INTRINSICS
            if (!filter && SDL_HasSSE()) {
                filter = SDL_ConvertMonoToStereo_SSE;
            }
#endif
            if (filter) {
                channel_converter = filter;
            }
        }

        if (SDL_AddAudioCVTFilter(cvt, channel_converter) < 0) {
            return -1;
        }

        if (src_channels < dst_channels) {
            cvt->len_mult = ((cvt->len_mult * dst_channels) + (src_channels - 1)) / src_channels;
        }

        cvt->len_ratio = (cvt->len_ratio * dst_channels) / src_channels;
        src_channels = dst_channels;
    }

    /* Do rate conversion, if necessary. Updates (cvt). */
    if (SDL_BuildAudioResampleCVT(cvt, dst_channels, src_rate, dst_rate) < 0) {
        return -1; /* shouldn't happen, but just in case... */
    }

    /* Move to final data type. */
    if (SDL_BuildAudioTypeCVTFromFloat(cvt, dst_format) < 0) {
        return -1; /* shouldn't happen, but just in case... */
    }

    cvt->needed = (cvt->filter_index != 0);
    return cvt->needed;
}

typedef int (*SDL_ResampleAudioStreamFunc)(SDL_AudioStream *stream, const void *inbuf, const int inbuflen, void *outbuf, const int outbuflen);
typedef void (*SDL_ResetAudioStreamResamplerFunc)(SDL_AudioStream *stream);
typedef void (*SDL_CleanupAudioStreamResamplerFunc)(SDL_AudioStream *stream);

struct _SDL_AudioStream
{
    SDL_AudioCVT cvt_before_resampling;
    SDL_AudioCVT cvt_after_resampling;
    SDL_DataQueue *queue;
    SDL_bool first_run;
    Uint8 *staging_buffer;
    int staging_buffer_size;
    int staging_buffer_filled;
    Uint8 *work_buffer_base; /* maybe unaligned pointer from SDL_realloc(). */
    int work_buffer_len;
    int src_sample_frame_size;
    SDL_AudioFormat src_format;
    Uint8 src_channels;
    int src_rate;
    int dst_sample_frame_size;
    SDL_AudioFormat dst_format;
    Uint8 dst_channels;
    int dst_rate;
    double rate_incr;
    Uint8 pre_resample_channels;
    int packetlen;
    int resampler_padding_samples;
    float *resampler_padding;
    void *resampler_state;
    SDL_ResampleAudioStreamFunc resampler_func;
    SDL_ResetAudioStreamResamplerFunc reset_resampler_func;
    SDL_CleanupAudioStreamResamplerFunc cleanup_resampler_func;
};

static Uint8 *EnsureStreamBufferSize(SDL_AudioStream *stream, int newlen)
{
    Uint8 *ptr;
    size_t offset;

    if (stream->work_buffer_len >= newlen) {
        ptr = stream->work_buffer_base;
    } else {
        ptr = (Uint8 *)SDL_realloc(stream->work_buffer_base, (size_t)newlen + 32);
        if (!ptr) {
            SDL_OutOfMemory();
            return NULL;
        }
        /* Make sure we're aligned to 16 bytes for SIMD code. */
        stream->work_buffer_base = ptr;
        stream->work_buffer_len = newlen;
    }

    offset = ((size_t)ptr) & 15;
    return offset ? ptr + (16 - offset) : ptr;
}

#ifdef HAVE_LIBSAMPLERATE_H
static int SDL_ResampleAudioStream_SRC(SDL_AudioStream *stream, const void *_inbuf, const int inbuflen, void *_outbuf, const int outbuflen)
{
    const float *inbuf = (const float *)_inbuf;
    float *outbuf = (float *)_outbuf;
    const int framelen = sizeof(float) * stream->pre_resample_channels;
    SRC_STATE *state = (SRC_STATE *)stream->resampler_state;
    SRC_DATA data;
    int result;

    SDL_assert(inbuf != ((const float *)outbuf)); /* SDL_AudioStreamPut() shouldn't allow in-place resamples. */

    data.data_in = (float *)inbuf; /* Older versions of libsamplerate had a non-const pointer, but didn't write to it */
    data.input_frames = inbuflen / framelen;
    data.input_frames_used = 0;

    data.data_out = outbuf;
    data.output_frames = outbuflen / framelen;

    data.end_of_input = 0;
    data.src_ratio = stream->rate_incr;

    result = SRC_src_process(state, &data);
    if (result != 0) {
        SDL_SetError("src_process() failed: %s", SRC_src_strerror(result));
        return 0;
    }

    /* If this fails, we need to store them off somewhere */
    SDL_assert(data.input_frames_used == data.input_frames);

    return data.output_frames_gen * (sizeof(float) * stream->pre_resample_channels);
}

static void SDL_ResetAudioStreamResampler_SRC(SDL_AudioStream *stream)
{
    SRC_src_reset((SRC_STATE *)stream->resampler_state);
}

static void SDL_CleanupAudioStreamResampler_SRC(SDL_AudioStream *stream)
{
    SRC_STATE *state = (SRC_STATE *)stream->resampler_state;
    if (state) {
        SRC_src_delete(state);
    }

    stream->resampler_state = NULL;
    stream->resampler_func = NULL;
    stream->reset_resampler_func = NULL;
    stream->cleanup_resampler_func = NULL;
}

static SDL_bool SetupLibSampleRateResampling(SDL_AudioStream *stream)
{
    int result = 0;
    SRC_STATE *state = NULL;

    if (SRC_available) {
        state = SRC_src_new(SRC_converter, stream->pre_resample_channels, &result);
        if (!state) {
            SDL_SetError("src_new() failed: %s", SRC_src_strerror(result));
        }
    }

    if (!state) {
        SDL_CleanupAudioStreamResampler_SRC(stream);
        return SDL_FALSE;
    }

    stream->resampler_state = state;
    stream->resampler_func = SDL_ResampleAudioStream_SRC;
    stream->reset_resampler_func = SDL_ResetAudioStreamResampler_SRC;
    stream->cleanup_resampler_func = SDL_CleanupAudioStreamResampler_SRC;

    return SDL_TRUE;
}
#endif /* HAVE_LIBSAMPLERATE_H */

static int SDL_ResampleAudioStream(SDL_AudioStream *stream, const void *_inbuf, const int inbuflen, void *_outbuf, const int outbuflen)
{
    const Uint8 *inbufend = ((const Uint8 *)_inbuf) + inbuflen;
    const float *inbuf = (const float *)_inbuf;
    float *outbuf = (float *)_outbuf;
    const int chans = (int)stream->pre_resample_channels;
    const int inrate = stream->src_rate;
    const int outrate = stream->dst_rate;
    const int paddingsamples = stream->resampler_padding_samples;
    const int paddingbytes = paddingsamples * sizeof(float);
    float *lpadding = (float *)stream->resampler_state;
    const float *rpadding = (const float *)inbufend; /* we set this up so there are valid padding samples at the end of the input buffer. */
    const int cpy = SDL_min(inbuflen, paddingbytes);
    int retval;

    SDL_assert(inbuf != ((const float *)outbuf)); /* SDL_AudioStreamPut() shouldn't allow in-place resamples. */

    retval = SDL_ResampleAudio(chans, inrate, outrate, lpadding, rpadding, inbuf, inbuflen, outbuf, outbuflen);

    /* update our left padding with end of current input, for next run. */
    SDL_memcpy((lpadding + paddingsamples) - (cpy / sizeof(float)), inbufend - cpy, cpy);
    return retval;
}

static void SDL_ResetAudioStreamResampler(SDL_AudioStream *stream)
{
    /* set all the padding to silence. */
    const int len = stream->resampler_padding_samples;
    SDL_memset(stream->resampler_state, '\0', len * sizeof(float));
}

static void SDL_CleanupAudioStreamResampler(SDL_AudioStream *stream)
{
    SDL_free(stream->resampler_state);
}

SDL_AudioStream *SDL_NewAudioStream(const SDL_AudioFormat src_format,
                   const Uint8 src_channels,
                   const int src_rate,
                   const SDL_AudioFormat dst_format,
                   const Uint8 dst_channels,
                   const int dst_rate)
{
    int packetlen = 4096; /* !!! FIXME: good enough for now. */
    Uint8 pre_resample_channels;
    SDL_AudioStream *retval;

    if (src_channels == 0) {
        SDL_InvalidParamError("src_channels");
        return NULL;
    }

    if (dst_channels == 0) {
        SDL_InvalidParamError("dst_channels");
        return NULL;
    }

    retval = (SDL_AudioStream *)SDL_calloc(1, sizeof(SDL_AudioStream));
    if (!retval) {
        SDL_OutOfMemory();
        return NULL;
    }

    /* If increasing channels, do it after resampling, since we'd just
       do more work to resample duplicate channels. If we're decreasing, do
       it first so we resample the interpolated data instead of interpolating
       the resampled data (!!! FIXME: decide if that works in practice, though!). */
    pre_resample_channels = SDL_min(src_channels, dst_channels);

    retval->first_run = SDL_TRUE;
    retval->src_sample_frame_size = (SDL_AUDIO_BITSIZE(src_format) / 8) * src_channels;
    retval->src_format = src_format;
    retval->src_channels = src_channels;
    retval->src_rate = src_rate;
    retval->dst_sample_frame_size = (SDL_AUDIO_BITSIZE(dst_format) / 8) * dst_channels;
    retval->dst_format = dst_format;
    retval->dst_channels = dst_channels;
    retval->dst_rate = dst_rate;
    retval->pre_resample_channels = pre_resample_channels;
    retval->packetlen = packetlen;
    retval->rate_incr = ((double)dst_rate) / ((double)src_rate);
    retval->resampler_padding_samples = ResamplerPadding(retval->src_rate, retval->dst_rate) * pre_resample_channels;
    retval->resampler_padding = (float *)SDL_calloc(retval->resampler_padding_samples ? retval->resampler_padding_samples : 1, sizeof(float));

    if (!retval->resampler_padding) {
        SDL_FreeAudioStream(retval);
        SDL_OutOfMemory();
        return NULL;
    }

    retval->staging_buffer_size = ((retval->resampler_padding_samples / retval->pre_resample_channels) * retval->src_sample_frame_size);
    if (retval->staging_buffer_size > 0) {
        retval->staging_buffer = (Uint8 *)SDL_malloc(retval->staging_buffer_size);
        if (!retval->staging_buffer) {
            SDL_FreeAudioStream(retval);
            SDL_OutOfMemory();
            return NULL;
        }
    }

    /* Not resampling? It's an easy conversion (and maybe not even that!) */
    if (src_rate == dst_rate) {
        retval->cvt_before_resampling.needed = SDL_FALSE;
        if (SDL_BuildAudioCVT(&retval->cvt_after_resampling, src_format, src_channels, dst_rate, dst_format, dst_channels, dst_rate) < 0) {
            SDL_FreeAudioStream(retval);
            return NULL; /* SDL_BuildAudioCVT should have called SDL_SetError. */
        }
    } else {
        /* Don't resample at first. Just get us to Float32 format. */
        /* !!! FIXME: convert to int32 on devices without hardware float. */
        if (SDL_BuildAudioCVT(&retval->cvt_before_resampling, src_format, src_channels, src_rate, AUDIO_F32SYS, pre_resample_channels, src_rate) < 0) {
            SDL_FreeAudioStream(retval);
            return NULL; /* SDL_BuildAudioCVT should have called SDL_SetError. */
        }

#ifdef HAVE_LIBSAMPLERATE_H
        SetupLibSampleRateResampling(retval);
#endif

        if (!retval->resampler_func) {
            retval->resampler_state = SDL_calloc(retval->resampler_padding_samples, sizeof(float));
            if (!retval->resampler_state) {
                SDL_FreeAudioStream(retval);
                SDL_OutOfMemory();
                return NULL;
            }

            retval->resampler_func = SDL_ResampleAudioStream;
            retval->reset_resampler_func = SDL_ResetAudioStreamResampler;
            retval->cleanup_resampler_func = SDL_CleanupAudioStreamResampler;
        }

        /* Convert us to the final format after resampling. */
        if (SDL_BuildAudioCVT(&retval->cvt_after_resampling, AUDIO_F32SYS, pre_resample_channels, dst_rate, dst_format, dst_channels, dst_rate) < 0) {
            SDL_FreeAudioStream(retval);
            return NULL; /* SDL_BuildAudioCVT should have called SDL_SetError. */
        }
    }

    retval->queue = SDL_NewDataQueue(packetlen, (size_t)packetlen * 2);
    if (!retval->queue) {
        SDL_FreeAudioStream(retval);
        return NULL; /* SDL_NewDataQueue should have called SDL_SetError. */
    }

    return retval;
}

static int SDL_AudioStreamPutInternal(SDL_AudioStream *stream, const void *buf, int len, int *maxputbytes)
{
    int buflen = len;
    int workbuflen;
    Uint8 *workbuf;
    Uint8 *resamplebuf = NULL;
    int resamplebuflen = 0;
    int neededpaddingbytes;
    int paddingbytes;

    /* !!! FIXME: several converters can take advantage of SIMD, but only
       !!! FIXME:  if the data is aligned to 16 bytes. EnsureStreamBufferSize()
       !!! FIXME:  guarantees the buffer will align, but the
       !!! FIXME:  converters will iterate over the data backwards if
       !!! FIXME:  the output grows, and this means we won't align if buflen
       !!! FIXME:  isn't a multiple of 16. In these cases, we should chop off
       !!! FIXME:  a few samples at the end and convert them separately. */

    /* no padding prepended on first run. */
    neededpaddingbytes = stream->resampler_padding_samples * sizeof(float);
    paddingbytes = stream->first_run ? 0 : neededpaddingbytes;
    stream->first_run = SDL_FALSE;

    /* Make sure the work buffer can hold all the data we need at once... */
    workbuflen = buflen;
    if (stream->cvt_before_resampling.needed) {
        workbuflen *= stream->cvt_before_resampling.len_mult;
    }

    if (stream->dst_rate != stream->src_rate) {
        /* resamples can't happen in place, so make space for second buf. */
        const int framesize = stream->pre_resample_channels * sizeof(float);
        const int frames = workbuflen / framesize;
        resamplebuflen = ((int)SDL_ceil(frames * stream->rate_incr)) * framesize;
#if DEBUG_AUDIOSTREAM
        SDL_Log("AUDIOSTREAM: will resample %d bytes to %d (ratio=%.6f)\n", workbuflen, resamplebuflen, stream->rate_incr);
#endif
        workbuflen += resamplebuflen;
    }

    if (stream->cvt_after_resampling.needed) {
        /* !!! FIXME: buffer might be big enough already? */
        workbuflen *= stream->cvt_after_resampling.len_mult;
    }

    workbuflen += neededpaddingbytes;

#if DEBUG_AUDIOSTREAM
    SDL_Log("AUDIOSTREAM: Putting %d bytes of preconverted audio, need %d byte work buffer\n", buflen, workbuflen);
#endif

    workbuf = EnsureStreamBufferSize(stream, workbuflen);
    if (!workbuf) {
        return -1; /* probably out of memory. */
    }

    resamplebuf = workbuf; /* default if not resampling. */

    SDL_memcpy(workbuf + paddingbytes, buf, buflen);

    if (stream->cvt_before_resampling.needed) {
        stream->cvt_before_resampling.buf = workbuf + paddingbytes;
        stream->cvt_before_resampling.len = buflen;
        if (SDL_ConvertAudio(&stream->cvt_before_resampling) == -1) {
            return -1; /* uhoh! */
        }
        buflen = stream->cvt_before_resampling.len_cvt;

#if DEBUG_AUDIOSTREAM
        SDL_Log("AUDIOSTREAM: After initial conversion we have %d bytes\n", buflen);
#endif
    }

    if (stream->dst_rate != stream->src_rate) {
        /* save off some samples at the end; they are used for padding now so
           the resampler is coherent and then used at the start of the next
           put operation. Prepend last put operation's padding, too. */

        /* prepend prior put's padding. :P */
        if (paddingbytes) {
            SDL_memcpy(workbuf, stream->resampler_padding, paddingbytes);
            buflen += paddingbytes;
        }

        /* save off the data at the end for the next run. */
        SDL_memcpy(stream->resampler_padding, workbuf + (buflen - neededpaddingbytes), neededpaddingbytes);

        resamplebuf = workbuf + buflen; /* skip to second piece of workbuf. */
        SDL_assert(buflen >= neededpaddingbytes);
        if (buflen > neededpaddingbytes) {
            buflen = stream->resampler_func(stream, workbuf, buflen - neededpaddingbytes, resamplebuf, resamplebuflen);
        } else {
            buflen = 0;
        }

#if DEBUG_AUDIOSTREAM
        SDL_Log("AUDIOSTREAM: After resampling we have %d bytes\n", buflen);
#endif
    }

    if (stream->cvt_after_resampling.needed && (buflen > 0)) {
        stream->cvt_after_resampling.buf = resamplebuf;
        stream->cvt_after_resampling.len = buflen;
        if (SDL_ConvertAudio(&stream->cvt_after_resampling) == -1) {
            return -1; /* uhoh! */
        }
        buflen = stream->cvt_after_resampling.len_cvt;

#if DEBUG_AUDIOSTREAM
        SDL_Log("AUDIOSTREAM: After final conversion we have %d bytes\n", buflen);
#endif
    }

#if DEBUG_AUDIOSTREAM
    SDL_Log("AUDIOSTREAM: Final output is %d bytes\n", buflen);
#endif

    if (maxputbytes) {
        const int maxbytes = *maxputbytes;
        if (buflen > maxbytes) {
            buflen = maxbytes;
        }
        *maxputbytes -= buflen;
    }

    /* resamplebuf holds the final output, even if we didn't resample. */
    return buflen ? SDL_WriteToDataQueue(stream->queue, resamplebuf, buflen) : 0;
}

int SDL_AudioStreamPut(SDL_AudioStream *stream, const void *buf, int len)
{
    /* !!! FIXME: several converters can take advantage of SIMD, but only
       !!! FIXME:  if the data is aligned to 16 bytes. EnsureStreamBufferSize()
       !!! FIXME:  guarantees the buffer will align, but the
       !!! FIXME:  converters will iterate over the data backwards if
       !!! FIXME:  the output grows, and this means we won't align if buflen
       !!! FIXME:  isn't a multiple of 16. In these cases, we should chop off
       !!! FIXME:  a few samples at the end and convert them separately. */

#if DEBUG_AUDIOSTREAM
    SDL_Log("AUDIOSTREAM: wants to put %d preconverted bytes\n", buflen);
#endif

    if (!stream) {
        return SDL_InvalidParamError("stream");
    }
    if (!buf) {
        return SDL_InvalidParamError("buf");
    }
    if (len == 0) {
        return 0; /* nothing to do. */
    }
    if ((len % stream->src_sample_frame_size) != 0) {
        return SDL_SetError("Can't add partial sample frames");
    }

    if (!stream->cvt_before_resampling.needed &&
        (stream->dst_rate == stream->src_rate) &&
        !stream->cvt_after_resampling.needed) {
#if DEBUG_AUDIOSTREAM
        SDL_Log("AUDIOSTREAM: no conversion needed at all, queueing %d bytes.\n", len);
#endif
        return SDL_WriteToDataQueue(stream->queue, buf, len);
    }

    while (len > 0) {
        int amount;

        /* If we don't have a staging buffer or we're given enough data that
           we don't need to store it for later, skip the staging process.
         */
        if (!stream->staging_buffer_filled && len >= stream->staging_buffer_size) {
            return SDL_AudioStreamPutInternal(stream, buf, len, NULL);
        }

        /* If there's not enough data to fill the staging buffer, just save it */
        if ((stream->staging_buffer_filled + len) < stream->staging_buffer_size) {
            SDL_memcpy(stream->staging_buffer + stream->staging_buffer_filled, buf, len);
            stream->staging_buffer_filled += len;
            return 0;
        }

        /* Fill the staging buffer, process it, and continue */
        amount = (stream->staging_buffer_size - stream->staging_buffer_filled);
        SDL_assert(amount > 0);
        SDL_memcpy(stream->staging_buffer + stream->staging_buffer_filled, buf, amount);
        stream->staging_buffer_filled = 0;
        if (SDL_AudioStreamPutInternal(stream, stream->staging_buffer, stream->staging_buffer_size, NULL) < 0) {
            return -1;
        }
        buf = (void *)((Uint8 *)buf + amount);
        len -= amount;
    }
    return 0;
}

int SDL_AudioStreamFlush(SDL_AudioStream *stream)
{
    if (!stream) {
        return SDL_InvalidParamError("stream");
    }

#if DEBUG_AUDIOSTREAM
    SDL_Log("AUDIOSTREAM: flushing! staging_buffer_filled=%d bytes\n", stream->staging_buffer_filled);
#endif

    /* shouldn't use a staging buffer if we're not resampling. */
    SDL_assert((stream->dst_rate != stream->src_rate) || (stream->staging_buffer_filled == 0));

    if (stream->staging_buffer_filled > 0) {
        /* push the staging buffer + silence. We need to flush out not just
           the staging buffer, but the piece that the stream was saving off
           for right-side resampler padding. */
        const SDL_bool first_run = stream->first_run;
        const int filled = stream->staging_buffer_filled;
        int actual_input_frames = filled / stream->src_sample_frame_size;
        if (!first_run) {
            actual_input_frames += stream->resampler_padding_samples / stream->pre_resample_channels;
        }

        if (actual_input_frames > 0) { /* don't bother if nothing to flush. */
            /* This is how many bytes we're expecting without silence appended. */
            int flush_remaining = ((int)SDL_ceil(actual_input_frames * stream->rate_incr)) * stream->dst_sample_frame_size;

#if DEBUG_AUDIOSTREAM
            SDL_Log("AUDIOSTREAM: flushing with padding to get max %d bytes!\n", flush_remaining);
#endif

            SDL_memset(stream->staging_buffer + filled, '\0', stream->staging_buffer_size - filled);
            if (SDL_AudioStreamPutInternal(stream, stream->staging_buffer, stream->staging_buffer_size, &flush_remaining) < 0) {
                return -1;
            }

            /* we have flushed out (or initially filled) the pending right-side
               resampler padding, but we need to push more silence to guarantee
               the staging buffer is fully flushed out, too. */
            SDL_memset(stream->staging_buffer, '\0', filled);
            if (SDL_AudioStreamPutInternal(stream, stream->staging_buffer, stream->staging_buffer_size, &flush_remaining) < 0) {
                return -1;
            }
        }
    }

    stream->staging_buffer_filled = 0;
    stream->first_run = SDL_TRUE;

    return 0;
}

/* get converted/resampled data from the stream */
int SDL_AudioStreamGet(SDL_AudioStream *stream, void *buf, int len)
{
#if DEBUG_AUDIOSTREAM
    SDL_Log("AUDIOSTREAM: want to get %d converted bytes\n", len);
#endif

    if (!stream) {
        return SDL_InvalidParamError("stream");
    }
    if (!buf) {
        return SDL_InvalidParamError("buf");
    }
    if (len <= 0) {
        return 0; /* nothing to do. */
    }
    if ((len % stream->dst_sample_frame_size) != 0) {
        return SDL_SetError("Can't request partial sample frames");
    }

    return (int)SDL_ReadFromDataQueue(stream->queue, buf, len);
}

/* number of converted/resampled bytes available */
int SDL_AudioStreamAvailable(SDL_AudioStream *stream)
{
    return stream ? (int)SDL_CountDataQueue(stream->queue) : 0;
}

void SDL_AudioStreamClear(SDL_AudioStream *stream)
{
    if (!stream) {
        SDL_InvalidParamError("stream");
    } else {
        SDL_ClearDataQueue(stream->queue, (size_t)stream->packetlen * 2);
        if (stream->reset_resampler_func) {
            stream->reset_resampler_func(stream);
        }
        stream->first_run = SDL_TRUE;
        stream->staging_buffer_filled = 0;
    }
}

/* dispose of a stream */
void SDL_FreeAudioStream(SDL_AudioStream *stream)
{
    if (stream) {
        if (stream->cleanup_resampler_func) {
            stream->cleanup_resampler_func(stream);
        }
        SDL_FreeDataQueue(stream->queue);
        SDL_free(stream->staging_buffer);
        SDL_free(stream->work_buffer_base);
        SDL_free(stream->resampler_padding);
        SDL_free(stream);
    }
}

/* vi: set ts=4 sw=4 expandtab: */
