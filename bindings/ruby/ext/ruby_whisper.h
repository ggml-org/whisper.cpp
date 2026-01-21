#ifndef RUBY_WHISPER_H
#define RUBY_WHISPER_H

#include "whisper.h"

typedef struct {
  VALUE *context;
  VALUE user_data;
  VALUE callback;
  VALUE callbacks;
} ruby_whisper_callback_container;

typedef struct {
  struct whisper_context *context;
} ruby_whisper;

typedef struct {
  struct whisper_full_params params;
  bool diarize;
  ruby_whisper_callback_container *new_segment_callback_container;
  ruby_whisper_callback_container *progress_callback_container;
  ruby_whisper_callback_container *encoder_begin_callback_container;
  ruby_whisper_callback_container *abort_callback_container;
  VALUE vad_params;
} ruby_whisper_params;

typedef struct {
  struct whisper_vad_params params;
} ruby_whisper_vad_params;

typedef struct {
  VALUE context;
  int index;
} ruby_whisper_segment;

typedef struct {
  whisper_token_data *token_data;
  const char *text;
} ruby_whisper_token;

typedef struct {
  VALUE context;
} ruby_whisper_model;

typedef struct {
  struct whisper_vad_segments *segments;
} ruby_whisper_vad_segments;

typedef struct {
  VALUE segments;
  int index;
} ruby_whisper_vad_segment;

typedef struct {
  struct whisper_vad_context *context;
} ruby_whisper_vad_context;

#define GetContext(obj, rw) do { \
  TypedData_Get_Struct((obj), ruby_whisper, &ruby_whisper_type, (rw)); \
  if ((rw)->context == NULL) { \
    rb_raise(rb_eRuntimeError, "Not initialized"); \
  } \
} while (0)

#define GetToken(obj, rwt) do {                                             \
  TypedData_Get_Struct((obj), ruby_whisper_token, &ruby_whisper_token_type, (rwt)); \
  if ((rwt)->token_data == NULL) { \
    rb_raise(rb_eRuntimeError, "Not initialized"); \
  } \
} while (0)

#define GetVADSegments(obj, rwvss) do { \
  TypedData_Get_Struct((obj), ruby_whisper_vad_segments, &ruby_whisper_vad_segments_type, (rwvss)); \
  if ((rwvss)->segments == NULL) { \
    rb_raise(rb_eRuntimeError, "Not initialized"); \
  } \
} while (0)

#endif
