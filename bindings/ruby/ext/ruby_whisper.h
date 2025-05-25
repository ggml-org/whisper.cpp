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
  VALUE context;
} ruby_whisper_model;

#endif
