#ifndef RUBY_WHISPER_H
#define RUBY_WHISPER_H

#include <ruby.h>
#include <ruby/version.h>
#include <ruby/util.h>
#include <ruby/thread.h>
#include <ruby/thread_native.h>
#include <ruby/atomic.h>
#include <ruby/memory_view.h>
#include "whisper.h"
#include "parakeet.h"
#include "ruby_whisper_log_settable.h"

#if RUBY_API_VERSION_MAJOR < 4
// Exists but not declared as public API
int ruby_thread_has_gvl_p(void);
#endif

typedef struct {
  VALUE *context;
  VALUE user_data;
  VALUE callback;
  VALUE callbacks;
} ruby_whisper_callback_container;

typedef struct ruby_whisper_abort_callback_user_data {
  volatile rb_atomic_t is_interrupted;
  ruby_whisper_callback_container *callback_container;
} ruby_whisper_abort_callback_user_data;

typedef struct ruby_whisper_log {
  enum ggml_log_level level;
  char *text;
  size_t length;
  size_t capacity;
} ruby_whisper_log;

typedef struct ruby_whisper_log_queue {
  rb_nativethread_lock_t lock;
  rb_nativethread_cond_t cond;
  bool is_open;

  size_t head;
  size_t tail;
  size_t size;
  ruby_whisper_log *logs;
} ruby_whisper_log_queue;

typedef struct {
  struct whisper_context *context;
} ruby_whisper;

typedef struct ruby_whisper_context_params {
  struct whisper_context_params params;
} ruby_whisper_context_params;

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
  VALUE text;
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

typedef struct parsed_samples_t {
  float *samples;
  int n_samples;
  rb_memory_view_t memview;
  bool memview_exported;
} parsed_samples_t;

typedef struct {
  VALUE *context;
  VALUE *params;
  float *samples;
  int n_samples;
} ruby_whisper_full_args;

typedef struct ruby_whisper_full_parallel_args {
  VALUE *context;
  VALUE *params;
  float *samples;
  int n_samples;
  int n_processors;
} ruby_whisper_full_parallel_args;

typedef struct {
  struct parakeet_full_params params;
  ruby_whisper_callback_container *new_segment_callback_container;
  ruby_whisper_callback_container *new_token_callback_container;
  ruby_whisper_callback_container *progress_callback_container;
  ruby_whisper_callback_container *encoder_begin_callback_container;
  ruby_whisper_callback_container *abort_callback_container;
} ruby_whisper_parakeet_params;

typedef struct {
  struct parakeet_context_params params;
} ruby_whisper_parakeet_context_params;

typedef struct {
  struct parakeet_context *context;
} ruby_whisper_parakeet_context;

typedef struct {
  VALUE context;
  int index;
} ruby_whisper_parakeet_segment;

typedef struct {
  parakeet_token_data *token_data;
  VALUE text;
} ruby_whisper_parakeet_token;

typedef struct {
  VALUE context;
} ruby_whisper_parakeet_model;

extern ID id_extended;
extern ID id_log_callback_thread;
extern ID id_start_log_callback_thread;
extern ID id_alive_p;
extern ID id_join;
extern void ruby_whisper_log_queue_initialize(ruby_whisper_log_queue *log_queue);
extern void ruby_whisper_log_queue_open(ruby_whisper_log_queue *log_queue);
extern void ruby_whisper_log_queue_close(ruby_whisper_log_queue *log_queue);
extern void ruby_whisper_log_queue_enqueue(ruby_whisper_log_queue *log_queue, enum ggml_log_level level, const char *text);
extern VALUE ruby_whisper_log_queue_drain(ruby_whisper_log_queue *log_queue);

#define GetContext(obj, rw) do { \
  TypedData_Get_Struct((obj), ruby_whisper, &ruby_whisper_type, (rw)); \
  if ((rw)->context == NULL) { \
    rb_raise(rb_eRuntimeError, "Not initialized"); \
  } \
} while (0)

#define GetContextParams(obj, rwcp) do { \
  TypedData_Get_Struct((obj), ruby_whisper_context_params, &ruby_whisper_context_params_type, (rwcp)); \
} while (0)

#define GetToken(obj, rwt) do { \
  TypedData_Get_Struct((obj), ruby_whisper_token, &ruby_whisper_token_type, (rwt)); \
  if ((rwt)->token_data == NULL) { \
    rb_raise(rb_eRuntimeError, "Not initialized"); \
  } \
} while (0)

#define GetVADContext(obj, rwvc) do { \
    TypedData_Get_Struct((obj), ruby_whisper_vad_context, &ruby_whisper_vad_context_type, (rwvc)); \
    if ((rwvc)->context == NULL) { \
      rb_raise(rb_eRuntimeError, "Not initialized"); \
    } \
} while (0)

#define GetVADParams(obj, rwvp) do { \
  TypedData_Get_Struct((obj), ruby_whisper_vad_params, &ruby_whisper_vad_params_type, (rwvp)); \
} while (0)

#define GetVADSegments(obj, rwvss) do { \
  TypedData_Get_Struct((obj), ruby_whisper_vad_segments, &ruby_whisper_vad_segments_type, (rwvss)); \
  if ((rwvss)->segments == NULL) { \
    rb_raise(rb_eRuntimeError, "Not initialized"); \
  } \
} while (0)

#define GetParakeetContextParams(obj, rwpcp) do { \
  TypedData_Get_Struct((obj), ruby_whisper_parakeet_context_params, &ruby_whisper_parakeet_context_params_type, (rwpcp)); \
} while (0)

#define GetParakeetContext(obj, rwpc) do { \
  TypedData_Get_Struct((obj), ruby_whisper_parakeet_context, &ruby_whisper_parakeet_context_type, (rwpc)); \
  if ((rwpc)->context == NULL) { \
    rb_raise(rb_eRuntimeError, "Not initialized"); \
  } \
} while (0)

#define GetParakeetParams(obj, rwpp) do { \
  TypedData_Get_Struct((obj), ruby_whisper_parakeet_params, &ruby_whisper_parakeet_params_type, (rwpp)); \
  if (!(rwpp)->new_segment_callback_container || \
      !(rwpp)->new_token_callback_container || \
      !(rwpp)->progress_callback_container || \
      !(rwpp)->encoder_begin_callback_container || \
      !(rwpp)->abort_callback_container) { \
    rb_raise(rb_eRuntimeError, "Not initialized"); \
  } \
} while (0)

#define GetParakeetSegment(obj, rwps) do { \
  TypedData_Get_Struct((obj), ruby_whisper_parakeet_segment, &ruby_whisper_parakeet_segment_type, (rwps)); \
  if (!(rwps)->context) { \
    rb_raise(rb_eRuntimeError, "Not initialized"); \
  } \
} while (0)

#define GetParakeetToken(obj, rwpt) do { \
  TypedData_Get_Struct((obj), ruby_whisper_parakeet_token, &ruby_whisper_parakeet_token_type, (rwpt)); \
  if (!(rwpt)->token_data) { \
    rb_raise(rb_eRuntimeError, "Not initialized"); \
  } \
} while (0)

#define GetParakeetModel(obj, rwpm) do { \
  TypedData_Get_Struct((obj), ruby_whisper_parakeet_model, &ruby_whisper_parakeet_model_type, (rwpm)); \
  if (NIL_P((rwpm)->context)) { \
    rb_raise(rb_eRuntimeError, "Not initialized"); \
  } \
} while (0)

#endif
