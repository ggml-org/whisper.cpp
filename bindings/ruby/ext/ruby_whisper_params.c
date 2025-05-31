#include <ruby.h>
#include "ruby_whisper.h"

#define BOOL_PARAMS_SETTER(self, prop, value) \
  ruby_whisper_params *rwp; \
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp); \
  if (value == Qfalse || value == Qnil) { \
    rwp->params.prop = false; \
  } else { \
    rwp->params.prop = true; \
  } \
  return value; \

#define BOOL_PARAMS_GETTER(self,  prop) \
  ruby_whisper_params *rwp; \
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp); \
  if (rwp->params.prop) { \
    return Qtrue; \
  } else { \
    return Qfalse; \
  }

#define DEFINE_PARAM(param_name, nth) \
  id_ ## param_name = rb_intern(#param_name); \
  param_names[nth] = id_ ## param_name; \
  rb_define_method(cParams, #param_name, ruby_whisper_params_get_ ## param_name, 0); \
  rb_define_method(cParams, #param_name "=", ruby_whisper_params_set_ ## param_name, 1);

#define RUBY_WHISPER_PARAMS_PARAM_NAMES_COUNT 35

extern VALUE cParams;
extern VALUE cVADParams;

extern ID id_call;

extern VALUE ruby_whisper_normalize_model_path(VALUE model_path);
extern VALUE rb_whisper_segment_s_new(VALUE context, int index);
extern const rb_data_type_t ruby_whisper_vad_params_type;

static ID param_names[RUBY_WHISPER_PARAMS_PARAM_NAMES_COUNT];
static ID id_language;
static ID id_translate;
static ID id_no_context;
static ID id_single_segment;
static ID id_print_special;
static ID id_print_progress;
static ID id_print_realtime;
static ID id_print_timestamps;
static ID id_suppress_blank;
static ID id_suppress_nst;
static ID id_token_timestamps;
static ID id_split_on_word;
static ID id_initial_prompt;
static ID id_diarize;
static ID id_offset;
static ID id_duration;
static ID id_max_text_tokens;
static ID id_temperature;
static ID id_max_initial_ts;
static ID id_length_penalty;
static ID id_temperature_inc;
static ID id_entropy_thold;
static ID id_logprob_thold;
static ID id_no_speech_thold;
static ID id_new_segment_callback;
static ID id_new_segment_callback_user_data;
static ID id_progress_callback;
static ID id_progress_callback_user_data;
static ID id_encoder_begin_callback;
static ID id_encoder_begin_callback_user_data;
static ID id_abort_callback;
static ID id_abort_callback_user_data;
static ID id_vad;
static ID id_vad_model_path;
static ID id_vad_params;

static void
rb_whisper_callbcack_container_mark(ruby_whisper_callback_container *rwc)
{
  if (rwc == NULL) return;

  rb_gc_mark(rwc->user_data);
  rb_gc_mark(rwc->callback);
  rb_gc_mark(rwc->callbacks);
}

static ruby_whisper_callback_container*
rb_whisper_callback_container_allocate() {
  ruby_whisper_callback_container *container;
  container = ALLOC(ruby_whisper_callback_container);
  container->context = NULL;
  container->user_data = Qnil;
  container->callback = Qnil;
  container->callbacks = rb_ary_new();
  return container;
}

static void new_segment_callback(struct whisper_context *ctx, struct whisper_state *state, int n_new, void *user_data) {
  const ruby_whisper_callback_container *container = (ruby_whisper_callback_container *)user_data;

  // Currently, doesn't support state because
  // those require to resolve GC-related problems.
  if (!NIL_P(container->callback)) {
    rb_funcall(container->callback, id_call, 4, *container->context, Qnil, INT2NUM(n_new), container->user_data);
  }
  const long callbacks_len = RARRAY_LEN(container->callbacks);
  if (0 == callbacks_len) {
    return;
  }
  const int n_segments = whisper_full_n_segments_from_state(state);
  for (int i = n_new; i > 0; i--) {
    int i_segment = n_segments - i;
    VALUE segment = rb_whisper_segment_s_new(*container->context, i_segment);
    for (int j = 0; j < callbacks_len; j++) {
      VALUE cb = rb_ary_entry(container->callbacks, j);
      rb_funcall(cb, id_call, 1, segment);
    }
  }
}

static void progress_callback(struct whisper_context *ctx, struct whisper_state *state, int progress_cur, void *user_data) {
  const ruby_whisper_callback_container *container = (ruby_whisper_callback_container *)user_data;
  const VALUE progress = INT2NUM(progress_cur);
  // Currently, doesn't support state because
  // those require to resolve GC-related problems.
  if (!NIL_P(container->callback)) {
    rb_funcall(container->callback, id_call, 4, *container->context, Qnil, progress, container->user_data);
  }
  const long callbacks_len = RARRAY_LEN(container->callbacks);
  if (0 == callbacks_len) {
    return;
  }
  for (int j = 0; j < callbacks_len; j++) {
    VALUE cb = rb_ary_entry(container->callbacks, j);
    rb_funcall(cb, id_call, 1, progress);
  }
}

static bool encoder_begin_callback(struct whisper_context *ctx, struct whisper_state *state, void *user_data) {
  const ruby_whisper_callback_container *container = (ruby_whisper_callback_container *)user_data;
  bool is_aborted = false;
  VALUE result;

  // Currently, doesn't support state because
  // those require to resolve GC-related problems.
  if (!NIL_P(container->callback)) {
    result = rb_funcall(container->callback, id_call, 3, *container->context, Qnil, container->user_data);
    if (result == Qfalse) {
      is_aborted = true;
    }
  }
  const long callbacks_len = RARRAY_LEN(container->callbacks);
  if (0 == callbacks_len) {
    return !is_aborted;
  }
  for (int j = 0; j < callbacks_len; j++) {
    VALUE cb = rb_ary_entry(container->callbacks, j);
    result = rb_funcall(cb, id_call, 0);
    if (result == Qfalse) {
      is_aborted = true;
    }
  }
  return !is_aborted;
}

static bool abort_callback(void * user_data) {
  const ruby_whisper_callback_container *container = (ruby_whisper_callback_container *)user_data;
  if (!NIL_P(container->callback)) {
    VALUE result = rb_funcall(container->callback, id_call, 1, container->user_data);
    if (!NIL_P(result) && Qfalse != result) {
      return true;
    }
  }
  const long callbacks_len = RARRAY_LEN(container->callbacks);
  if (0 == callbacks_len) {
    return false;
  }
  for (int j = 0; j < callbacks_len; j++) {
    VALUE cb = rb_ary_entry(container->callbacks, j);
    VALUE result = rb_funcall(cb, id_call, 1, container->user_data);
    if (!NIL_P(result) && Qfalse != result) {
      return true;
    }
  }
  return false;
}

static void register_callbacks(ruby_whisper_params * rwp, VALUE * context) {
  if (!NIL_P(rwp->new_segment_callback_container->callback) || 0 != RARRAY_LEN(rwp->new_segment_callback_container->callbacks)) {
    rwp->new_segment_callback_container->context = context;
    rwp->params.new_segment_callback = new_segment_callback;
    rwp->params.new_segment_callback_user_data = rwp->new_segment_callback_container;
  }

  if (!NIL_P(rwp->progress_callback_container->callback) || 0 != RARRAY_LEN(rwp->progress_callback_container->callbacks)) {
    rwp->progress_callback_container->context = context;
    rwp->params.progress_callback = progress_callback;
    rwp->params.progress_callback_user_data = rwp->progress_callback_container;
  }

  if (!NIL_P(rwp->encoder_begin_callback_container->callback) || 0 != RARRAY_LEN(rwp->encoder_begin_callback_container->callbacks)) {
    rwp->encoder_begin_callback_container->context = context;
    rwp->params.encoder_begin_callback = encoder_begin_callback;
    rwp->params.encoder_begin_callback_user_data = rwp->encoder_begin_callback_container;
  }

  if (!NIL_P(rwp->abort_callback_container->callback) || 0 != RARRAY_LEN(rwp->abort_callback_container->callbacks)) {
    rwp->abort_callback_container->context = context;
    rwp->params.abort_callback = abort_callback;
    rwp->params.abort_callback_user_data = rwp->abort_callback_container;
  }
}

static void set_vad_params(ruby_whisper_params *rwp)
{
  ruby_whisper_vad_params * rwvp;
  TypedData_Get_Struct(rwp->vad_params, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  rwp->params.vad_params = rwvp->params;
}

void
prepare_transcription(ruby_whisper_params *rwp, VALUE *context)
{
  register_callbacks(rwp, context);
  set_vad_params(rwp);
}

void
rb_whisper_params_mark(void *p)
{
  ruby_whisper_params *rwp = (ruby_whisper_params *)p;
  rb_whisper_callbcack_container_mark(rwp->new_segment_callback_container);
  rb_whisper_callbcack_container_mark(rwp->progress_callback_container);
  rb_whisper_callbcack_container_mark(rwp->encoder_begin_callback_container);
  rb_whisper_callbcack_container_mark(rwp->abort_callback_container);
  rb_gc_mark(rwp->vad_params);
}

void
ruby_whisper_params_free(ruby_whisper_params *rwp)
{
}

void
rb_whisper_params_free(void *p)
{
  ruby_whisper_params *rwp = (ruby_whisper_params *)p;
  // How to free user_data and callback only when not referred to by others?
  ruby_whisper_params_free(rwp);
  free(rwp);
}

static size_t
ruby_whisper_params_memsize(const void *p)
{
  const ruby_whisper_params *rwp = (const ruby_whisper_params *)p;

  return sizeof(ruby_whisper_params) + sizeof(rwp->params) + sizeof(rwp->vad_params);
}

const rb_data_type_t ruby_whisper_params_type = {
  "ruby_whisper_params",
  {
    rb_whisper_params_mark,
    rb_whisper_params_free,
    ruby_whisper_params_memsize,
  },
  0, 0,
  0
};

static VALUE
ruby_whisper_params_allocate(VALUE klass)
{
  ruby_whisper_params *rwp;
  VALUE obj = TypedData_Make_Struct(klass, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
  rwp->diarize = false;
  rwp->vad_params = TypedData_Wrap_Struct(cVADParams, &ruby_whisper_vad_params_type, (void *)&rwp->params.vad_params);
  rwp->new_segment_callback_container = rb_whisper_callback_container_allocate();
  rwp->progress_callback_container = rb_whisper_callback_container_allocate();
  rwp->encoder_begin_callback_container = rb_whisper_callback_container_allocate();
  rwp->abort_callback_container = rb_whisper_callback_container_allocate();
  return obj;
}

/*
 * params.language = "auto" | "en", etc...
 *
 * call-seq:
 *   language = lang_name -> lang_name
 */
static VALUE
ruby_whisper_params_set_language(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  if (value == Qfalse || value == Qnil) {
    rwp->params.language = "auto";
  } else {
    rwp->params.language = StringValueCStr(value);
  }
  return value;
}
/*
 * call-seq:
 *   language -> String
 */
static VALUE
ruby_whisper_params_get_language(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  if (rwp->params.language) {
    return rb_str_new2(rwp->params.language);
  } else {
    return rb_str_new2("auto");
  }
}
/*
 * call-seq:
 *   translate = do_translate -> do_translate
 */
static VALUE
ruby_whisper_params_set_translate(VALUE self, VALUE value)
{
  BOOL_PARAMS_SETTER(self, translate, value)
}
/*
 * call-seq:
 *   translate -> bool
 */
static VALUE
ruby_whisper_params_get_translate(VALUE self)
{
  BOOL_PARAMS_GETTER(self, translate)
}
/*
 * call-seq:
 *   no_context = dont_use_context -> dont_use_context
 */
static VALUE
ruby_whisper_params_set_no_context(VALUE self, VALUE value)
{
  BOOL_PARAMS_SETTER(self, no_context, value)
}
/*
 * If true, does not use past transcription (if any) as initial prompt for the decoder.
 *
 * call-seq:
 *   no_context -> bool
 */
static VALUE
ruby_whisper_params_get_no_context(VALUE self)
{
  BOOL_PARAMS_GETTER(self, no_context)
}
/*
 * call-seq:
 *   single_segment = force_single -> force_single
 */
static VALUE
ruby_whisper_params_set_single_segment(VALUE self, VALUE value)
{
  BOOL_PARAMS_SETTER(self, single_segment, value)
}
/*
 * If true, forces single segment output (useful for streaming).
 *
 * call-seq:
 *   single_segment -> bool
 */
static VALUE
ruby_whisper_params_get_single_segment(VALUE self)
{
  BOOL_PARAMS_GETTER(self, single_segment)
}
/*
 * call-seq:
 *   print_special = force_print -> force_print
 */
static VALUE
ruby_whisper_params_set_print_special(VALUE self, VALUE value)
{
  BOOL_PARAMS_SETTER(self, print_special, value)
}
/*
 * If true, prints special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.).
 *
 * call-seq:
 *   print_special -> bool
 */
static VALUE
ruby_whisper_params_get_print_special(VALUE self)
{
  BOOL_PARAMS_GETTER(self, print_special)
}
/*
 * call-seq:
 *   print_progress = force_print -> force_print
 */
static VALUE
ruby_whisper_params_set_print_progress(VALUE self, VALUE value)
{
  BOOL_PARAMS_SETTER(self, print_progress, value)
}
/*
 * If true, prints progress information.
 *
 * call-seq:
 *   print_progress -> bool
 */
static VALUE
ruby_whisper_params_get_print_progress(VALUE self)
{
  BOOL_PARAMS_GETTER(self, print_progress)
}
/*
 * call-seq:
 *   print_realtime = force_print -> force_print
 */
static VALUE
ruby_whisper_params_set_print_realtime(VALUE self, VALUE value)
{
  BOOL_PARAMS_SETTER(self, print_realtime, value)
}
/*
 * If true, prints results from within whisper.cpp. (avoid it, use callback instead)
 * call-seq:
 *   print_realtime -> bool
 */
static VALUE
ruby_whisper_params_get_print_realtime(VALUE self)
{
  BOOL_PARAMS_GETTER(self, print_realtime)
}
/*
 * call-seq:
 *   print_timestamps = force_print -> force_print
 */
static VALUE
ruby_whisper_params_set_print_timestamps(VALUE self, VALUE value)
{
  BOOL_PARAMS_SETTER(self, print_timestamps, value)
}
/*
 * If true, prints timestamps for each text segment when printing realtime.
 *
 * call-seq:
 *   print_timestamps -> bool
 */
static VALUE
ruby_whisper_params_get_print_timestamps(VALUE self)
{
  BOOL_PARAMS_GETTER(self, print_timestamps)
}
/*
 * call-seq:
 *   suppress_blank = force_suppress -> force_suppress
 */
static VALUE
ruby_whisper_params_set_suppress_blank(VALUE self, VALUE value)
{
  BOOL_PARAMS_SETTER(self, suppress_blank, value)
}
/*
 * If true, suppresses blank outputs.
 *
 * call-seq:
 *   suppress_blank -> bool
 */
static VALUE
ruby_whisper_params_get_suppress_blank(VALUE self)
{
  BOOL_PARAMS_GETTER(self, suppress_blank)
}
/*
 * call-seq:
 *   suppress_nst = force_suppress -> force_suppress
 */
static VALUE
ruby_whisper_params_set_suppress_nst(VALUE self, VALUE value)
{
  BOOL_PARAMS_SETTER(self, suppress_nst, value)
}
/*
 * If true, suppresses non-speech-tokens.
 *
 * call-seq:
 *   suppress_nst -> bool
 */
static VALUE
ruby_whisper_params_get_suppress_nst(VALUE self)
{
  BOOL_PARAMS_GETTER(self, suppress_nst)
}
/*
 * If true, enables token-level timestamps.
 *
 * call-seq:
 *   token_timestamps -> bool
 */
static VALUE
ruby_whisper_params_get_token_timestamps(VALUE self)
{
  BOOL_PARAMS_GETTER(self, token_timestamps)
}
/*
 * call-seq:
 *   token_timestamps = force_timestamps -> force_timestamps
 */
static VALUE
ruby_whisper_params_set_token_timestamps(VALUE self, VALUE value)
{
  BOOL_PARAMS_SETTER(self, token_timestamps, value)
}
/*
 * If true, split on word rather than on token (when used with max_len).
 *
 * call-seq:
 *   translate -> bool
 */
static VALUE
ruby_whisper_params_get_split_on_word(VALUE self)
{
  BOOL_PARAMS_GETTER(self, split_on_word)
}
/*
 * call-seq:
 *   split_on_word = force_split -> force_split
 */
static VALUE
ruby_whisper_params_set_split_on_word(VALUE self, VALUE value)
{
  BOOL_PARAMS_SETTER(self, split_on_word, value)
}
/*
 * Tokens to provide to the whisper decoder as initial prompt
 * these are prepended to any existing text context from a previous call
 * use whisper_tokenize() to convert text to tokens.
 * Maximum of whisper_n_text_ctx()/2 tokens are used (typically 224).
 *
 * call-seq:
 *   initial_prompt -> String
 */
static VALUE
ruby_whisper_params_get_initial_prompt(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return rwp->params.initial_prompt == NULL ? Qnil : rb_str_new2(rwp->params.initial_prompt);
}
/*
 * call-seq:
 *   initial_prompt = prompt -> prompt
 */
static VALUE
ruby_whisper_params_set_initial_prompt(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->params.initial_prompt = StringValueCStr(value);
  return value;
}
/*
 * If true, enables diarization.
 *
 * call-seq:
 *   diarize -> bool
 */
static VALUE
ruby_whisper_params_get_diarize(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  if (rwp->diarize) {
    return Qtrue;
  } else {
    return Qfalse;
  }
}
/*
 * call-seq:
 *   diarize = force_diarize -> force_diarize
 */
static VALUE
ruby_whisper_params_set_diarize(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  if (value == Qfalse || value == Qnil) {
    rwp->diarize = false;
  } else {
    rwp->diarize = true;
  } \
  return value;
}

/*
 * Start offset in ms.
 *
 * call-seq:
 *   offset -> Integer
 */
static VALUE
ruby_whisper_params_get_offset(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return INT2NUM(rwp->params.offset_ms);
}
/*
 * call-seq:
 *   offset = offset_ms -> offset_ms
 */
static VALUE
ruby_whisper_params_set_offset(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->params.offset_ms = NUM2INT(value);
  return value;
}
/*
 * Audio duration to process in ms.
 *
 * call-seq:
 *   duration -> Integer
 */
static VALUE
ruby_whisper_params_get_duration(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return INT2NUM(rwp->params.duration_ms);
}
/*
 * call-seq:
 *   duration = duration_ms -> duration_ms
 */
static VALUE
ruby_whisper_params_set_duration(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->params.duration_ms = NUM2INT(value);
  return value;
}

/*
 * Max tokens to use from past text as prompt for the decoder.
 *
 * call-seq:
 *   max_text_tokens -> Integer
 */
static VALUE
ruby_whisper_params_get_max_text_tokens(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return INT2NUM(rwp->params.n_max_text_ctx);
}
/*
 * call-seq:
 *   max_text_tokens = n_tokens -> n_tokens
 */
static VALUE
ruby_whisper_params_set_max_text_tokens(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->params.n_max_text_ctx = NUM2INT(value);
  return value;
}
/*
 * call-seq:
 *   temperature -> Float
 */
static VALUE
ruby_whisper_params_get_temperature(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return DBL2NUM(rwp->params.temperature);
}
/*
 * call-seq:
 *   temperature = temp -> temp
 */
static VALUE
ruby_whisper_params_set_temperature(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->params.temperature = RFLOAT_VALUE(value);
  return value;
}
/*
 * See https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L97
 *
 * call-seq:
 *   max_initial_ts -> Flaot
 */
static VALUE
ruby_whisper_params_get_max_initial_ts(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return DBL2NUM(rwp->params.max_initial_ts);
}
/*
 * call-seq:
 *   max_initial_ts = timestamp -> timestamp
 */
static VALUE
ruby_whisper_params_set_max_initial_ts(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->params.max_initial_ts = RFLOAT_VALUE(value);
  return value;
}
/*
 * call-seq:
 *   length_penalty -> Float
 */
static VALUE
ruby_whisper_params_get_length_penalty(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return DBL2NUM(rwp->params.length_penalty);
}
/*
 * call-seq:
 *   length_penalty = penalty -> penalty
 */
static VALUE
ruby_whisper_params_set_length_penalty(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->params.length_penalty = RFLOAT_VALUE(value);
  return value;
}
/*
 * call-seq:
 *   temperature_inc -> Float
 */
static VALUE
ruby_whisper_params_get_temperature_inc(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return DBL2NUM(rwp->params.temperature_inc);
}
/*
 * call-seq:
 *   temperature_inc = inc -> inc
 */
static VALUE
ruby_whisper_params_set_temperature_inc(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->params.temperature_inc = RFLOAT_VALUE(value);
  return value;
}
/*
 * Similar to OpenAI's "compression_ratio_threshold"
 *
 * call-seq:
 *   entropy_thold -> Float
 */
static VALUE
ruby_whisper_params_get_entropy_thold(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return DBL2NUM(rwp->params.entropy_thold);
}
/*
 * call-seq:
 *   entropy_thold = threshold -> threshold
 */
static VALUE
ruby_whisper_params_set_entropy_thold(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->params.entropy_thold = RFLOAT_VALUE(value);
  return value;
}
/*
 * call-seq:
 *   logprob_thold -> Float
 */
static VALUE
ruby_whisper_params_get_logprob_thold(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return DBL2NUM(rwp->params.logprob_thold);
}
/*
 * call-seq:
 *   logprob_thold = threshold -> threshold
 */
static VALUE
ruby_whisper_params_set_logprob_thold(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->params.logprob_thold = RFLOAT_VALUE(value);
  return value;
}
/*
 * call-seq:
 *   no_speech_thold -> Float
 */
static VALUE
ruby_whisper_params_get_no_speech_thold(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return DBL2NUM(rwp->params.no_speech_thold);
}
/*
 * call-seq:
 *   no_speech_thold = threshold -> threshold
 */
static VALUE
ruby_whisper_params_set_no_speech_thold(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->params.no_speech_thold = RFLOAT_VALUE(value);
  return value;
}
static VALUE
ruby_whisper_params_get_new_segment_callback(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return rwp->new_segment_callback_container->callback;
}
/*
 * Sets new segment callback, called for every newly generated text segment.
 *
 *   params.new_segment_callback = ->(context, _, n_new, user_data) {
 *     # ...
 *   }
 *
 * call-seq:
 *   new_segment_callback = callback -> callback
 */
static VALUE
ruby_whisper_params_set_new_segment_callback(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->new_segment_callback_container->callback = value;
  return value;
}
static VALUE
ruby_whisper_params_get_new_segment_callback_user_data(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return rwp->new_segment_callback_container->user_data;
}
/*
 * Sets user data passed to the last argument of new segment callback.
 *
 * call-seq:
 *   new_segment_callback_user_data = user_data -> use_data
 */
static VALUE
ruby_whisper_params_set_new_segment_callback_user_data(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->new_segment_callback_container->user_data = value;
  return value;
}
static VALUE
ruby_whisper_params_get_progress_callback(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return rwp->progress_callback_container->callback;
}
/*
 * Sets progress callback, called on each progress update.
 *
 *   params.new_segment_callback = ->(context, _, progress, user_data) {
 *     # ...
 *   }
 *
 * +progress+ is an Integer between 0 and 100.
 *
 * call-seq:
 *   progress_callback = callback -> callback
 */
static VALUE
ruby_whisper_params_set_progress_callback(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->progress_callback_container->callback = value;
  return value;
}
static VALUE
ruby_whisper_params_get_progress_callback_user_data(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return rwp->progress_callback_container->user_data;
}
/*
 * Sets user data passed to the last argument of progress callback.
 *
 * call-seq:
 *   progress_callback_user_data = user_data -> use_data
 */
static VALUE
ruby_whisper_params_set_progress_callback_user_data(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->progress_callback_container->user_data = value;
  return value;
}

static VALUE
ruby_whisper_params_get_encoder_begin_callback(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return rwp->encoder_begin_callback_container->callback;
}

/*
 * Sets encoder begin callback, called when the encoder starts.
 *
 *   params.encoder_begin_callback = ->(context, _, user_data) {
 *     # ...
 *   }
 *
 * call-seq:
 *   encoder_begin_callback = callback -> callback
 */
static VALUE
ruby_whisper_params_set_encoder_begin_callback(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->encoder_begin_callback_container->callback = value;
  return value;
}

static VALUE
ruby_whisper_params_get_encoder_begin_callback_user_data(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return rwp->encoder_begin_callback_container->user_data;
}

/*
 * Sets user data passed to the last argument of encoder begin callback.
 *
 * call-seq:
 *   encoder_begin_callback_user_data = user_data -> use_data
 */
static VALUE
ruby_whisper_params_set_encoder_begin_callback_user_data(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->encoder_begin_callback_container->user_data = value;
  return value;
}

static VALUE
ruby_whisper_params_get_abort_callback(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return rwp->abort_callback_container->callback;
}
/*
 * Sets abort callback, called to check if the process should be aborted.
 *
 *   params.abort_callback = ->(user_data) {
 *     # ...
 *   }
 *
 * call-seq:
 *   abort_callback = callback -> callback
 */
static VALUE
ruby_whisper_params_set_abort_callback(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->abort_callback_container->callback = value;
  return value;
}
static VALUE
ruby_whisper_params_get_abort_callback_user_data(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return rwp->abort_callback_container->user_data;
}
/*
 * Sets user data passed to the last argument of abort callback.
 *
 * call-seq:
 *   abort_callback_user_data = user_data -> use_data
 */
static VALUE
ruby_whisper_params_set_abort_callback_user_data(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->abort_callback_container->user_data = value;
  return value;
}

/*
 * call-seq:
 *   vad = use_vad -> use_vad
 */
static VALUE
ruby_whisper_params_get_vad(VALUE self)
{
  BOOL_PARAMS_GETTER(self, vad)
}

static VALUE
ruby_whisper_params_set_vad(VALUE self, VALUE value)
{
  BOOL_PARAMS_SETTER(self, vad, value)
}

/*
 * call-seq:
 *   vad_model_path = model_path -> model_path
 */
static VALUE
ruby_whisper_params_set_vad_model_path(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  if (NIL_P(value)) {
    rwp->params.vad_model_path = NULL;
    return value;
  }
  VALUE path = ruby_whisper_normalize_model_path(value);
  rwp->params.vad_model_path = StringValueCStr(path);
  return value;
}

static VALUE
ruby_whisper_params_get_vad_model_path(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return rwp->params.vad_model_path == NULL ? Qnil : rb_str_new2(rwp->params.vad_model_path);
}

/*
 * call-seq:
 *   vad_params = params -> params
 */
static VALUE
ruby_whisper_params_set_vad_params(VALUE self, VALUE value)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  rwp->vad_params = value;
  return value;
}

static VALUE
ruby_whisper_params_get_vad_params(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  return rwp->vad_params;
}

#define SET_PARAM_IF_SAME(param_name) \
  if (id == id_ ## param_name) { \
    ruby_whisper_params_set_ ## param_name(self, value); \
    continue; \
  }

static VALUE
ruby_whisper_params_initialize(int argc, VALUE *argv, VALUE self)
{
  VALUE kw_hash;
  VALUE values[RUBY_WHISPER_PARAMS_PARAM_NAMES_COUNT] = {Qundef};
  VALUE value;
  ruby_whisper_params *rwp;
  ID id;
  int i;

  rb_scan_args_kw(RB_SCAN_ARGS_KEYWORDS, argc, argv, ":", &kw_hash);
  if (NIL_P(kw_hash)) {
    return self;
  }

  rb_get_kwargs(kw_hash, param_names, 0, RUBY_WHISPER_PARAMS_PARAM_NAMES_COUNT, values);
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);

  for (i = 0; i < RUBY_WHISPER_PARAMS_PARAM_NAMES_COUNT; i++) {
    id = param_names[i];
    value = values[i];
    if (value == Qundef) {
      continue;
    }
    if (id == id_diarize) {
      rwp->diarize = value;
      continue;
    } else {
      SET_PARAM_IF_SAME(language)
      SET_PARAM_IF_SAME(translate)
      SET_PARAM_IF_SAME(no_context)
      SET_PARAM_IF_SAME(single_segment)
      SET_PARAM_IF_SAME(print_special)
      SET_PARAM_IF_SAME(print_progress)
      SET_PARAM_IF_SAME(print_realtime)
      SET_PARAM_IF_SAME(print_timestamps)
      SET_PARAM_IF_SAME(suppress_blank)
      SET_PARAM_IF_SAME(suppress_nst)
      SET_PARAM_IF_SAME(token_timestamps)
      SET_PARAM_IF_SAME(split_on_word)
      SET_PARAM_IF_SAME(initial_prompt)
      SET_PARAM_IF_SAME(offset)
      SET_PARAM_IF_SAME(duration)
      SET_PARAM_IF_SAME(max_text_tokens)
      SET_PARAM_IF_SAME(temperature)
      SET_PARAM_IF_SAME(max_initial_ts)
      SET_PARAM_IF_SAME(length_penalty)
      SET_PARAM_IF_SAME(temperature_inc)
      SET_PARAM_IF_SAME(entropy_thold)
      SET_PARAM_IF_SAME(logprob_thold)
      SET_PARAM_IF_SAME(no_speech_thold)
      SET_PARAM_IF_SAME(new_segment_callback)
      SET_PARAM_IF_SAME(new_segment_callback_user_data)
      SET_PARAM_IF_SAME(progress_callback)
      SET_PARAM_IF_SAME(progress_callback_user_data)
      SET_PARAM_IF_SAME(encoder_begin_callback)
      SET_PARAM_IF_SAME(encoder_begin_callback_user_data)
      SET_PARAM_IF_SAME(abort_callback)
      SET_PARAM_IF_SAME(abort_callback_user_data)
      SET_PARAM_IF_SAME(vad)
      SET_PARAM_IF_SAME(vad_model_path)
      SET_PARAM_IF_SAME(vad_params)
    }
  }

  return self;
}

#undef SET_PARAM_IF_SAME

/*
 * Hook called on new segment. Yields each Whisper::Segment.
 *
 *   whisper.on_new_segment do |segment|
 *     # ...
 *   end
 *
 * call-seq:
 *   on_new_segment {|segment| ... }
 */
static VALUE
ruby_whisper_params_on_new_segment(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  const VALUE blk = rb_block_proc();
  rb_ary_push(rwp->new_segment_callback_container->callbacks, blk);
  return Qnil;
}

/*
 * Hook called on progress update. Yields each progress Integer between 0 and 100.
 *
 *   whisper.on_progress do |progress|
 *     # ...
 *   end
 *
 * call-seq:
 *   on_progress {|progress| ... }
 */
static VALUE
ruby_whisper_params_on_progress(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  const VALUE blk = rb_block_proc();
  rb_ary_push(rwp->progress_callback_container->callbacks, blk);
  return Qnil;
}

/*
 * Hook called when the encoder starts.
 *
 *   whisper.on_encoder_begin do
 *     # ...
 *   end
 *
 * call-seq:
 *   on_encoder_begin { ... }
 */
static VALUE
ruby_whisper_params_on_encoder_begin(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  const VALUE blk = rb_block_proc();
  rb_ary_push(rwp->encoder_begin_callback_container->callbacks, blk);
  return Qnil;
}

/*
 * Call block to determine whether abort or not. Return +true+ when you want to abort.
 *
 *   params.abort_on do
 *     if some_condition
 *       true # abort
 *     else
 *       false # continue
 *     end
 *   end
 *
 * call-seq:
 *   abort_on { ... }
 */
static VALUE
ruby_whisper_params_abort_on(VALUE self)
{
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  const VALUE blk = rb_block_proc();
  rb_ary_push(rwp->abort_callback_container->callbacks, blk);
  return Qnil;
}

void
init_ruby_whisper_params(VALUE *mWhisper)
{
  cParams  = rb_define_class_under(*mWhisper, "Params", rb_cObject);

  rb_define_alloc_func(cParams, ruby_whisper_params_allocate);
  rb_define_method(cParams, "initialize", ruby_whisper_params_initialize, -1);

  DEFINE_PARAM(language, 0)
  DEFINE_PARAM(translate, 1)
  DEFINE_PARAM(no_context, 2)
  DEFINE_PARAM(single_segment, 3)
  DEFINE_PARAM(print_special, 4)
  DEFINE_PARAM(print_progress, 5)
  DEFINE_PARAM(print_realtime, 6)
  DEFINE_PARAM(print_timestamps, 7)
  DEFINE_PARAM(suppress_blank, 8)
  DEFINE_PARAM(suppress_nst, 9)
  DEFINE_PARAM(token_timestamps, 10)
  DEFINE_PARAM(split_on_word, 11)
  DEFINE_PARAM(initial_prompt, 12)
  DEFINE_PARAM(diarize, 13)
  DEFINE_PARAM(offset, 14)
  DEFINE_PARAM(duration, 15)
  DEFINE_PARAM(max_text_tokens, 16)
  DEFINE_PARAM(temperature, 17)
  DEFINE_PARAM(max_initial_ts, 18)
  DEFINE_PARAM(length_penalty, 19)
  DEFINE_PARAM(temperature_inc, 20)
  DEFINE_PARAM(entropy_thold, 21)
  DEFINE_PARAM(logprob_thold, 22)
  DEFINE_PARAM(no_speech_thold, 23)
  DEFINE_PARAM(new_segment_callback, 24)
  DEFINE_PARAM(new_segment_callback_user_data, 25)
  DEFINE_PARAM(progress_callback, 26)
  DEFINE_PARAM(progress_callback_user_data, 27)
  DEFINE_PARAM(encoder_begin_callback, 28)
  DEFINE_PARAM(encoder_begin_callback_user_data, 29)
  DEFINE_PARAM(abort_callback, 30)
  DEFINE_PARAM(abort_callback_user_data, 31)
  DEFINE_PARAM(vad, 32)
  DEFINE_PARAM(vad_model_path, 33)
  DEFINE_PARAM(vad_params, 34)

  rb_define_method(cParams, "on_new_segment", ruby_whisper_params_on_new_segment, 0);
  rb_define_method(cParams, "on_progress", ruby_whisper_params_on_progress, 0);
  rb_define_method(cParams, "on_encoder_begin", ruby_whisper_params_on_encoder_begin, 0);
  rb_define_method(cParams, "abort_on", ruby_whisper_params_abort_on, 0);
}
