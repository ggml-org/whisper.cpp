#include <ruby.h>
#include <ruby/memory_view.h>
#include "ruby_whisper.h"

extern ID id_to_s;
extern ID id___method__;
extern ID id_to_enum;
extern ID id_length;
extern ID id_next;
extern ID id_new;
extern ID id_to_path;
extern ID id_URI;
extern ID id_pre_converted_models;
extern ID id_coreml_compiled_models;
extern ID id_cache;
extern ID id_n_processors;

extern VALUE cContext;
extern VALUE eError;
extern VALUE cModel;

extern const rb_data_type_t ruby_whisper_params_type;
extern VALUE ruby_whisper_transcribe(int argc, VALUE *argv, VALUE self);
extern VALUE rb_whisper_model_s_new(VALUE context);
extern VALUE rb_whisper_segment_s_new(VALUE context, int index);
extern void prepare_transcription(ruby_whisper_params *rwp, VALUE *context);

ID transcribe_option_names[1];

static void
ruby_whisper_free(ruby_whisper *rw)
{
  if (rw->context) {
    whisper_free(rw->context);
    rw->context = NULL;
  }
}

void
rb_whisper_mark(ruby_whisper *rw)
{
  // call rb_gc_mark on any ruby references in rw
}

void
rb_whisper_free(void *p)
{
  ruby_whisper *rw = (ruby_whisper *)p;
  ruby_whisper_free(rw);
  free(rw);
}

static size_t
ruby_whisper_memsize(const void *p)
{
  const ruby_whisper *rw = (const ruby_whisper *)p;
  size_t size = sizeof(rw);
  if (!rw) {
    return 0;
  }
  if (rw->context) {
    size += sizeof(rw->context);
  }
  return size;
}

const rb_data_type_t ruby_whisper_type = {
  "ruby_whisper",
  {0, rb_whisper_free, ruby_whisper_memsize,},
  0, 0,
  0
};

static VALUE
ruby_whisper_allocate(VALUE klass)
{
  ruby_whisper *rw;
  VALUE obj = TypedData_Make_Struct(klass, ruby_whisper, &ruby_whisper_type, rw);
  rw->context = NULL;
  return obj;
}

VALUE
ruby_whisper_normalize_model_path(VALUE model_path)
{
  VALUE pre_converted_models = rb_funcall(cModel, id_pre_converted_models, 0);
  VALUE pre_converted_model = rb_hash_aref(pre_converted_models, model_path);
  if (!NIL_P(pre_converted_model)) {
    model_path = pre_converted_model;
#ifdef RUBY_WHISPER_USE_COREML
    VALUE coreml_converted_models = rb_funcall(cModel, id_coreml_compiled_models, 0);
    VALUE coreml_converted_model = rb_hash_aref(coreml_converted_models, pre_converted_model);
    if (!NIL_P(coreml_converted_model)) {
      rb_funcall(coreml_converted_model, id_cache, 0);
    }
#endif
  }
  else if (TYPE(model_path) == T_STRING) {
    const char * model_path_str = StringValueCStr(model_path);
    if (strncmp("http://", model_path_str, 7) == 0 || strncmp("https://", model_path_str, 8) == 0) {
      VALUE uri_class = rb_const_get(cModel, id_URI);
      model_path = rb_class_new_instance(1, &model_path, uri_class);
    }
  }
  else if (rb_obj_is_kind_of(model_path, rb_path2class("URI::HTTP"))) {
    VALUE uri_class = rb_const_get(cModel, id_URI);
    model_path = rb_class_new_instance(1, &model_path, uri_class);
  }
  if (rb_respond_to(model_path, id_to_path)) {
    model_path = rb_funcall(model_path, id_to_path, 0);
  }

  return model_path;
}

/*
 * call-seq:
 *   new("base.en") -> Whisper::Context
 *   new("path/to/model.bin") -> Whisper::Context
 *   new(Whisper::Model::URI.new("https://example.net/uri/of/model.bin")) -> Whisper::Context
 */
static VALUE
ruby_whisper_initialize(int argc, VALUE *argv, VALUE self)
{
  ruby_whisper *rw;
  VALUE whisper_model_file_path;

  // TODO: we can support init from buffer here too maybe another ruby object to expose
  rb_scan_args(argc, argv, "01", &whisper_model_file_path);
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);

  whisper_model_file_path = ruby_whisper_normalize_model_path(whisper_model_file_path);
  if (!rb_respond_to(whisper_model_file_path, id_to_s)) {
    rb_raise(rb_eRuntimeError, "Expected file path to model to initialize Whisper::Context");
  }
  rw->context = whisper_init_from_file_with_params(StringValueCStr(whisper_model_file_path), whisper_context_default_params());
  if (rw->context == NULL) {
    rb_raise(rb_eRuntimeError, "error: failed to initialize whisper context");
  }
  return self;
}

/*
 * call-seq:
 *   model_n_vocab -> Integer
 */
VALUE ruby_whisper_model_n_vocab(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_vocab(rw->context));
}

/*
 * call-seq:
 *   model_n_audio_ctx -> Integer
 */
VALUE ruby_whisper_model_n_audio_ctx(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_audio_ctx(rw->context));
}

/*
 * call-seq:
 *   model_n_audio_state -> Integer
 */
VALUE ruby_whisper_model_n_audio_state(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_audio_state(rw->context));
}

/*
 * call-seq:
 *   model_n_audio_head -> Integer
 */
VALUE ruby_whisper_model_n_audio_head(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_audio_head(rw->context));
}

/*
 * call-seq:
 *   model_n_audio_layer -> Integer
 */
VALUE ruby_whisper_model_n_audio_layer(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_audio_layer(rw->context));
}

/*
 * call-seq:
 *   model_n_text_ctx -> Integer
 */
VALUE ruby_whisper_model_n_text_ctx(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_text_ctx(rw->context));
}

/*
 * call-seq:
 *   model_n_text_state -> Integer
 */
VALUE ruby_whisper_model_n_text_state(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_text_state(rw->context));
}

/*
 * call-seq:
 *   model_n_text_head -> Integer
 */
VALUE ruby_whisper_model_n_text_head(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_text_head(rw->context));
}

/*
 * call-seq:
 *   model_n_text_layer -> Integer
 */
VALUE ruby_whisper_model_n_text_layer(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_text_layer(rw->context));
}

/*
 * call-seq:
 *   model_n_mels -> Integer
 */
VALUE ruby_whisper_model_n_mels(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_mels(rw->context));
}

/*
 * call-seq:
 *   model_ftype -> Integer
 */
VALUE ruby_whisper_model_ftype(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_ftype(rw->context));
}

/*
 * call-seq:
 *   model_type -> String
 */
VALUE ruby_whisper_model_type(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return rb_str_new2(whisper_model_type_readable(rw->context));
}

/*
 * Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
 * Not thread safe for same context
 * Uses the specified decoding strategy to obtain the text.
 *
 * call-seq:
 *   full(params, samples, n_samples) -> nil
 *   full(params, samples) -> nil
 *
 * The second argument +samples+ must be an array of samples, respond to :length, or be a MemoryView of an array of float. It must be 32 bit float PCM audio data.
 */
VALUE ruby_whisper_full(int argc, VALUE *argv, VALUE self)
{
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }

  ruby_whisper *rw;
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  VALUE params = argv[0];
  TypedData_Get_Struct(params, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  VALUE samples = argv[1];
  int n_samples;
  rb_memory_view_t view;
  const bool memory_view_available_p = rb_memory_view_available_p(samples);
  if (argc == 3) {
    n_samples = NUM2INT(argv[2]);
    if (TYPE(samples) == T_ARRAY) {
      if (RARRAY_LEN(samples) < n_samples) {
        rb_raise(rb_eArgError, "samples length %ld is less than n_samples %d", RARRAY_LEN(samples), n_samples);
      }
    }
    // Should check when samples.respond_to?(:length)?
  } else {
    if (TYPE(samples) == T_ARRAY) {
      if (RARRAY_LEN(samples) > INT_MAX) {
        rb_raise(rb_eArgError, "samples are too long");
      }
      n_samples = (int)RARRAY_LEN(samples);
    } else if (memory_view_available_p) {
      if (!rb_memory_view_get(samples, &view, RUBY_MEMORY_VIEW_SIMPLE)) {
        view.obj = Qnil;
        rb_raise(rb_eArgError, "unable to get a memory view");
      }
      ssize_t n_samples_size = view.byte_size / view.item_size;
      if (n_samples_size > INT_MAX) {
        rb_raise(rb_eArgError, "samples are too long");
      }
      n_samples = (int)n_samples_size;
    } else if (rb_respond_to(samples, id_length)) {
      n_samples = NUM2INT(rb_funcall(samples, id_length, 0));
    } else {
      rb_raise(rb_eArgError, "samples must respond to :length or be a MemoryView of an array of flaot when n_samples is not given");
    }
  }
  float * c_samples = (float *)malloc(n_samples * sizeof(float));
  if (memory_view_available_p)  {
    c_samples = (float *)view.data;
  } else {
    if (TYPE(samples) == T_ARRAY) {
      for (int i = 0; i < n_samples; i++) {
        c_samples[i] = RFLOAT_VALUE(rb_ary_entry(samples, i));
      }
    } else {
      // TODO: use rb_block_call
      VALUE iter = rb_funcall(samples, id_to_enum, 1, rb_str_new2("each"));
      for (int i = 0; i < n_samples; i++) {
        // TODO: check if iter is exhausted and raise ArgumentError appropriately
        VALUE sample = rb_funcall(iter, id_next, 0);
        c_samples[i] = RFLOAT_VALUE(sample);
      }
    }
  }
  prepare_transcription(rwp, &self);
  const int result = whisper_full(rw->context, rwp->params, c_samples, n_samples);
  if (0 == result) {
    return self;
  } else {
    rb_exc_raise(rb_funcall(eError, id_new, 1, result));
  }
}

/*
 * Split the input audio in chunks and process each chunk separately using whisper_full_with_state()
 * Result is stored in the default state of the context
 * Not thread safe if executed in parallel on the same context.
 * It seems this approach can offer some speedup in some cases.
 * However, the transcription accuracy can be worse at the beginning and end of each chunk.
 *
 * call-seq:
 *   full_parallel(params, samples) -> nil
 *   full_parallel(params, samples, n_samples) -> nil
 *   full_parallel(params, samples, n_samples, n_processors) -> nil
 *   full_parallel(params, samples, nil, n_processors) -> nil
 */
static VALUE
ruby_whisper_full_parallel(int argc, VALUE *argv,VALUE self)
{
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }

  ruby_whisper *rw;
  ruby_whisper_params *rwp;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  VALUE params = argv[0];
  TypedData_Get_Struct(params, ruby_whisper_params, &ruby_whisper_params_type, rwp);
  VALUE samples = argv[1];
  int n_samples;
  int n_processors;
  rb_memory_view_t view;
  const bool memory_view_available_p = rb_memory_view_available_p(samples);
  switch (argc) {
  case 2:
    n_processors = 1;
    break;
  case 3:
    n_processors = 1;
    break;
  case 4:
    n_processors = NUM2INT(argv[3]);
    break;
  }
  if (argc >= 3 && !NIL_P(argv[2])) {
    n_samples = NUM2INT(argv[2]);
    if (TYPE(samples) == T_ARRAY) {
      if (RARRAY_LEN(samples) < n_samples) {
        rb_raise(rb_eArgError, "samples length %ld is less than n_samples %d", RARRAY_LEN(samples), n_samples);
      }
    }
    // Should check when samples.respond_to?(:length)?
  } else if (memory_view_available_p) {
    if (!rb_memory_view_get(samples, &view, RUBY_MEMORY_VIEW_SIMPLE)) {
      view.obj = Qnil;
      rb_raise(rb_eArgError, "unable to get a memory view");
    }
    ssize_t n_samples_size = view.byte_size / view.item_size;
    if (n_samples_size > INT_MAX) {
      rb_raise(rb_eArgError, "samples are too long");
    }
    n_samples = (int)n_samples_size;
  } else {
    if (TYPE(samples) == T_ARRAY) {
      if (RARRAY_LEN(samples) > INT_MAX) {
        rb_raise(rb_eArgError, "samples are too long");
      }
      n_samples = (int)RARRAY_LEN(samples);
    } else if (rb_respond_to(samples, id_length)) {
      n_samples = NUM2INT(rb_funcall(samples, id_length, 0));
    } else {
      rb_raise(rb_eArgError, "samples must respond to :length or be a MemoryView of an array of flaot when n_samples is not given");
    }
  }
  float * c_samples = (float *)malloc(n_samples * sizeof(float));
  if (memory_view_available_p) {
    c_samples = (float *)view.data;
  } else {
    if (TYPE(samples) == T_ARRAY) {
      for (int i = 0; i < n_samples; i++) {
        c_samples[i] = RFLOAT_VALUE(rb_ary_entry(samples, i));
      }
    } else {
      // FIXME: use rb_block_call
      VALUE iter = rb_funcall(samples, id_to_enum, 1, rb_str_new2("each"));
      for (int i = 0; i < n_samples; i++) {
        // TODO: check if iter is exhausted and raise ArgumentError
        VALUE sample = rb_funcall(iter, id_next, 0);
        c_samples[i] = RFLOAT_VALUE(sample);
      }
    }
  }
  prepare_transcription(rwp, &self);
  const int result = whisper_full_parallel(rw->context, rwp->params, c_samples, n_samples, n_processors);
  if (0 == result) {
    return self;
  } else {
    rb_exc_raise(rb_funcall(eError, id_new, 1, result));
  }
}

/*
 * Number of segments.
 *
 * call-seq:
 *   full_n_segments -> Integer
 */
static VALUE
ruby_whisper_full_n_segments(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_full_n_segments(rw->context));
}

/*
 * Language ID, which can be converted to string by Whisper.lang_str and Whisper.lang_str_full.
 *
 * call-seq:
 *   full_lang_id -> Integer
 */
static VALUE
ruby_whisper_full_lang_id(VALUE self)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_full_lang_id(rw->context));
}

static int ruby_whisper_full_check_segment_index(const ruby_whisper * rw, const VALUE i_segment)
{
  const int c_i_segment = NUM2INT(i_segment);
  if (c_i_segment < 0 || c_i_segment >= whisper_full_n_segments(rw->context)) {
    rb_raise(rb_eIndexError, "segment index %d out of range", c_i_segment);
  }
  return c_i_segment;
}

/*
 * Start time of a segment indexed by +segment_index+ in centiseconds (10 times milliseconds).
 *
 *   full_get_segment_t0(3) # => 1668 (16680 ms)
 *
 * call-seq:
 *   full_get_segment_t0(segment_index) -> Integer
 */
static VALUE
ruby_whisper_full_get_segment_t0(VALUE self, VALUE i_segment)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  const int c_i_segment = ruby_whisper_full_check_segment_index(rw, i_segment);
  const int64_t t0 = whisper_full_get_segment_t0(rw->context, c_i_segment);
  return LONG2NUM(t0);
}

/*
 * End time of a segment indexed by +segment_index+ in centiseconds (10 times milliseconds).
 *
 *   full_get_segment_t1(3) # => 1668 (16680 ms)
 *
 * call-seq:
 *   full_get_segment_t1(segment_index) -> Integer
 */
static VALUE
ruby_whisper_full_get_segment_t1(VALUE self, VALUE i_segment)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  const int c_i_segment = ruby_whisper_full_check_segment_index(rw, i_segment);
  const int64_t t1 = whisper_full_get_segment_t1(rw->context, c_i_segment);
  return LONG2NUM(t1);
}

/*
 * Whether the next segment indexed by +segment_index+ is predicated as a speaker turn.
 *
 *   full_get_segment_speacker_turn_next(3) # => true
 *
 * call-seq:
 *   full_get_segment_speacker_turn_next(segment_index) -> bool
 */
static VALUE
ruby_whisper_full_get_segment_speaker_turn_next(VALUE self, VALUE i_segment)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  const int c_i_segment = ruby_whisper_full_check_segment_index(rw, i_segment);
  const bool speaker_turn_next = whisper_full_get_segment_speaker_turn_next(rw->context, c_i_segment);
  return speaker_turn_next ? Qtrue : Qfalse;
}

/*
 * Text of a segment indexed by +segment_index+.
 *
 *   full_get_segment_text(3) # => "ask not what your country can do for you, ..."
 *
 * call-seq:
 *   full_get_segment_text(segment_index) -> String
 */
static VALUE
ruby_whisper_full_get_segment_text(VALUE self, VALUE i_segment)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  const int c_i_segment = ruby_whisper_full_check_segment_index(rw, i_segment);
  const char * text = whisper_full_get_segment_text(rw->context, c_i_segment);
  return rb_str_new2(text);
}

/*
 * call-seq:
 *   full_get_segment_no_speech_prob(segment_index) -> Float
 */
static VALUE
ruby_whisper_full_get_segment_no_speech_prob(VALUE self, VALUE i_segment)
{
  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  const int c_i_segment = ruby_whisper_full_check_segment_index(rw, i_segment);
  const float no_speech_prob = whisper_full_get_segment_no_speech_prob(rw->context, c_i_segment);
  return DBL2NUM(no_speech_prob);
}

// High level API

static VALUE
ruby_whisper_full_get_segment(VALUE self, VALUE i_segment)
{
  return rb_whisper_segment_s_new(self, NUM2INT(i_segment));
}

/*
 * Yields each Whisper::Segment:
 *
 *   whisper.transcribe("path/to/audio.wav", params)
 *   whisper.each_segment do |segment|
 *     puts segment.text
 *   end
 *
 * Returns an Enumerator if no block given:
 *
 *   whisper.transcribe("path/to/audio.wav", params)
 *   enum = whisper.each_segment
 *   enum.to_a # => [#<Whisper::Segment>, ...]
 *
 * call-seq:
 *   each_segment {|segment| ... }
 *   each_segment -> Enumerator
 */
static VALUE
ruby_whisper_each_segment(VALUE self)
{
  if (!rb_block_given_p()) {
    const VALUE method_name = rb_funcall(self, id___method__, 0);
    return rb_funcall(self, id_to_enum, 1, method_name);
  }

  ruby_whisper *rw;
  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);

  const int n_segments = whisper_full_n_segments(rw->context);
  for (int i = 0; i < n_segments; ++i) {
    rb_yield(rb_whisper_segment_s_new(self, i));
  }

  return self;
}

/*
 * call-seq:
 *   model -> Whisper::Model
 */
static VALUE
ruby_whisper_get_model(VALUE self)
{
  return rb_whisper_model_s_new(self);
}

void
init_ruby_whisper_context(VALUE *mWhisper)
{
  cContext = rb_define_class_under(*mWhisper, "Context", rb_cObject);

  transcribe_option_names[0] = id_n_processors;

  rb_define_alloc_func(cContext, ruby_whisper_allocate);
  rb_define_method(cContext, "initialize", ruby_whisper_initialize, -1);

  rb_define_method(cContext, "transcribe", ruby_whisper_transcribe, -1);
  rb_define_method(cContext, "model_n_vocab", ruby_whisper_model_n_vocab, 0);
  rb_define_method(cContext, "model_n_audio_ctx", ruby_whisper_model_n_audio_ctx, 0);
  rb_define_method(cContext, "model_n_audio_state", ruby_whisper_model_n_audio_state, 0);
  rb_define_method(cContext, "model_n_audio_head", ruby_whisper_model_n_audio_head, 0);
  rb_define_method(cContext, "model_n_audio_layer", ruby_whisper_model_n_audio_layer, 0);
  rb_define_method(cContext, "model_n_text_ctx", ruby_whisper_model_n_text_ctx, 0);
  rb_define_method(cContext, "model_n_text_state", ruby_whisper_model_n_text_state, 0);
  rb_define_method(cContext, "model_n_text_head", ruby_whisper_model_n_text_head, 0);
  rb_define_method(cContext, "model_n_text_layer", ruby_whisper_model_n_text_layer, 0);
  rb_define_method(cContext, "model_n_mels", ruby_whisper_model_n_mels, 0);
  rb_define_method(cContext, "model_ftype", ruby_whisper_model_ftype, 0);
  rb_define_method(cContext, "model_type", ruby_whisper_model_type, 0);
  rb_define_method(cContext, "full_n_segments", ruby_whisper_full_n_segments, 0);
  rb_define_method(cContext, "full_lang_id", ruby_whisper_full_lang_id, 0);
  rb_define_method(cContext, "full_get_segment_t0", ruby_whisper_full_get_segment_t0, 1);
  rb_define_method(cContext, "full_get_segment_t1", ruby_whisper_full_get_segment_t1, 1);
  rb_define_method(cContext, "full_get_segment_speaker_turn_next", ruby_whisper_full_get_segment_speaker_turn_next, 1);
  rb_define_method(cContext, "full_get_segment_text", ruby_whisper_full_get_segment_text, 1);
  rb_define_method(cContext, "full_get_segment_no_speech_prob", ruby_whisper_full_get_segment_no_speech_prob, 1);
  rb_define_method(cContext, "full", ruby_whisper_full, -1);
  rb_define_method(cContext, "full_parallel", ruby_whisper_full_parallel, -1);

  // High level
  rb_define_method(cContext, "full_get_segment", ruby_whisper_full_get_segment, 1);
  rb_define_method(cContext, "each_segment", ruby_whisper_each_segment, 0);

  rb_define_method(cContext, "model", ruby_whisper_get_model, 0);
}
