#include "ruby_whisper.h"

extern ID id_to_s;
extern ID id___method__;
extern ID id_to_enum;
extern ID id_new;

extern VALUE cParakeetContext;
extern VALUE eError;

extern VALUE ruby_whisper_normalize_model_path(VALUE model_path);
extern VALUE ruby_whisper_parakeet_transcribe(VALUE self, VALUE audio_path, VALUE params);
extern VALUE ruby_whisper_parakeet_segment_init(VALUE context, int index);
extern parsed_samples_t parse_samples(VALUE *samples, VALUE *n_samples);
extern VALUE release_samples(VALUE rb_parsed_args);
extern void ruby_whisper_parakeet_prepare_transcription(ruby_whisper_parakeet_params *rwpp, VALUE *context, ruby_whisper_parakeet_abort_callback_user_data *abort_callback_user_data);
extern rb_data_type_t ruby_whisper_parakeet_params_type;

static void
ruby_whisper_parakeet_context_free(void *p)
{
  ruby_whisper_parakeet_context *rwpc = (ruby_whisper_parakeet_context *)p;
  if (rwpc->context) {
    parakeet_free(rwpc->context);
    rwpc->context = NULL;
  }
  xfree(rwpc);
}

static size_t
ruby_whisper_parakeet_context_memsize(const void *p)
{
  ruby_whisper_parakeet_context *rwpc = (ruby_whisper_parakeet_context *)p;
  if (!rwpc) {
    return 0;
  }
  size_t size = sizeof(*rwpc);
  return size;
}

const rb_data_type_t ruby_whisper_parakeet_context_type = {
  "ruby_whisper_parakeet_context",
  {0, ruby_whisper_parakeet_context_free, ruby_whisper_parakeet_context_memsize,},
  0, 0,
  0
};

static VALUE
ruby_whisper_parakeet_context_allocate(VALUE klass)
{
  ruby_whisper_parakeet_context *rwpc;

  VALUE obj = TypedData_Make_Struct(klass, ruby_whisper_parakeet_context, &ruby_whisper_parakeet_context_type, rwpc);
  rwpc->context = NULL;

  return obj;
}

typedef struct {
  struct parakeet_context **context;
  char *model_path;
} ruby_whisper_parakeet_context_init_args;

static void*
ruby_whisper_parakeet_context_init_without_gvl(void *args)
{
  ruby_whisper_parakeet_context_init_args *init_args = (ruby_whisper_parakeet_context_init_args *)args;
  *init_args->context = parakeet_init_from_file_with_params(init_args->model_path, parakeet_context_default_params());
  return NULL;
}

static VALUE
ruby_whisper_parakeet_context_initialize(int argc, VALUE *argv, VALUE self)
{
  ruby_whisper_parakeet_context *rwpc;
  VALUE model_path;

  rb_scan_args(argc, argv, "1", &model_path);
  TypedData_Get_Struct(self, ruby_whisper_parakeet_context, &ruby_whisper_parakeet_context_type, rwpc);

  model_path = ruby_whisper_normalize_model_path(model_path);
  if (!rb_respond_to(model_path, id_to_s)) {
    rb_raise(rb_eRuntimeError, "Expected file path to model to initialize Parakeet::Context");
  }
  ruby_whisper_parakeet_context_init_args init_args = {
    &rwpc->context,
    StringValueCStr(model_path),
  };
  rb_thread_call_without_gvl(ruby_whisper_parakeet_context_init_without_gvl, (void *)&init_args, NULL, NULL);
  if (rwpc->context == NULL) {
    rb_raise(rb_eRuntimeError, "Failed to load model");
  }

  return Qnil;
}

static VALUE
ruby_whisper_parakeet_context_each_segment(VALUE self)
{
  if (!rb_block_given_p()) {
    const VALUE method_name = rb_funcall(self, id___method__, 0);
    return rb_funcall(self, id_to_enum, 1, method_name);
  }

  ruby_whisper_parakeet_context *rwpc;
  GetParakeetContext(self, rwpc);

  const int n_segments = parakeet_full_n_segments(rwpc->context);
  for (int i = 0; i < n_segments; ++i) {
    rb_yield(ruby_whisper_parakeet_segment_init(self, i));
  }

  return self;
}

typedef struct {
  struct parakeet_context *context;
  struct parakeet_full_params *params;
  float *samples;
  int n_samples;
  int result;
} parakeet_full_without_gvl_args;

static void*
parakeet_full_without_gvl(void *rb_args)
{
  parakeet_full_without_gvl_args *args = (parakeet_full_without_gvl_args *)rb_args;
  args->result = parakeet_full(args->context, *args->params, args->samples, args->n_samples);

  return NULL;
}

typedef struct {
  ruby_whisper_parakeet_abort_callback_user_data *abort_callback_user_data;
} parakeet_full_ubf_args;

static void
parakeet_full_ubf(void *rb_args)
{
  parakeet_full_ubf_args *args = (parakeet_full_ubf_args *)rb_args;

  RUBY_ATOMIC_SET(args->abort_callback_user_data->is_interrupted, 1);
}

typedef struct {
  VALUE *context;
  VALUE *params;
  float *samples;
  int n_samples;
} parakeet_full_args;

static VALUE
parakeet_full_body(VALUE rb_args)
{
  parakeet_full_args *args = (parakeet_full_args *)rb_args;
  ruby_whisper_parakeet_context *rwpc;
  GetParakeetContext(*args->context, rwpc);
  ruby_whisper_parakeet_params *rwpp;
  GetParakeetParams(*args->params, rwpp);

  ruby_whisper_parakeet_abort_callback_user_data abort_callback_user_data = {
    0,
    NULL,
  };
  ruby_whisper_parakeet_prepare_transcription(rwpp, args->context, &abort_callback_user_data);

  parakeet_full_without_gvl_args full_without_gvl_args = {
    rwpc->context,
    &rwpp->params,
    args->samples,
    args->n_samples,
    0
  };
  parakeet_full_ubf_args full_ubf_args = {
    &abort_callback_user_data,
  };
  rb_thread_call_without_gvl(parakeet_full_without_gvl, (void *)&full_without_gvl_args, parakeet_full_ubf, (void *)&full_ubf_args);

  return INT2NUM(full_without_gvl_args.result);
}

static VALUE
ruby_whisper_parakeet_context_full(int argc, VALUE *argv, VALUE self)
{
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }

  VALUE n_samples = argc == 2 ? Qnil : argv[2];

  struct parsed_samples_t parsed = parse_samples(&argv[1], &n_samples);
  parakeet_full_args args = {
    &self,
    &argv[0],
    parsed.samples,
    parsed.n_samples,
  };
  VALUE rb_result = rb_ensure(parakeet_full_body, (VALUE)&args, release_samples, (VALUE)&parsed);
  const int result = NUM2INT(rb_result);
  if (result == 0) {
    return self;
  } else {
    rb_exc_raise(rb_funcall(eError, id_new, 1, result));
  }
}

void
init_ruby_whisper_parakeet_context(VALUE *mParakeet)
{
  cParakeetContext = rb_define_class_under(*mParakeet, "Context", rb_cObject);

  rb_define_alloc_func(cParakeetContext, ruby_whisper_parakeet_context_allocate);

  rb_define_method(cParakeetContext, "initialize", ruby_whisper_parakeet_context_initialize, -1);
  rb_define_method(cParakeetContext, "transcribe", ruby_whisper_parakeet_transcribe, 2);
  rb_define_method(cParakeetContext, "each_segment", ruby_whisper_parakeet_context_each_segment, 0);
  rb_define_method(cParakeetContext, "full", ruby_whisper_parakeet_context_full, -1);
}
