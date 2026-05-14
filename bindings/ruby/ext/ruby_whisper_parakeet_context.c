#include "ruby_whisper.h"

extern ID id_to_s;

extern VALUE ruby_whisper_normalize_model_path(VALUE model_path);
extern VALUE ruby_whisper_parakeet_transcribe(VALUE self, VALUE audio_path, VALUE params);

static void
ruby_whisper_parakeet_context_free(void *p)
{
  ruby_whisper_parakeet_context *rwpc = (ruby_whisper_parakeet_context *)p;
  if (rwpc->context) {
    parakeet_free(rwpc->context);
    rwpc->context = NULL;
  }
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
  rwpc->context = parakeet_init_from_file_with_params(StringValueCStr(model_path), parakeet_context_default_params());
  if (rwpc->context == NULL) {
    rb_raise(rb_eRuntimeError, "Failed to load model");
  }

  return Qnil;
}

void
init_ruby_whisper_parakeet_context(VALUE *mParakeet)
{
  VALUE cParakeetContext = rb_define_class_under(*mParakeet, "Context", rb_cObject);

  rb_define_alloc_func(cParakeetContext, ruby_whisper_parakeet_context_allocate);

  rb_define_method(cParakeetContext, "initialize", ruby_whisper_parakeet_context_initialize, -1);
  rb_define_method(cParakeetContext, "transcribe", ruby_whisper_parakeet_transcribe, 2);
}
