#include <ruby.h>
#include "ruby_whisper.h"

extern VALUE cVADParams;

static size_t
ruby_whisper_vad_params_memsize(const void *p)
{
  const struct ruby_whisper_vad_params *params = p;
  size_t size = sizeof(params);
  if (!params) {
    return 0;
  }
  return size;
}

static const rb_data_type_t ruby_whisper_vad_params_type = {
  "ruby_whisper_vad_params",
  {0, 0, ruby_whisper_vad_params_memsize,},
  0, 0,
  0
};

static VALUE
ruby_whisper_vad_params_s_allocate(VALUE klass)
{
  ruby_whisper_vad_params *rwvp;
  VALUE obj = TypedData_Make_Struct(klass, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  rwvp->params = whisper_vad_default_params();
  return obj;
}

/*
 * call-seq:
 *   threshold = th -> th
 */
static VALUE
ruby_whisper_vad_params_set_threshold(VALUE self, VALUE value)
{
  ruby_whisper_vad_params *rwvp;
  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  rwvp->params.threshold = RFLOAT_VALUE(value);
  return value;
}

static VALUE
ruby_whisper_vad_params_get_threshold(VALUE self)
{
  ruby_whisper_vad_params *rwvp;
  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  return DBL2NUM(rwvp->params.threshold);
}

void
init_ruby_whisper_vad_params(VALUE *mWhisper)
{
  cVADParams = rb_define_class_under(*mWhisper, "VADParams", rb_cObject);
  rb_define_alloc_func(cVADParams, ruby_whisper_vad_params_s_allocate);

  rb_define_method(cVADParams, "threshold=", ruby_whisper_vad_params_set_threashold, 1);
  rb_define_method(cVADParams, "threshold", ruby_whisper_vad_params_get_threashold, 0);
}
