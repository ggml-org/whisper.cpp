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
 * Probability threshold to consider as speech.
 *
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

/*
 * Min duration for a valid speech segment.
 *
 * call-seq:
 *   min_speech_duration_ms = duration_ms -> duration_ms
 */
static VALUE
ruby_whisper_vad_params_set_min_speech_duration_ms(VALUE self, VALUE value)
{
  ruby_whisper_vad_params *rwvp;
  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  rwvp->params.min_speech_duration_ms = NUM2INT(value);
  return value;
}

static VALUE
ruby_whisper_vad_params_get_min_speech_duration_ms(VALUE self)
{
  ruby_whisper_vad_params *rwvp;
  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  return INT2NUM(rwvp->params.min_speech_duration_ms);
}

/*
 * Min silence duration to consider speech as ended.
 *
 * call-seq:
 *   min_silence_duration_ms = duration_ms -> duration_ms
 */
static VALUE
ruby_whisper_vad_params_set_min_silence_duration_ms(VALUE self, VALUE value)
{
  ruby_whisper_vad_params *rwvp;
  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  rwvp->params.min_silence_duration_ms = NUM2INT(value);
  return value;
}

static VALUE
ruby_whisper_vad_params_get_min_silence_duration_ms(VALUE self)
{
  ruby_whisper_vad_params *rwvp;
  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  return INT2NUM(rwvp->params.min_silence_duration_ms);
}

/*
 * Max duration of a speech segment before forcing a new segment.
 *
 * call-seq:
 *   max_speech_duration_s = duration_s -> duration_s
 */
static VALUE
ruby_whisper_vad_params_set_max_speech_duration_s(VALUE self, VALUE value)
{
  ruby_whisper_vad_params *rwvp;
  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  rwvp->params.max_speech_duration_s = RFLOAT_VALUE(value);
  return value;
}

static VALUE
ruby_whisper_vad_params_get_max_speech_duration_s(VALUE self)
{
  ruby_whisper_vad_params *rwvp;
  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  return DBL2NUM(rwvp->params.max_speech_duration_s);
}

/*
 * Padding added before and after speech segments.
 *
 * call-seq:
 *   speech_pad_ms = pad_ms -> pad_ms
 */
static VALUE
ruby_whisper_vad_params_set_speech_pad_ms(VALUE self, VALUE value)
{
  ruby_whisper_vad_params *rwvp;
  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  rwvp->params.speech_pad_ms = NUM2INT(value);
  return value;
}

static VALUE
ruby_whisper_vad_params_get_speech_pad_ms(VALUE self)
{
  ruby_whisper_vad_params *rwvp;
  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  return INT2NUM(rwvp->params.speech_pad_ms);
}

/*
 * Overlap in seconds when copying audio samples from speech segment.
 *
 * call-seq:
 *   samples_overlap = overlap -> overlap
 */
static VALUE
ruby_whisper_vad_params_set_samples_overlap(VALUE self, VALUE value)
{
  ruby_whisper_vad_params *rwvp;
  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  rwvp->params.samples_overlap = RFLOAT_VALUE(value);
  return value;
}

static VALUE
ruby_whisper_vad_params_get_samples_overlap(VALUE self)
{
  ruby_whisper_vad_params *rwvp;
  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);
  return DBL2NUM(rwvp->params.samples_overlap);
}

void
init_ruby_whisper_vad_params(VALUE *mVAD)
{
  cVADParams = rb_define_class_under(*mVAD, "Params", rb_cObject);
  rb_define_alloc_func(cVADParams, ruby_whisper_vad_params_s_allocate);

  rb_define_method(cVADParams, "threshold=", ruby_whisper_vad_params_set_threashold, 1);
  rb_define_method(cVADParams, "threshold", ruby_whisper_vad_params_get_threashold, 0);
  rb_define_method(cVADParams, "min_speech_duration_ms=", ruby_whisper_vad_params_set_min_speech_duration_ms, 1);
  rb_define_method(cVADParams, "min_speech_duration_ms", ruby_whisper_vad_params_get_min_speech_duration_ms, 0);
  rb_define_method(cVADParams, "min_silence_duration_ms=", ruby_whisper_vad_params_set_min_silence_duration_ms, 1);
  rb_define_method(cVADParams, "min_silence_duration_ms", ruby_whisper_vad_params_get_min_silence_duration_ms, 0);
  rb_define_method(cVADParams, "max_speech_duration_s=", ruby_whisper_vad_params_set_max_speech_duration_s, 1);
  rb_define_method(cVADParams, "max_speech_duration_s", ruby_whisper_vad_params_get_max_speech_duration_s, 0);
  rb_define_method(cVADParams, "speech_pad_ms=", ruby_whisper_vad_params_set_speech_pad_ms, 1);
  rb_define_method(cVADParams, "speech_pad_ms", ruby_whisper_vad_params_get_speech_pad_ms, 0);
  rb_define_method(cVADParams, "samples_overlap=", ruby_whisper_vad_params_set_samples_overlap, 1);
  rb_define_method(cVADParams, "samples_overlap", ruby_whisper_vad_params_get_samples_overlap, 0);
}
