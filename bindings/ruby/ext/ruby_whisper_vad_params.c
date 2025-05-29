#include <ruby.h>
#include "ruby_whisper.h"

#define DEFINE_PARAM(param_name, nth) \
  id_ ## param_name = rb_intern(#param_name); \
  param_names[nth] = id_ ## param_name; \
  rb_define_method(cVADParams, #param_name, ruby_whisper_vad_params_get_ ## param_name, 0); \
  rb_define_method(cVADParams, #param_name "=", ruby_whisper_vad_params_set_ ## param_name, 1);

#define NUM_PARAMS 6

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

static ID param_names[NUM_PARAMS];
static ID id_threshold;
static ID id_min_speech_duration_ms;
static ID id_min_silence_duration_ms;
static ID id_max_speech_duration_s;
static ID id_speech_pad_ms;
static ID id_samples_overlap;

const rb_data_type_t ruby_whisper_vad_params_type = {
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

static VALUE
ruby_whisper_vad_params_equal(VALUE self, VALUE other)
{
  ruby_whisper_vad_params *rwvp1;
  ruby_whisper_vad_params *rwvp2;

  if (self == other) {
    return Qtrue;
  }

  if (!rb_obj_is_kind_of(other, cVADParams)) {
    return Qfalse;
  }

  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp1);
  TypedData_Get_Struct(other, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp2);

  if (rwvp1->params.threshold != rwvp2->params.threshold) {
    return Qfalse;
  }
  if (rwvp1->params.min_speech_duration_ms != rwvp2->params.min_speech_duration_ms) {
    return Qfalse;
  }
  if (rwvp1->params.min_silence_duration_ms != rwvp2->params.min_silence_duration_ms) {
    return Qfalse;
  }
  if (rwvp1->params.max_speech_duration_s != rwvp2->params.max_speech_duration_s) {
    return Qfalse;
  }
  if (rwvp1->params.speech_pad_ms != rwvp2->params.speech_pad_ms) {
    return Qfalse;
  }
  if (rwvp1->params.samples_overlap != rwvp2->params.samples_overlap) {
    return Qfalse;
  }

  return Qtrue;
}

#define SET_PARAM_IF_SAME(param_name) \
  if (id == id_ ## param_name) { \
    ruby_whisper_vad_params_set_ ## param_name(self, value); \
    continue; \
  }

VALUE
ruby_whisper_vad_params_initialize(int argc, VALUE *argv, VALUE self)
{
  VALUE kw_hash;
  VALUE values[NUM_PARAMS] = {Qundef};
  VALUE value;
  ruby_whisper_vad_params *rwvp;
  ID id;
  int i;

  TypedData_Get_Struct(self, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);

  rb_scan_args_kw(RB_SCAN_ARGS_KEYWORDS, argc, argv, ":", &kw_hash);
  if (NIL_P(kw_hash)) {
    return self;
  }

  rb_get_kwargs(kw_hash, param_names, 0, NUM_PARAMS, values);

  for (i = 0; i < NUM_PARAMS; i++) {
    id = param_names[i];
    value = values[i];
    if (value == Qundef) {
      continue;
    }
    SET_PARAM_IF_SAME(threshold)
    SET_PARAM_IF_SAME(min_speech_duration_ms)
    SET_PARAM_IF_SAME(min_silence_duration_ms)
    SET_PARAM_IF_SAME(max_speech_duration_s)
    SET_PARAM_IF_SAME(speech_pad_ms)
    SET_PARAM_IF_SAME(samples_overlap)
  }

  return self;
}

#undef SET_PARAM_IF_SAME

void
init_ruby_whisper_vad_params(VALUE *mVAD)
{
  cVADParams = rb_define_class_under(*mVAD, "Params", rb_cObject);
  rb_define_alloc_func(cVADParams, ruby_whisper_vad_params_s_allocate);
  rb_define_method(cVADParams, "initialize", ruby_whisper_vad_params_initialize, -1);

  DEFINE_PARAM(threshold, 0)
  DEFINE_PARAM(min_speech_duration_ms, 1)
  DEFINE_PARAM(min_silence_duration_ms, 2)
  DEFINE_PARAM(max_speech_duration_s, 3)
  DEFINE_PARAM(speech_pad_ms, 4)
  DEFINE_PARAM(samples_overlap, 5)

  rb_define_method(cVADParams, "==", ruby_whisper_vad_params_equal, 1);
}

#undef DEFINE_PARAM
#undef NUM_PARAMS
