#include "ruby_whisper.h"

#define NUM_PARAMS 6

#define DEF_BOOLEAN_ATTR_METHOD(name) \
static VALUE \
ruby_whisper_context_params_get_ ## name(VALUE self) { \
  ruby_whisper_context_params *rwcp; \
  GetContextParams(self, rwcp); \
  return rwcp->params.name ? Qtrue : Qfalse; \
} \
static VALUE \
ruby_whisper_context_params_set_ ## name(VALUE self, VALUE value) { \
  ruby_whisper_context_params *rwcp; \
  GetContextParams(self, rwcp); \
  rwcp->params.name = RTEST(value); \
  return value; \
}

#define DEF_INT_ATTR_METHOD(name) \
static VALUE \
ruby_whisper_context_params_get_ ## name(VALUE self) { \
  ruby_whisper_context_params *rwcp; \
  GetContextParams(self, rwcp); \
  return INT2NUM(rwcp->params.name); \
} \
static VALUE \
ruby_whisper_context_params_set_ ## name(VALUE self, VALUE value) { \
  ruby_whisper_context_params *rwcp; \
  GetContextParams(self, rwcp); \
  rwcp->params.name = NUM2INT(value); \
  return value; \
}

#define DEFINE_PARAM(param_name, nth) \
  id_ ## param_name = rb_intern(#param_name); \
  param_names[nth] = id_ ## param_name; \
  rb_define_method(cContextParams, #param_name, ruby_whisper_context_params_get_ ## param_name, 0); \
  rb_define_method(cContextParams, #param_name "=", ruby_whisper_context_params_set_ ## param_name, 1);

VALUE cContextParams;

static ID param_names[NUM_PARAMS];
static ID id_use_gpu;
static ID id_flash_attn;
static ID id_gpu_device;
static ID id_dtw_token_timestamps;
static ID id_dtw_aheads_preset;
static ID id_dtw_n_top;

static size_t
ruby_whisper_context_params_memsize(const void *p)
{
  const ruby_whisper_context_params *rwcp = (ruby_whisper_context_params *)p;
  if (!rwcp) {
    return 0;
  }
  return sizeof(ruby_whisper_context_params);
}

const rb_data_type_t ruby_whisper_context_params_type = {
  "ruby_whisper_context_params",
  {0, RUBY_DEFAULT_FREE, ruby_whisper_context_params_memsize,},
  0, 0,
  0
};

static VALUE
ruby_whisper_context_params_s_allocate(VALUE klass)
{
  ruby_whisper_context_params *rwcp;
  return TypedData_Make_Struct(klass, ruby_whisper_context_params, &ruby_whisper_context_params_type, rwcp);
}

DEF_BOOLEAN_ATTR_METHOD(use_gpu);
DEF_BOOLEAN_ATTR_METHOD(flash_attn);
DEF_INT_ATTR_METHOD(gpu_device);
DEF_BOOLEAN_ATTR_METHOD(dtw_token_timestamps);
DEF_INT_ATTR_METHOD(dtw_aheads_preset);

static VALUE
ruby_whisper_context_params_get_dtw_n_top(VALUE self) {
  ruby_whisper_context_params *rwcp;
  GetContextParams(self, rwcp);

  int dtw_n_top = rwcp->params.dtw_n_top;

  return dtw_n_top == -1 ? Qnil : INT2NUM(dtw_n_top);
}

static VALUE
ruby_whisper_context_params_set_dtw_n_top(VALUE self, VALUE value) {
  ruby_whisper_context_params *rwcp;
  GetContextParams(self, rwcp);

  rwcp->params.dtw_n_top = NIL_P(value) ? -1 : NUM2INT(value);

  return value;
}

#define SET_PARAM_IF_SAME(param_name) \
  if (id == id_ ## param_name) { \
    ruby_whisper_context_params_set_ ## param_name(self, value); \
    continue; \
  }

static VALUE
ruby_whisper_context_params_initialize(int argc, VALUE *argv, VALUE self)
{
  ruby_whisper_context_params *rwcp;
  TypedData_Get_Struct(self, ruby_whisper_context_params, &ruby_whisper_context_params_type, rwcp);
  rwcp->params = whisper_context_default_params();

  VALUE kw_hash;
  rb_scan_args_kw(RB_SCAN_ARGS_KEYWORDS, argc, argv, ":", &kw_hash);
  if (NIL_P(kw_hash)) {
    return Qnil;
  }

  VALUE values[NUM_PARAMS] = {Qundef};
  rb_get_kwargs(kw_hash, param_names, 0, NUM_PARAMS, values);

  ID id;
  VALUE value;
  for (int i = 0; i < NUM_PARAMS; i++) {
    id = param_names[i];
    value = values[i];
    if (value == Qundef) {
      continue;
    }
    SET_PARAM_IF_SAME(use_gpu)
    SET_PARAM_IF_SAME(flash_attn)
    SET_PARAM_IF_SAME(gpu_device)
    SET_PARAM_IF_SAME(dtw_token_timestamps)
    SET_PARAM_IF_SAME(dtw_aheads_preset)
    SET_PARAM_IF_SAME(dtw_n_top)
  }

  return Qnil;
}

#undef SET_PARAM_IF_SAME

void
init_ruby_whisper_context_params(VALUE *cContext)
{
  cContextParams = rb_define_class_under(*cContext, "Params", rb_cObject);

  rb_define_alloc_func(cContextParams, ruby_whisper_context_params_s_allocate);
  rb_define_method(cContextParams, "initialize", ruby_whisper_context_params_initialize, -1);

  DEFINE_PARAM(use_gpu, 0)
  DEFINE_PARAM(flash_attn, 1)
  DEFINE_PARAM(gpu_device, 2)
  DEFINE_PARAM(dtw_token_timestamps, 3)
  DEFINE_PARAM(dtw_aheads_preset, 4)
  DEFINE_PARAM(dtw_n_top, 5)
}

#undef DEFINE_PARAM
#undef DEF_INT_ATTR_METHOD
#undef DEF_BOOLEAN_ATTR_METHOD
#undef NUM_PARAMS
