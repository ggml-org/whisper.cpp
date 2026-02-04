#include "ruby_whisper.h"

#define DEF_BOOLEAN_ATTR_METHOD(name) \
static VALUE \
ruby_whisper_context_params_get_ ## name(VALUE self) { \
  ruby_whisper_context_params *rwcp; \
  GetContextParams(self, rwcp); \
  return rwcp->params->name ? Qtrue : Qfalse; \
} \
static VALUE \
ruby_whisper_context_params_set_ ## name(VALUE self, VALUE value) { \
  ruby_whisper_context_params *rwcp; \
  GetContextParams(self, rwcp); \
  rwcp->params->name = RTEST(value); \
  return value; \
}

#define DEF_INT_ATTR_METHOD(name) \
static VALUE \
ruby_whisper_context_params_get_ ## name(VALUE self) { \
  ruby_whisper_context_params *rwcp; \
  GetContextParams(self, rwcp); \
  return INT2NUM(rwcp->params->name); \
} \
static VALUE \
ruby_whisper_context_params_set_ ## name(VALUE self, VALUE value) { \
  ruby_whisper_context_params *rwcp; \
  GetContextParams(self, rwcp); \
  rwcp->params->name = NUM2INT(value); \
  return value; \
}

VALUE cContextParams;

static size_t
ruby_whisper_context_params_memsize(const void *p)
{
  const ruby_whisper_context_params *rwcp = (ruby_whisper_context_params *)p;
  if (!rwcp) {
    return 0;
  }
  return sizeof(rwcp) + sizeof(*rwcp->params);
}

static void
ruby_whisper_context_params_free(void *p)
{
  ruby_whisper_context_params *rwcp = (ruby_whisper_context_params *)p;
  if (rwcp->params) {
    whisper_free_context_params(rwcp->params);
    rwcp->params = NULL;
  }
  xfree(rwcp);
}

const rb_data_type_t ruby_whisper_context_params_type = {
  "ruby_whisper_context_params",
  {0, ruby_whisper_context_params_free, ruby_whisper_context_params_memsize,},
  0, 0,
  0
};

static VALUE
ruby_whisper_context_params_s_allocate(VALUE klass)
{
  ruby_whisper_context_params *rwcp;
  VALUE obj = TypedData_Make_Struct(klass, ruby_whisper_context_params, &ruby_whisper_context_params_type, rwcp);
  rwcp->params = ALLOC(struct whisper_context_params);

  return obj;
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

  int dtw_n_top = rwcp->params->dtw_n_top;

  return dtw_n_top == -1 ? Qnil : INT2NUM(dtw_n_top);
}

static VALUE
ruby_whisper_context_params_set_dtw_n_top(VALUE self, VALUE value) {
  ruby_whisper_context_params *rwcp;
  GetContextParams(self, rwcp);

  rwcp->params->dtw_n_top = NIL_P(value) ? -1 : NUM2INT(value);

  return value;
}

static VALUE
ruby_whisper_context_params_initialize(VALUE self)
{
  ruby_whisper_context_params *rwcp;
  TypedData_Get_Struct(self, ruby_whisper_context_params, &ruby_whisper_context_params_type, rwcp);
  *rwcp->params = whisper_context_default_params();

  return Qnil;
}

void
init_ruby_whisper_context_params(VALUE *cContext)
{
  cContextParams = rb_define_class_under(*cContext, "Params", rb_cObject);

  rb_define_alloc_func(cContextParams, ruby_whisper_context_params_s_allocate);
  rb_define_method(cContextParams, "initialize", ruby_whisper_context_params_initialize, 0);
  rb_define_method(cContextParams, "use_gpu", ruby_whisper_context_params_get_use_gpu, 0);
  rb_define_method(cContextParams, "use_gpu=", ruby_whisper_context_params_set_use_gpu, 1);
  rb_define_method(cContextParams, "flash_attn", ruby_whisper_context_params_get_flash_attn, 0);
  rb_define_method(cContextParams, "flash_attn=", ruby_whisper_context_params_set_flash_attn, 1);
  rb_define_method(cContextParams, "gpu_device", ruby_whisper_context_params_get_gpu_device, 0);
  rb_define_method(cContextParams, "gpu_device=", ruby_whisper_context_params_set_gpu_device, 1);
  rb_define_method(cContextParams, "dtw_token_timestamps", ruby_whisper_context_params_get_dtw_token_timestamps, 0);
  rb_define_method(cContextParams, "dtw_token_timestamps=", ruby_whisper_context_params_set_dtw_token_timestamps, 1);
  rb_define_method(cContextParams, "dtw_aheads_preset", ruby_whisper_context_params_get_dtw_aheads_preset, 0);
  rb_define_method(cContextParams, "dtw_aheads_preset=", ruby_whisper_context_params_set_dtw_aheads_preset, 1);
  rb_define_method(cContextParams, "dtw_n_top", ruby_whisper_context_params_get_dtw_n_top, 0);
  rb_define_method(cContextParams, "dtw_n_top=", ruby_whisper_context_params_set_dtw_n_top, 1);
}
