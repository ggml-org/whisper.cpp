#include "ruby_whisper.h"

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
}
