#include "ruby_whisper.h"

#define ITERATE_ATTRS(ITERATOR) \
  ITERATOR(use_gpu, BOOL) \
  ITERATOR(gpu_device, INT)

#define VAL_FROM_BOOL(v) ((v) ? Qtrue : Qfalse)
#define VAL_TO_BOOL(v) (RTEST(v))
#define VAL_FROM_INT(v) (INT2NUM(v))
#define VAL_TO_INT(v) (NUM2INT(v))
#define READER(type) VAL_FROM_##type
#define WRITER(type) VAL_TO_##type

#define DEF_ATTR(name, type) \
  static VALUE \
  ruby_whisper_parakeet_context_params_get_##name(VALUE self) \
  { \
    ruby_whisper_parakeet_context_params *rwpcp; \
    GetParakeetContextParams(self, rwpcp); \
    return READER(type)(rwpcp->params.name); \
  } \
  static VALUE \
  ruby_whisper_parakeet_context_params_set_##name(VALUE self, VALUE val) \
  { \
    ruby_whisper_parakeet_context_params *rwpcp; \
    GetParakeetContextParams(self, rwpcp); \
    rwpcp->params.name = WRITER(type)(val); \
    return val; \
  }

enum {
#define DEF_IDX(name, type) RUBY_WHISPER_PARAKEET_CONTEXT_PARAMS_##name,

  ITERATE_ATTRS(DEF_IDX)
  RUBY_WHISPER_PARAKEET_NUM_CONTEXT_PARAMS
};

extern VALUE cParakeetContextParams;

typedef VALUE (*param_writer_t)(VALUE, VALUE);

static ID param_names[RUBY_WHISPER_PARAKEET_NUM_CONTEXT_PARAMS];
static param_writer_t param_writers[RUBY_WHISPER_PARAKEET_NUM_CONTEXT_PARAMS];

static size_t
ruby_whisper_parakeet_context_params_memsize(const void *p)
{
  if (!p) {
    return 0;
  }
  return sizeof(ruby_whisper_parakeet_context_params);
}

const rb_data_type_t ruby_whisper_parakeet_context_params_type = {
  "ruby_whisper_parakeet_context_params",
  {0, RUBY_DEFAULT_FREE, ruby_whisper_parakeet_context_params_memsize,},
  0, 0,
  0,
};

static VALUE
ruby_whisper_parakeet_context_params_s_allocate(VALUE klass)
{
  ruby_whisper_parakeet_context_params *rwpcp;
  return TypedData_Make_Struct(klass, ruby_whisper_parakeet_context_params, &ruby_whisper_parakeet_context_params_type, rwpcp);
}

static VALUE
ruby_whisper_parakeet_context_params_initialize(int argc, VALUE *argv, VALUE self)
{
  VALUE kw_hash;
  VALUE values[RUBY_WHISPER_PARAKEET_NUM_CONTEXT_PARAMS] = {Qundef};
  VALUE value;
  ruby_whisper_parakeet_context_params *rwpcp;
  int i;

  TypedData_Get_Struct(self, ruby_whisper_parakeet_context_params, &ruby_whisper_parakeet_context_params_type, rwpcp);
  rwpcp->params = parakeet_context_default_params();

  rb_scan_args_kw(RB_SCAN_ARGS_KEYWORDS, argc, argv, ":", &kw_hash);
  if (NIL_P(kw_hash)) {
    return Qnil;
  }

  rb_get_kwargs(kw_hash, param_names, 0, RUBY_WHISPER_PARAKEET_NUM_CONTEXT_PARAMS, values);
  for (i = 0; i < RUBY_WHISPER_PARAKEET_NUM_CONTEXT_PARAMS; i++) {
    value = values[i];
    if (value == Qundef) {
      continue;
    }
    param_writers[i](self, value);
  }

  return Qnil;
}

ITERATE_ATTRS(DEF_ATTR)

void
init_ruby_whisper_parakeet_context_params(VALUE *cParakeetContext)
{
  cParakeetContextParams = rb_define_class_under(*cParakeetContext, "Params", rb_cObject);

  rb_define_alloc_func(cParakeetContextParams, ruby_whisper_parakeet_context_params_s_allocate);

  rb_define_method(cParakeetContextParams, "initialize", ruby_whisper_parakeet_context_params_initialize, -1);

  int i = 0;
#define REGISTER_ATTR(name, type) \
  param_names[i] = rb_intern(#name); \
  param_writers[i] = ruby_whisper_parakeet_context_params_set_##name; \
  rb_define_method(cParakeetContextParams, #name, ruby_whisper_parakeet_context_params_get_##name, 0); \
  rb_define_method(cParakeetContextParams, #name "=", ruby_whisper_parakeet_context_params_set_##name, 1); \
  i++;

  ITERATE_ATTRS(REGISTER_ATTR)
}
