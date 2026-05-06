#include "ruby_whisper.h"

#define ITERATE_PARAMS(ITERATOR) \
  ITERATOR(n_threads, INT) \
  ITERATOR(offset_ms, INT) \
  ITERATOR(duration_ms, INT) \
  ITERATOR(no_context, BOOL) \
  ITERATOR(audio_ctx, INT) \
  ITERATOR(chunk_length_ms, INT) \
  ITERATOR(left_context_ms, INT) \
  ITERATOR(right_context_ms, INT)

#define ITERATE_CALLBACK_PARAMS(ITERATOR) \
  ITERATOR(new_segment_callback) \
  ITERATOR(new_token_callback) \
  ITERATOR(progress_callback) \
  ITERATOR(encoder_begin_callback) \
  ITERATOR(abort_callback)

#define ITERATE_USER_DATA_PARAMS(ITERATOR) \
  ITERATOR(new_segment_callback) \
  ITERATOR(new_token_callback) \
  ITERATOR(progress_callback) \
  ITERATOR(encoder_begin_callback) \
  ITERATOR(abort_callback)

enum {
#define DEF_IDX(name, type) RUBY_WHISPER_PARAKEET_PARAM_##name,
#define DEF_IDX_CALLBACK(name) RUBY_WHISPER_PARAKEET_PARAM_##name,
#define DEF_IDX_USER_DATA(name) RUBY_WHISPER_PARAKEET_PARAM_##name##_user_data,
  ITERATE_PARAMS(DEF_IDX)
  ITERATE_CALLBACK_PARAMS(DEF_IDX_CALLBACK)
  ITERATE_USER_DATA_PARAMS(DEF_IDX_USER_DATA)
#undef DEF_IDX
#undef DEF_IDX_CALLBACK
#undef DEF_IDX_USER_DATA
  RUBY_WHISPER_PARAKEET_NUM_PARAMS
};

#define VAL_TO_INT(v) (NUM2INT(v))
#define VAL_FROM_INT(v) (INT2NUM(v))
#define VAL_TO_BOOL(v) (RTEST(v))
#define VAL_FROM_BOOL(v) (v ? Qtrue : Qfalse)

extern void ruby_whisper_callback_container_mark(ruby_whisper_callback_container *rwc);
extern ruby_whisper_callback_container* ruby_whisper_callback_container_allocate(void);

static ID param_names[RUBY_WHISPER_PARAKEET_NUM_PARAMS];
typedef VALUE (*param_writer_t)(VALUE, VALUE);
static param_writer_t param_writers[RUBY_WHISPER_PARAKEET_NUM_PARAMS];

static void
ruby_whisper_parakeet_params_mark(void *p)
{
  ruby_whisper_parakeet_params *rwpp = (ruby_whisper_parakeet_params *)p;
  ruby_whisper_callback_container_mark(rwpp->new_segment_callback_container);
  ruby_whisper_callback_container_mark(rwpp->new_token_callback_container);
  ruby_whisper_callback_container_mark(rwpp->progress_callback_container);
  ruby_whisper_callback_container_mark(rwpp->encoder_begin_callback_container);
  ruby_whisper_callback_container_mark(rwpp->abort_callback_container);
}

static void
ruby_whisper_parakeet_params_free(void *p)
{
  ruby_whisper_parakeet_params *rwpp = (ruby_whisper_parakeet_params *)p;
  if (rwpp->params.new_segment_callback_user_data) {
    xfree(&rwpp->params.new_segment_callback_user_data);
  }
  if (rwpp->params.new_token_callback_user_data) {
    xfree(&rwpp->params.new_token_callback_user_data);
  }
  if (rwpp->params.progress_callback_user_data) {
    xfree(&rwpp->params.progress_callback_user_data);
  }
  if (rwpp->params.encoder_begin_callback_user_data) {
    xfree(&rwpp->params.encoder_begin_callback_user_data);
  }
  if (rwpp->params.abort_callback_user_data) {
    xfree(&rwpp->params.abort_callback_user_data);
  }
  xfree(rwpp);
}

static size_t
ruby_whisper_parakeet_params_memsize(const void *p)
{
  const struct ruby_whisper_parakeet_params *params = p;
  size_t size = sizeof(params);
  if (!params) {
    return 0;
  }
  return size;
}

const rb_data_type_t ruby_whisper_parakeet_params_type = {
  "ruby_whisper_parakeet_params",
  {ruby_whisper_parakeet_params_mark, ruby_whisper_parakeet_params_free, ruby_whisper_parakeet_params_memsize,},
  0, 0,
  0
};

#define DEF_BOOL_PARAM_ATTR(name) \
  static VALUE \
  ruby_whisper_parakeet_params_get_##name(VALUE self) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    GetParakeetParams(self, rwpp); \
    return VAL_FROM_BOOL(rwpp->params.name); \
  } \
  static VALUE \
  ruby_whisper_parakeet_params_set_##name(VALUE self, VALUE val) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    GetParakeetParams(self, rwpp); \
    rwpp->params.name = VAL_TO_BOOL(val); \
    return val; \
  }

#define DEF_INT_PARAM_ATTR(name) \
  static VALUE \
  ruby_whisper_parakeet_params_get_##name(VALUE self) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    GetParakeetParams(self, rwpp); \
    return VAL_FROM_INT(rwpp->params.name); \
  } \
  static VALUE \
  ruby_whisper_parakeet_params_set_##name(VALUE self, VALUE val) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    GetParakeetParams(self, rwpp); \
    rwpp->params.name = VAL_TO_INT(val); \
    return val; \
  }

#define CALLBACK_CONTAINER_NAME(name) name ## _container

#define DEF_CALLBACK_PARAM_ATTR(name) \
  static VALUE \
  ruby_whisper_parakeet_params_get_##name(VALUE self) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    GetParakeetParams(self, rwpp); \
    return rwpp->CALLBACK_CONTAINER_NAME(name)->callback; \
  } \
  static VALUE \
  ruby_whisper_parakeet_params_set_##name(VALUE self, VALUE val) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    GetParakeetParams(self, rwpp); \
    rwpp->CALLBACK_CONTAINER_NAME(name)->callback = (val); \
    return val; \
  }

#define DEF_USER_DATA_PARAM_ATTR(name) \
  static VALUE \
  ruby_whisper_parakeet_params_get_##name##_user_data(VALUE self) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    GetParakeetParams(self, rwpp); \
    return rwpp->CALLBACK_CONTAINER_NAME(name)->user_data; \
  } \
  static VALUE \
  ruby_whisper_parakeet_params_set_##name##_user_data(VALUE self, VALUE val) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    GetParakeetParams(self, rwpp); \
    rwpp->CALLBACK_CONTAINER_NAME(name)->user_data = val; \
    return val; \
  }

#define DEF_PARAM_ATTR(name, type) DEF_PARAM_ATTR_I(name, type)
#define DEF_PARAM_ATTR_I(name, type) DEF_##type##_PARAM_ATTR(name)

ITERATE_PARAMS(DEF_PARAM_ATTR)
ITERATE_CALLBACK_PARAMS(DEF_CALLBACK_PARAM_ATTR)
ITERATE_USER_DATA_PARAMS(DEF_USER_DATA_PARAM_ATTR)

static VALUE
ruby_whisper_parakeet_params_s_allocate(VALUE klass)
{
  ruby_whisper_parakeet_params *rwpp;
  VALUE obj = TypedData_Make_Struct(klass, ruby_whisper_parakeet_params, &ruby_whisper_parakeet_params_type, rwpp);
  rwpp->params = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
  return obj;
}

static VALUE
ruby_whisper_parakeet_params_initialize(int argc, VALUE *argv, VALUE self)
{
  VALUE kw_hash;
  VALUE values[RUBY_WHISPER_PARAKEET_NUM_PARAMS] = {Qundef};
  VALUE id;
  VALUE value;
  ruby_whisper_parakeet_params *rwpp;
  int i;

  TypedData_Get_Struct(self, ruby_whisper_parakeet_params, &ruby_whisper_parakeet_params_type, rwpp);

  rwpp->new_segment_callback_container = ruby_whisper_callback_container_allocate();
  rwpp->new_token_callback_container = ruby_whisper_callback_container_allocate();
  rwpp->progress_callback_container = ruby_whisper_callback_container_allocate();
  rwpp->encoder_begin_callback_container = ruby_whisper_callback_container_allocate();
  rwpp->abort_callback_container = ruby_whisper_callback_container_allocate();

  rb_scan_args_kw(RB_SCAN_ARGS_KEYWORDS, argc, argv, ":", &kw_hash);
  if (NIL_P(kw_hash)) {
    return Qnil;
  }

  rb_get_kwargs(kw_hash, param_names, 0, RUBY_WHISPER_PARAKEET_NUM_PARAMS, values);

  for (i = 0; i < RUBY_WHISPER_PARAKEET_NUM_PARAMS; i++) {
    id = param_names[i];
    value = values[i];
    if (value == Qundef) {
      continue;
    }
    param_writers[i](self, value);
  }

  return Qnil;
}

void
init_ruby_whisper_parakeet_params(VALUE *mParakeet)
{
  VALUE cParakeetParams = rb_define_class_under(*mParakeet, "Params", rb_cObject);
  rb_define_alloc_func(cParakeetParams, ruby_whisper_parakeet_params_s_allocate);

  rb_define_method(cParakeetParams, "initialize", ruby_whisper_parakeet_params_initialize, -1);

  int i = 0;
#define REGISTER_PARAM_ATTR(name, type) \
  param_names[i] = rb_intern(#name); \
  param_writers[i] = ruby_whisper_parakeet_params_set_##name; \
  rb_define_method(cParakeetParams, #name, ruby_whisper_parakeet_params_get_##name, 0); \
  rb_define_method(cParakeetParams, #name "=", ruby_whisper_parakeet_params_set_##name, 1); \
  i++;
#define REGISTER_CALLBACK_PARAM_ATTR(name) \
  param_names[i] = rb_intern(#name); \
  param_writers[i] = ruby_whisper_parakeet_params_set_##name; \
  rb_define_method(cParakeetParams, #name, ruby_whisper_parakeet_params_get_##name, 0); \
  rb_define_method(cParakeetParams, #name "=", ruby_whisper_parakeet_params_set_##name, 1); \
  i++;
#define REGISTER_USER_DATA_PARAM_ATTR(name) \
  param_names[i] = rb_intern(#name "_user_data"); \
  param_writers[i] = ruby_whisper_parakeet_params_set_##name##_user_data; \
  rb_define_method(cParakeetParams, #name "_user_data", ruby_whisper_parakeet_params_get_##name##_user_data, 0); \
  rb_define_method(cParakeetParams, #name "_user_data=", ruby_whisper_parakeet_params_set_##name##_user_data, 1); \
  i++;

  ITERATE_PARAMS(REGISTER_PARAM_ATTR)
  ITERATE_CALLBACK_PARAMS(REGISTER_CALLBACK_PARAM_ATTR)
  ITERATE_USER_DATA_PARAMS(REGISTER_USER_DATA_PARAM_ATTR)

#undef REGISTER_PARAM_ATTR
#undef REGISTER_CALLBACK_PARAM_ATTR
#undef REGISTER_USER_DATA_PARAM_ATTR
}

#undef VAL_TO_INT
#undef VAL_FROM_INT
#undef VAL_TO_BOOL
#undef VAL_FROM_BOOL
#undef DEF_BOOL_PARAM_ATTR
#undef DEF_INT_PARAM_ATTR
#undef CALLBACK_CONTAINER_NAME
#undef DEF_CALLBACK_PARAM_ATTR
#undef DEF_USER_DATA_PARAM_ATTR
#undef DEF_PARAM_ATTR
#undef DEF_PARAM_ATTR_I
#undef ITERATE_PARAMS
#undef ITERATE_CALLBACK_PARAMS
#undef ITERATE_USER_DATA_PARAMS
