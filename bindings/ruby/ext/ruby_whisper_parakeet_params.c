#include "ruby_whisper.h"

#define ITERATE_PARAMS(ITERATOR) \
  ITERATOR(n_threads, n_threads,INT) \
  ITERATOR(offset_ms, offset_ms, INT) \
  ITERATOR(duration_ms, duration_ms, INT) \
  ITERATOR(no_context, no_context, BOOL) \
  ITERATOR(audio_ctx, audio_ctx, INT) \
  ITERATOR(chunk_length_ms, chunk_length_ms, INT) \
  ITERATOR(left_context_ms, left_context_ms, INT) \
  ITERATOR(right_context_ms, right_context_ms, INT) \
  ITERATOR(new_segment_callback, new_segment_callback, CALLBACK) \
  ITERATOR(new_segment_callback_user_data, new_segment_callback, USER_DATA) \
  ITERATOR(new_token_callback, new_token_callback, CALLBACK) \
  ITERATOR(new_token_callback_user_data, new_token_callback, USER_DATA) \
  ITERATOR(progress_callback, progress_callback, CALLBACK) \
  ITERATOR(progress_callback_user_data, progress_callback, USER_DATA) \
  ITERATOR(encoder_begin_callback, encoder_begin_callback, CALLBACK) \
  ITERATOR(encoder_begin_callback_user_data, encoder_begin_callback, USER_DATA) \
  ITERATOR(abort_callback, abort_callback, CALLBACK) \
  ITERATOR(abort_callback_user_data, abort_callback, USER_DATA)

enum {
#define DEF_IDX(name, cb, type) RUBY_WHISPER_PARAKEET_PARAM_##name,
  ITERATE_PARAMS(DEF_IDX)
#undef DEF_IDX
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

#define DEF_BOOL_PARAM_ATTR(name, cb) \
  static VALUE \
  ruby_whisper_parakeet_params_get_##name(VALUE self) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    TypedData_Get_Struct(self, ruby_whisper_parakeet_params, &ruby_whisper_parakeet_params_type, rwpp); \
    return VAL_FROM_BOOL(rwpp->params.name); \
  } \
  static VALUE \
  ruby_whisper_parakeet_params_set_##name(VALUE self, VALUE val) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    TypedData_Get_Struct(self, ruby_whisper_parakeet_params, &ruby_whisper_parakeet_params_type, rwpp); \
    rwpp->params.name = VAL_TO_BOOL(val); \
    return val; \
  }

#define DEF_INT_PARAM_ATTR(name, cb) \
  static VALUE \
  ruby_whisper_parakeet_params_get_##name(VALUE self) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    TypedData_Get_Struct(self, ruby_whisper_parakeet_params, &ruby_whisper_parakeet_params_type, rwpp); \
    return VAL_FROM_INT(rwpp->params.name); \
  } \
  static VALUE \
  ruby_whisper_parakeet_params_set_##name(VALUE self, VALUE val) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    TypedData_Get_Struct(self, ruby_whisper_parakeet_params, &ruby_whisper_parakeet_params_type, rwpp); \
    rwpp->params.name = VAL_TO_INT(val); \
    return val; \
  }

#define CALLBACK_CONTAINER_NAME(name) name ## _container

#define DEF_CALLBACK_PARAM_ATTR(name, cb) \
  static VALUE \
  ruby_whisper_parakeet_params_get_##name(VALUE self) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    TypedData_Get_Struct(self, ruby_whisper_parakeet_params, &ruby_whisper_parakeet_params_type, rwpp); \
    return rwpp->CALLBACK_CONTAINER_NAME(cb)->callback; \
  } \
  static VALUE \
  ruby_whisper_parakeet_params_set_##name(VALUE self, VALUE val) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    TypedData_Get_Struct(self, ruby_whisper_parakeet_params, &ruby_whisper_parakeet_params_type, rwpp); \
    rwpp->CALLBACK_CONTAINER_NAME(cb)->callback = (val); \
    return val; \
  }

#define DEF_USER_DATA_PARAM_ATTR(name, cb) \
  static VALUE \
  ruby_whisper_parakeet_params_get_##name(VALUE self) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    TypedData_Get_Struct(self, ruby_whisper_parakeet_params, &ruby_whisper_parakeet_params_type, rwpp); \
    return rwpp->CALLBACK_CONTAINER_NAME(cb)->user_data; \
  } \
  static VALUE \
  ruby_whisper_parakeet_params_set_##name(VALUE self, VALUE val) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    TypedData_Get_Struct(self, ruby_whisper_parakeet_params, &ruby_whisper_parakeet_params_type, rwpp); \
    rwpp->CALLBACK_CONTAINER_NAME(cb)->user_data = val; \
    return val; \
  }

#define DEF_PARAM_ATTR(name, cb, type) DEF_PARAM_ATTR_I(name, cb, type)
#define DEF_PARAM_ATTR_I(name, cb, type) DEF_##type##_PARAM_ATTR(name, cb)

ITERATE_PARAMS(DEF_PARAM_ATTR)

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
#define REGISTER_PARAM_ATTR(name, cb, type) \
  param_names[i] = rb_intern(#name); \
  param_writers[i] = ruby_whisper_parakeet_params_set_##name; \
  rb_define_method(cParakeetParams, #name, ruby_whisper_parakeet_params_get_##name, 0); \
  rb_define_method(cParakeetParams, #name "=", ruby_whisper_parakeet_params_set_##name, 1); \
  i++;

  ITERATE_PARAMS(REGISTER_PARAM_ATTR)

#undef REGISTER_PARAM_ATTR
}

#undef ITERATE_PARAMS
