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

enum {
#define DEF_IDX(name, type) RUBY_WHISPER_PARAKEET_PARAM_##name,
#define DEF_IDX_CALLBACK(name) RUBY_WHISPER_PARAKEET_PARAM_##name,
#define DEF_IDX_USER_DATA(name) RUBY_WHISPER_PARAKEET_PARAM_##name##_user_data,
  ITERATE_PARAMS(DEF_IDX)
  ITERATE_CALLBACK_PARAMS(DEF_IDX_CALLBACK)
  ITERATE_CALLBACK_PARAMS(DEF_IDX_USER_DATA)

  RUBY_WHISPER_PARAKEET_NUM_PARAMS
};

#define VAL_TO_INT(v) (NUM2INT(v))
#define VAL_FROM_INT(v) (INT2NUM(v))
#define VAL_TO_BOOL(v) (RTEST(v))
#define VAL_FROM_BOOL(v) (v ? Qtrue : Qfalse)

extern VALUE cParakeetParams;

extern void ruby_whisper_callback_container_mark(ruby_whisper_callback_container *rwc);
extern ruby_whisper_callback_container* ruby_whisper_callback_container_allocate(void);

static ID param_names[RUBY_WHISPER_PARAKEET_NUM_PARAMS];
typedef VALUE (*param_writer_t)(VALUE, VALUE);
static param_writer_t param_writers[RUBY_WHISPER_PARAKEET_NUM_PARAMS];

static bool
ruby_whisper_parakeet_abort_callback(void *user_data)
{
  ruby_whisper_parakeet_abort_callback_user_data *data = (ruby_whisper_parakeet_abort_callback_user_data *)user_data;

  int is_interrupted = RUBY_ATOMIC_LOAD(data->is_interrupted);

  return is_interrupted == 1;
}

void
ruby_whisper_parakeet_prepare_transcription(ruby_whisper_parakeet_params *rwpp, ruby_whisper_parakeet_abort_callback_user_data *abort_callback_user_data)
{
  rwpp->params.abort_callback = ruby_whisper_parakeet_abort_callback;
  rwpp->params.abort_callback_user_data = (void *)abort_callback_user_data;
}

static void
ruby_whisper_parakeet_params_mark(void *p)
{
  ruby_whisper_parakeet_params *rwpp = (ruby_whisper_parakeet_params *)p;

#define MARK_CONTAINER(name) ruby_whisper_callback_container_mark(rwpp->name##_container);

  ITERATE_CALLBACK_PARAMS(MARK_CONTAINER)
}

static void
ruby_whisper_parakeet_params_free(void *p)
{
  ruby_whisper_parakeet_params *rwpp = (ruby_whisper_parakeet_params *)p;
  parakeet_free_params(&rwpp->params);

#define FREE_CONTAINER(name) \
  if (rwpp->name##_container) { \
    xfree(rwpp->name##_container); \
  }

  ITERATE_CALLBACK_PARAMS(FREE_CONTAINER)
}

static size_t
ruby_whisper_parakeet_params_memsize(const void *p)
{
  const struct ruby_whisper_parakeet_params *params = p;
  if (!params) {
    return 0;
  }
  return sizeof(ruby_whisper_parakeet_params);
}

const rb_data_type_t ruby_whisper_parakeet_params_type = {
  "ruby_whisper_parakeet_params",
  {ruby_whisper_parakeet_params_mark, ruby_whisper_parakeet_params_free, ruby_whisper_parakeet_params_memsize,},
  0, 0,
  0
};

#define READER(type) VAL_FROM_##type
#define WRITER(type) VAL_TO_##type
#define DEF_PARAM_ATTR(name, type) \
  static VALUE \
  ruby_whisper_parakeet_params_get_##name(VALUE self) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    GetParakeetParams(self, rwpp); \
    return READER(type)(rwpp->params.name); \
  } \
  static VALUE \
  ruby_whisper_parakeet_params_set_##name(VALUE self, VALUE val) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    GetParakeetParams(self, rwpp); \
    rwpp->params.name = WRITER(type)(val); \
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

#define DEF_HOOK(name) \
  static VALUE \
  ruby_whisper_parakeet_params_on_##name(VALUE self) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    GetParakeetParams(self, rwpp); \
    const VALUE blk = rb_block_proc(); \
    if (!rwpp->name##_container->callbacks) { \
      rwpp->name##_container->callbacks = rb_ary_new(); \
    } \
    rb_ary_push(rwpp->name##_container->callbacks, blk); \
    return Qnil; \
  }

ITERATE_PARAMS(DEF_PARAM_ATTR)
ITERATE_CALLBACK_PARAMS(DEF_CALLBACK_PARAM_ATTR)
ITERATE_CALLBACK_PARAMS(DEF_USER_DATA_PARAM_ATTR)
ITERATE_CALLBACK_PARAMS(DEF_HOOK)

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
  VALUE value;
  ruby_whisper_parakeet_params *rwpp;
  int i;

  TypedData_Get_Struct(self, ruby_whisper_parakeet_params, &ruby_whisper_parakeet_params_type, rwpp);

#define INIT_CONTAINER(name) rwpp->name##_container = ruby_whisper_callback_container_allocate();

  ITERATE_CALLBACK_PARAMS(INIT_CONTAINER)

  rb_scan_args_kw(RB_SCAN_ARGS_KEYWORDS, argc, argv, ":", &kw_hash);
  if (NIL_P(kw_hash)) {
    return Qnil;
  }

  rb_get_kwargs(kw_hash, param_names, 0, RUBY_WHISPER_PARAKEET_NUM_PARAMS, values);

  for (i = 0; i < RUBY_WHISPER_PARAKEET_NUM_PARAMS; i++) {
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
  cParakeetParams = rb_define_class_under(*mParakeet, "Params", rb_cObject);
  rb_define_alloc_func(cParakeetParams, ruby_whisper_parakeet_params_s_allocate);

  rb_define_method(cParakeetParams, "initialize", ruby_whisper_parakeet_params_initialize, -1);

  int i = 0;
#define REGISTER_PARAM(name) \
  param_names[i] = rb_intern(#name); \
  param_writers[i] = ruby_whisper_parakeet_params_set_##name; \
  rb_define_method(cParakeetParams, #name, ruby_whisper_parakeet_params_get_##name, 0); \
  rb_define_method(cParakeetParams, #name "=", ruby_whisper_parakeet_params_set_##name, 1); \
  i++;

#define REGISTER_PARAM_ATTR(name, type) REGISTER_PARAM(name)
#define REGISTER_CALLBACK_PARAM_ATTR(name) REGISTER_PARAM(name)
#define REGISTER_USER_DATA_PARAM_ATTR(name) REGISTER_PARAM(name##_user_data)

  ITERATE_PARAMS(REGISTER_PARAM_ATTR)
  ITERATE_CALLBACK_PARAMS(REGISTER_CALLBACK_PARAM_ATTR)
  ITERATE_CALLBACK_PARAMS(REGISTER_USER_DATA_PARAM_ATTR)

#define REGISTER_HOOK(name) \
  rb_define_method(cParakeetParams, "on_" #name, ruby_whisper_parakeet_params_on_##name, 0);

  ITERATE_CALLBACK_PARAMS(REGISTER_HOOK)
}
