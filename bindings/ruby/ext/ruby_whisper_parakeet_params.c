#include "ruby_whisper.h"

#define ITERATE_PARAMS(ITERATOR) \
  ITERATOR(n_threads, INT) \
  ITERATOR(offset_ms, INT) \
  ITERATOR(duration_ms, INT) \
  ITERATOR(no_context, BOOL) \
  ITERATOR(audio_ctx, INT)

#define ITERATE_NORMAL_CALLBACK_NAMES(ITERATOR, DATA) \
  ITERATOR(new_segment, DATA) \
  ITERATOR(new_token, DATA) \
  ITERATOR(progress, DATA) \
  ITERATOR(encoder_begin, DATA)

#define ITERATE_NORMAL_CALLBACK_PARAM(name, ITERATOR) ITERATOR(name##_callback)
#define ITERATE_NORMAL_CALLBACK_PARAMS(ITERATOR) \
  ITERATE_NORMAL_CALLBACK_NAMES(ITERATE_NORMAL_CALLBACK_PARAM, ITERATOR)

#define ITERATE_CALLBACK_PARAMS(ITERATOR) \
  ITERATE_NORMAL_CALLBACK_PARAMS(ITERATOR) \
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
extern ID id_call;

extern void ruby_whisper_callback_container_mark(ruby_whisper_callback_container *rwc);
extern ruby_whisper_callback_container* ruby_whisper_callback_container_allocate(void);
extern bool ruby_whisper_callback_container_is_present(const ruby_whisper_callback_container *container);
extern VALUE ruby_whisper_parakeet_segment_init(VALUE context, int index);
extern VALUE ruby_whisper_parakeet_token_s_from_token_data(struct parakeet_context *context, const parakeet_token_data *token_data);

static ID param_names[RUBY_WHISPER_PARAKEET_NUM_PARAMS];
typedef VALUE (*param_writer_t)(VALUE, VALUE);
static param_writer_t param_writers[RUBY_WHISPER_PARAKEET_NUM_PARAMS];

typedef struct {
  const ruby_whisper_callback_container *container;
  struct parakeet_state *state;
  int n_new;
} call_parakeet_new_segment_callbacks_args;

static void*
call_parakeet_new_segment_callbacks(void *v_args)
{
  call_parakeet_new_segment_callbacks_args *args = (call_parakeet_new_segment_callbacks_args *)v_args;
  const ruby_whisper_callback_container *container = args->container;

  if (!NIL_P(container->callback)) {
    rb_funcall(container->callback, id_call, 4, *container->context, Qnil, INT2NUM(args->n_new), container->user_data);
  }
  if (NIL_P(container->callbacks)) {
    return NULL;
  }
  const long n_callbacks = RARRAY_LEN(container->callbacks);
  if (n_callbacks == 0) {
    return NULL;
  }
  const int n_segments = parakeet_full_n_segments_from_state(args->state);
  for (int i = args->n_new; i > 0; i--) {
    int i_segment = n_segments - i;
    VALUE segment = ruby_whisper_parakeet_segment_init(*container->context, i_segment);
    for (int j = 0; j < n_callbacks; j++) {
      VALUE cb = rb_ary_entry(container->callbacks, j);
      rb_funcall(cb, id_call, 1, segment);
    }
  }

  return NULL;
}

static void
ruby_whisper_parakeet_new_segment_callback(struct parakeet_context *context, struct parakeet_state *state, int n_new, void *user_data)
{
  const ruby_whisper_callback_container *container = (ruby_whisper_callback_container *)user_data;
  if (!ruby_whisper_callback_container_is_present(container)) {
    return;
  }

  call_parakeet_new_segment_callbacks_args args = {
    container,
    state,
    n_new,
  };
  rb_thread_call_with_gvl(call_parakeet_new_segment_callbacks, (void *)&args);
}

typedef struct {
  const ruby_whisper_callback_container *container;
  struct parakeet_context *context;
  struct parakeet_state *state;
  const parakeet_token_data *token_data;
} call_parakeet_new_token_callbacks_args;

static void*
call_parakeet_new_token_callbacks(void *v_args)
{
  call_parakeet_new_token_callbacks_args *args = (call_parakeet_new_token_callbacks_args *)v_args;
  VALUE token = Qnil;
  const ruby_whisper_callback_container *container = args->container;

  if (!NIL_P(container->callback)) {
    token = ruby_whisper_parakeet_token_s_from_token_data(args->context, args->token_data);
    rb_funcall(container->callback, id_call, 4, *container->context, Qnil, token, container->user_data);
  }
  if (NIL_P(container->callbacks)) {
    return NULL;
  }
  const long n_callbacks = RARRAY_LEN(container->callbacks);
  if (n_callbacks == 0) {
    return NULL;
  }
  if (NIL_P(token)) {
    token = ruby_whisper_parakeet_token_s_from_token_data(args->context, args->token_data);
  }
  for (int i = 0; i < n_callbacks; i++) {
    VALUE cb = rb_ary_entry(container->callbacks, i);
    rb_funcall(cb, id_call, 1, token);
  }

  return NULL;
}

static void
ruby_whisper_parakeet_new_token_callback(struct parakeet_context *context, struct parakeet_state *state, const parakeet_token_data *token_data, void *user_data)
{
  const ruby_whisper_callback_container *container = (ruby_whisper_callback_container *)user_data;
  if (!ruby_whisper_callback_container_is_present(container)) {
    return;
  }

  call_parakeet_new_token_callbacks_args args = {
    container,
    context,
    state,
    token_data,
  };
  rb_thread_call_with_gvl(call_parakeet_new_token_callbacks, (void *)&args);
}

typedef struct {
  const ruby_whisper_callback_container *container;
  struct parakeet_state *state;
  int progress;
} call_parakeet_progress_callbacks_args;

static void*
call_parakeet_progress_callback(void *v_args)
{
  call_parakeet_progress_callbacks_args *args = (call_parakeet_progress_callbacks_args *)v_args;
  const ruby_whisper_callback_container *container = args->container;

  if (!NIL_P(container->callback)) {
    rb_funcall(container->callback, id_call, 4, *container->context, Qnil, INT2NUM(args->progress), container->user_data);
  }
  if (NIL_P(container->callbacks)) {
    return NULL;
  }
  const long n_callbacks = RARRAY_LEN(container->callbacks);
  if (n_callbacks == 0) {
    return NULL;
  }
  for (long i = 0; i < n_callbacks; i++) {
    VALUE cb = rb_ary_entry(container->callbacks, i);
    rb_funcall(cb, id_call, 1, INT2NUM(args->progress));
  }

  return NULL;
}

static void
ruby_whisper_parakeet_progress_callback(struct parakeet_context *context, struct parakeet_state *state, int progress, void *user_data)
{
  const ruby_whisper_callback_container *container = (ruby_whisper_callback_container *)user_data;
  if (!ruby_whisper_callback_container_is_present(container)) {
    return;
  }

  call_parakeet_progress_callbacks_args args = {
    container,
    state,
    progress,
  };
  rb_thread_call_with_gvl(call_parakeet_progress_callback, (void *)&args);
}

typedef struct {
  const ruby_whisper_callback_container *container;
  struct parakeet_state *state;
  bool is_continued;
} call_parakeet_encoder_begin_callbacks_args;

static void*
call_parakeet_encoder_begin_callbacks(void *v_args)
{
  call_parakeet_encoder_begin_callbacks_args *args = (call_parakeet_encoder_begin_callbacks_args *)v_args;
  const ruby_whisper_callback_container *container = args->container;
  VALUE result = Qnil;

  if (!NIL_P(container->callback)) {
    result = rb_funcall(container->callback, id_call, 3, *container->context, Qnil, container->user_data);
    if (result == Qfalse) {
      args->is_continued = false;
      return NULL;
    }
  }
  if (NIL_P(container->callbacks)) {
    return NULL;
  }
  const long n_callbacks = RARRAY_LEN(container->callbacks);
  if (n_callbacks == 0) {
    return NULL;
  }
  for (long i = 0; i < n_callbacks; i++) {
    VALUE cb = rb_ary_entry(container->callbacks, i);
    result = rb_funcall(cb, id_call, 0);
    if (result == Qfalse) {
      args->is_continued = false;
      return NULL;
    }
  }

  return NULL;
}

static bool
ruby_whisper_parakeet_encoder_begin_callback(struct parakeet_context *context, struct parakeet_state *state, void *user_data)
{
  const ruby_whisper_callback_container *container = (ruby_whisper_callback_container *)user_data;
  if (!ruby_whisper_callback_container_is_present(container)) {
    return true;
  }

  call_parakeet_encoder_begin_callbacks_args args = {
      container,
      state,
      true,
  };
  rb_thread_call_with_gvl(call_parakeet_encoder_begin_callbacks, (void *)&args);

  return args.is_continued;
}

typedef struct {
  const ruby_whisper_callback_container *container;
  bool is_interrupted;
} call_parakeet_abort_callbacks_args;

static void*
call_parakeet_abort_callbacks(void *v_args)
{
  call_parakeet_abort_callbacks_args *args = (call_parakeet_abort_callbacks_args *)v_args;
  const ruby_whisper_callback_container *container = args->container;
  VALUE result = Qnil;

  if (!NIL_P(container->callback)) {
    result = rb_funcall(container->callback, id_call, 1, container->user_data);
    if (RTEST(result)) {
      args->is_interrupted = true;
      return NULL;
    }
  }
  if (NIL_P(container->callbacks)) {
    return NULL;
  }
  const long n_callbacks = RARRAY_LEN(container->callbacks);
  if (n_callbacks == 0) {
    return NULL;
  }
  VALUE cb;
  for (long i = 0; i < n_callbacks; i++) {
    cb = rb_ary_entry(container->callbacks, i);
    result = rb_funcall(cb, id_call, 0);
    if (RTEST(result)) {
      args->is_interrupted = true;
      return NULL;
    }
  }

  return NULL;
}

static bool
ruby_whisper_parakeet_abort_callback(void *user_data)
{
  ruby_whisper_abort_callback_user_data *data = (ruby_whisper_abort_callback_user_data *)user_data;

  int is_interrupted = RUBY_ATOMIC_LOAD(data->is_interrupted);
  if (is_interrupted) {
    return true;
  }

  if (!(data->callback_container) || !ruby_whisper_callback_container_is_present(data->callback_container)) {
    return false;
  }

  call_parakeet_abort_callbacks_args args = {
    data->callback_container,
    false,
  };
  rb_thread_call_with_gvl(call_parakeet_abort_callbacks, (void *)&args);

  return args.is_interrupted;
}

#define CALLBACK_CONTAINER_NAME(name) name ## _container

void
ruby_whisper_parakeet_prepare_transcription(ruby_whisper_parakeet_params *rwpp, VALUE *context, ruby_whisper_abort_callback_user_data *abort_callback_user_data)
{
#define PARAM_NAME(name) name
#define USER_DATA_NAME(name) name##_user_data
#define REGISTER_CALLBACK(name) \
  if (ruby_whisper_callback_container_is_present(rwpp->CALLBACK_CONTAINER_NAME(name))) { \
    rwpp->CALLBACK_CONTAINER_NAME(name)->context = context; \
    rwpp->params.PARAM_NAME(name) = ruby_whisper_parakeet_##name; \
    rwpp->params.USER_DATA_NAME(name) = rwpp->CALLBACK_CONTAINER_NAME(name); \
  }

  ITERATE_NORMAL_CALLBACK_PARAMS(REGISTER_CALLBACK)

  if (ruby_whisper_callback_container_is_present(rwpp->abort_callback_container)) {
    abort_callback_user_data->callback_container = rwpp->abort_callback_container;
  }
  rwpp->params.abort_callback = ruby_whisper_parakeet_abort_callback;
  rwpp->params.abort_callback_user_data = (void *)abort_callback_user_data;
}

static void
ruby_whisper_parakeet_params_mark(void *p)
{
  ruby_whisper_parakeet_params *rwpp = (ruby_whisper_parakeet_params *)p;

#define MARK_CONTAINER(name) \
  if (rwpp->name##_container) { \
    ruby_whisper_callback_container_mark(rwpp->name##_container); \
  }

  ITERATE_CALLBACK_PARAMS(MARK_CONTAINER)
}

static void
ruby_whisper_parakeet_params_free(void *p)
{
  ruby_whisper_parakeet_params *rwpp = (ruby_whisper_parakeet_params *)p;

#define FREE_CONTAINER(name) \
  if (rwpp->name##_container) { \
    xfree(rwpp->name##_container); \
  }

  ITERATE_CALLBACK_PARAMS(FREE_CONTAINER)

  xfree(rwpp);
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

#define DEF_HOOK(name, data) \
  static VALUE \
  ruby_whisper_parakeet_params_on_##name(VALUE self) \
  { \
    ruby_whisper_parakeet_params *rwpp; \
    GetParakeetParams(self, rwpp); \
    const VALUE blk = rb_block_proc(); \
    if (NIL_P(rwpp->name##_callback_container->callbacks)) { \
      rwpp->name##_callback_container->callbacks = rb_ary_new(); \
    } \
    rb_ary_push(rwpp->name##_callback_container->callbacks, blk); \
    return Qnil; \
  }

ITERATE_PARAMS(DEF_PARAM_ATTR)
ITERATE_CALLBACK_PARAMS(DEF_CALLBACK_PARAM_ATTR)
ITERATE_CALLBACK_PARAMS(DEF_USER_DATA_PARAM_ATTR)
ITERATE_NORMAL_CALLBACK_NAMES(DEF_HOOK, _)

static VALUE
ruby_whisper_parakeet_params_abort_on(VALUE self)
{
  ruby_whisper_parakeet_params *rwpp;
  GetParakeetParams(self, rwpp);
  const VALUE blk = rb_block_proc();
  if (NIL_P(rwpp->abort_callback_container->callbacks)) {
    rwpp->abort_callback_container->callbacks = rb_ary_new();
  }
  rb_ary_push(rwpp->abort_callback_container->callbacks, blk);

  return Qnil;
}

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

#define REGISTER_HOOK(name, data) \
  rb_define_method(cParakeetParams, "on_" #name, ruby_whisper_parakeet_params_on_##name, 0);

  ITERATE_NORMAL_CALLBACK_NAMES(REGISTER_HOOK, _)

  rb_define_method(cParakeetParams, "abort_on", ruby_whisper_parakeet_params_abort_on, 0);
}
