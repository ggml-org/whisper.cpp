#include "ruby_whisper.h"

#define ITERATE_ATTRS(ITERATOR) \
  ITERATOR(n_vocab) \
  ITERATOR(n_audio_ctx) \
  ITERATOR(n_audio_state) \
  ITERATOR(n_audio_head) \
  ITERATOR(n_audio_layer) \
  ITERATOR(n_mels) \
  ITERATOR(ftype)

extern rb_data_type_t ruby_whisper_parakeet_context_type;
extern VALUE cParakeetModel;

static void
ruby_whisper_parakeet_model_mark(void *p)
{
  ruby_whisper_parakeet_model *rwpm = (ruby_whisper_parakeet_model *)p;
  if (!NIL_P(rwpm->context)) {
    rb_gc_mark(rwpm->context);
  }
}

static size_t
ruby_whisper_parakeet_model_memsize(const void *p)
{
  if (!p) {
    return 0;
  }
  return sizeof(ruby_whisper_parakeet_model);
}

static const rb_data_type_t ruby_whisper_parakeet_model_type = {
  "ruby_whisper_parakeet_model",
  {ruby_whisper_parakeet_model_mark, RUBY_DEFAULT_FREE, ruby_whisper_parakeet_model_memsize},
  0, 0,
  0
};

static VALUE
ruby_whisper_parakeet_model_s_allocate(VALUE klass)
{
  ruby_whisper_parakeet_model *rwpm;
  VALUE model = TypedData_Make_Struct(klass, ruby_whisper_parakeet_model, &ruby_whisper_parakeet_model_type, rwpm);
  rwpm->context = Qnil;

  return model;
}

VALUE
ruby_whisper_parakeet_model_s_new(VALUE context)
{
  const VALUE model = ruby_whisper_parakeet_model_s_allocate(cParakeetModel);
  ruby_whisper_parakeet_model *rwpm;
  TypedData_Get_Struct(model, ruby_whisper_parakeet_model, &ruby_whisper_parakeet_model_type, rwpm);
  rwpm->context = context;
  return model;
}

#define DEF_ATTR(name) \
  static VALUE \
  ruby_whisper_parakeet_model_get_##name(VALUE self) \
  { \
    ruby_whisper_parakeet_model *rwpm; \
    ruby_whisper_parakeet_context *rwpc; \
    GetParakeetModel(self, rwpm); \
    GetParakeetContext(rwpm->context, rwpc); \
    return INT2NUM(parakeet_model_##name(rwpc->context)); \
  }

ITERATE_ATTRS(DEF_ATTR)

void
init_ruby_whisper_parakeet_model(VALUE *mParakeet)
{
  cParakeetModel = rb_define_class_under(*mParakeet, "Model", rb_cObject);

  rb_define_alloc_func(cParakeetModel, ruby_whisper_parakeet_model_s_allocate);

#define REGISTER_ATTR(name) \
  rb_define_method(cParakeetModel, #name, ruby_whisper_parakeet_model_get_##name, 0);

  ITERATE_ATTRS(REGISTER_ATTR)
}
