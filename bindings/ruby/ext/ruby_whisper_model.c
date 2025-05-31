#include <ruby.h>
#include "ruby_whisper.h"

extern const rb_data_type_t ruby_whisper_type;

extern VALUE cModel;

static void rb_whisper_model_mark(void *p) {
  ruby_whisper_model *rwm = (ruby_whisper_model *)p;
  if (rwm->context) {
    rb_gc_mark(rwm->context);
  }
}

static size_t
ruby_whisper_model_memsize(const void *p)
{
  const ruby_whisper_model *rwm = (const ruby_whisper_model *)p;
  size_t size = sizeof(rwm);
  if (!rwm) {
    return 0;
  }
  return size;
}

static const rb_data_type_t rb_whisper_model_type = {
  "ruby_whisper_model",
  {rb_whisper_model_mark, RUBY_DEFAULT_FREE, ruby_whisper_model_memsize,},
  0, 0,
  0
};

static VALUE ruby_whisper_model_allocate(VALUE klass) {
  ruby_whisper_model *rwm;
  return TypedData_Make_Struct(klass, ruby_whisper_model, &rb_whisper_model_type, rwm);
}

VALUE rb_whisper_model_s_new(VALUE context) {
  ruby_whisper_model *rwm;
  const VALUE model = ruby_whisper_model_allocate(cModel);
  TypedData_Get_Struct(model, ruby_whisper_model, &rb_whisper_model_type, rwm);
  rwm->context = context;
  return model;
};

/*
 * call-seq:
 *   n_vocab -> Integer
 */
static VALUE
ruby_whisper_model_n_vocab(VALUE self)
{
  ruby_whisper_model *rwm;
  TypedData_Get_Struct(self, ruby_whisper_model, &rb_whisper_model_type, rwm);
  ruby_whisper *rw;
  TypedData_Get_Struct(rwm->context, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_vocab(rw->context));
}

/*
 * call-seq:
 *   n_audio_ctx -> Integer
 */
static VALUE
ruby_whisper_model_n_audio_ctx(VALUE self)
{
  ruby_whisper_model *rwm;
  TypedData_Get_Struct(self, ruby_whisper_model, &rb_whisper_model_type, rwm);
  ruby_whisper *rw;
  TypedData_Get_Struct(rwm->context, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_audio_ctx(rw->context));
}

/*
 * call-seq:
 *   n_audio_state -> Integer
 */
static VALUE
ruby_whisper_model_n_audio_state(VALUE self)
{
  ruby_whisper_model *rwm;
  TypedData_Get_Struct(self, ruby_whisper_model, &rb_whisper_model_type, rwm);
  ruby_whisper *rw;
  TypedData_Get_Struct(rwm->context, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_audio_state(rw->context));
}

/*
 * call-seq:
 *   n_audio_head -> Integer
 */
static VALUE
ruby_whisper_model_n_audio_head(VALUE self)
{
  ruby_whisper_model *rwm;
  TypedData_Get_Struct(self, ruby_whisper_model, &rb_whisper_model_type, rwm);
  ruby_whisper *rw;
  TypedData_Get_Struct(rwm->context, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_audio_head(rw->context));
}

/*
 * call-seq:
 *   n_audio_layer -> Integer
 */
static VALUE
ruby_whisper_model_n_audio_layer(VALUE self)
{
  ruby_whisper_model *rwm;
  TypedData_Get_Struct(self, ruby_whisper_model, &rb_whisper_model_type, rwm);
  ruby_whisper *rw;
  TypedData_Get_Struct(rwm->context, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_audio_layer(rw->context));
}

/*
 * call-seq:
 *   n_text_ctx -> Integer
 */
static VALUE
ruby_whisper_model_n_text_ctx(VALUE self)
{
  ruby_whisper_model *rwm;
  TypedData_Get_Struct(self, ruby_whisper_model, &rb_whisper_model_type, rwm);
  ruby_whisper *rw;
  TypedData_Get_Struct(rwm->context, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_text_ctx(rw->context));
}

/*
 * call-seq:
 *   n_text_state -> Integer
 */
static VALUE
ruby_whisper_model_n_text_state(VALUE self)
{
  ruby_whisper_model *rwm;
  TypedData_Get_Struct(self, ruby_whisper_model, &rb_whisper_model_type, rwm);
  ruby_whisper *rw;
  TypedData_Get_Struct(rwm->context, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_text_state(rw->context));
}

/*
 * call-seq:
 *   n_text_head -> Integer
 */
static VALUE
ruby_whisper_model_n_text_head(VALUE self)
{
  ruby_whisper_model *rwm;
  TypedData_Get_Struct(self, ruby_whisper_model, &rb_whisper_model_type, rwm);
  ruby_whisper *rw;
  TypedData_Get_Struct(rwm->context, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_text_head(rw->context));
}

/*
 * call-seq:
 *   n_text_layer -> Integer
 */
static VALUE
ruby_whisper_model_n_text_layer(VALUE self)
{
  ruby_whisper_model *rwm;
  TypedData_Get_Struct(self, ruby_whisper_model, &rb_whisper_model_type, rwm);
  ruby_whisper *rw;
  TypedData_Get_Struct(rwm->context, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_text_layer(rw->context));
}

/*
 * call-seq:
 *   n_mels -> Integer
 */
static VALUE
ruby_whisper_model_n_mels(VALUE self)
{
  ruby_whisper_model *rwm;
  TypedData_Get_Struct(self, ruby_whisper_model, &rb_whisper_model_type, rwm);
  ruby_whisper *rw;
  TypedData_Get_Struct(rwm->context, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_n_mels(rw->context));
}

/*
 * call-seq:
 *   ftype -> Integer
 */
static VALUE
ruby_whisper_model_ftype(VALUE self)
{
  ruby_whisper_model *rwm;
  TypedData_Get_Struct(self, ruby_whisper_model, &rb_whisper_model_type, rwm);
  ruby_whisper *rw;
  TypedData_Get_Struct(rwm->context, ruby_whisper, &ruby_whisper_type, rw);
  return INT2NUM(whisper_model_ftype(rw->context));
}

/*
 * call-seq:
 *   type -> String
 */
static VALUE
ruby_whisper_model_type(VALUE self)
{
  ruby_whisper_model *rwm;
  TypedData_Get_Struct(self, ruby_whisper_model, &rb_whisper_model_type, rwm);
  ruby_whisper *rw;
  TypedData_Get_Struct(rwm->context, ruby_whisper, &ruby_whisper_type, rw);
  return rb_str_new2(whisper_model_type_readable(rw->context));
}

void
init_ruby_whisper_model(VALUE *mWhisper)
{
  cModel = rb_define_class_under(*mWhisper, "Model", rb_cObject);

  rb_define_alloc_func(cModel, ruby_whisper_model_allocate);
  rb_define_method(cModel, "n_vocab", ruby_whisper_model_n_vocab, 0);
  rb_define_method(cModel, "n_audio_ctx", ruby_whisper_model_n_audio_ctx, 0);
  rb_define_method(cModel, "n_audio_state", ruby_whisper_model_n_audio_state, 0);
  rb_define_method(cModel, "n_audio_head", ruby_whisper_model_n_audio_head, 0);
  rb_define_method(cModel, "n_audio_layer", ruby_whisper_model_n_audio_layer, 0);
  rb_define_method(cModel, "n_text_ctx", ruby_whisper_model_n_text_ctx, 0);
  rb_define_method(cModel, "n_text_state", ruby_whisper_model_n_text_state, 0);
  rb_define_method(cModel, "n_text_head", ruby_whisper_model_n_text_head, 0);
  rb_define_method(cModel, "n_text_layer", ruby_whisper_model_n_text_layer, 0);
  rb_define_method(cModel, "n_mels", ruby_whisper_model_n_mels, 0);
  rb_define_method(cModel, "ftype", ruby_whisper_model_ftype, 0);
  rb_define_method(cModel, "type", ruby_whisper_model_type, 0);
}
