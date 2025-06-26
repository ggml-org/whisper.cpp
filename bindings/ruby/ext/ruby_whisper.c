#include <ruby.h>
#include <ruby/memory_view.h>
#include "ruby_whisper.h"

VALUE mWhisper;
VALUE mVAD;
VALUE cContext;
VALUE cParams;
VALUE cVADParams;
VALUE eError;

VALUE cSegment;
VALUE cModel;

ID id_to_s;
ID id_call;
ID id___method__;
ID id_to_enum;
ID id_length;
ID id_next;
ID id_new;
ID id_to_path;
ID id_URI;
ID id_pre_converted_models;
ID id_coreml_compiled_models;
ID id_cache;
ID id_n_processors;

static bool is_log_callback_finalized = false;

// High level API
extern VALUE ruby_whisper_segment_allocate(VALUE klass);

extern void init_ruby_whisper_context(VALUE *mWhisper);
extern void init_ruby_whisper_params(VALUE *mWhisper);
extern void init_ruby_whisper_error(VALUE *mWhisper);
extern void init_ruby_whisper_segment(VALUE *mWhisper, VALUE *cSegment);
extern void init_ruby_whisper_model(VALUE *mWhisper);
extern void init_ruby_whisper_vad_params(VALUE *mVAD);
extern void register_callbacks(ruby_whisper_params *rwp, VALUE *context);

/*
 * call-seq:
 *   lang_max_id -> Integer
 */
static VALUE ruby_whisper_s_lang_max_id(VALUE self) {
  return INT2NUM(whisper_lang_max_id());
}

/*
 * call-seq:
 *   lang_id(lang_name) -> Integer
 */
static VALUE ruby_whisper_s_lang_id(VALUE self, VALUE lang) {
  const char * lang_str = StringValueCStr(lang);
  const int id = whisper_lang_id(lang_str);
  if (-1 == id) {
    rb_raise(rb_eArgError, "language not found: %s", lang_str);
  }
  return INT2NUM(id);
}

/*
 * call-seq:
 *   lang_str(lang_id) -> String
 */
static VALUE ruby_whisper_s_lang_str(VALUE self, VALUE id) {
  const int lang_id = NUM2INT(id);
  const char * str = whisper_lang_str(lang_id);
  if (NULL == str) {
    rb_raise(rb_eIndexError, "id %d outside of language id", lang_id);
  }
  return rb_str_new2(str);
}

/*
 * call-seq:
 *   lang_str(lang_id) -> String
 */
static VALUE ruby_whisper_s_lang_str_full(VALUE self, VALUE id) {
  const int lang_id = NUM2INT(id);
  const char * str_full = whisper_lang_str_full(lang_id);
  if (NULL == str_full) {
    rb_raise(rb_eIndexError, "id %d outside of language id", lang_id);
  }
  return rb_str_new2(str_full);
}

/*
 * call-seq:
 *   system_info_str -> String
 */
static VALUE ruby_whisper_s_system_info_str(VALUE self) {
  return rb_str_new2(whisper_print_system_info());
}

static VALUE ruby_whisper_s_finalize_log_callback(VALUE self, VALUE id) {
  is_log_callback_finalized = true;
  return Qnil;
}

static void
ruby_whisper_log_callback(enum ggml_log_level level, const char * buffer, void * user_data) {
  if (is_log_callback_finalized) {
    return;
  }
  VALUE log_callback = rb_iv_get(mWhisper, "log_callback");
  VALUE udata = rb_iv_get(mWhisper, "user_data");
  rb_funcall(log_callback, id_call, 3, INT2NUM(level), rb_str_new2(buffer), udata);
}

/*
 * call-seq:
 *   log_set ->(level, buffer, user_data) { ... }, user_data -> nil
 */
static VALUE ruby_whisper_s_log_set(VALUE self, VALUE log_callback, VALUE user_data) {
  VALUE old_callback = rb_iv_get(self, "log_callback");
  if (!NIL_P(old_callback)) {
    rb_undefine_finalizer(old_callback);
  }

  rb_iv_set(self, "log_callback", log_callback);
  rb_iv_set(self, "user_data", user_data);

  VALUE finalize_log_callback = rb_funcall(mWhisper, rb_intern("method"), 1, rb_str_new2("finalize_log_callback"));
  rb_define_finalizer(log_callback, finalize_log_callback);

  whisper_log_set(ruby_whisper_log_callback, NULL);

  return Qnil;
}

void Init_whisper() {
  id_to_s = rb_intern("to_s");
  id_call = rb_intern("call");
  id___method__ = rb_intern("__method__");
  id_to_enum = rb_intern("to_enum");
  id_length = rb_intern("length");
  id_next = rb_intern("next");
  id_new = rb_intern("new");
  id_to_path = rb_intern("to_path");
  id_URI = rb_intern("URI");
  id_pre_converted_models = rb_intern("pre_converted_models");
  id_coreml_compiled_models = rb_intern("coreml_compiled_models");
  id_cache = rb_intern("cache");
  id_n_processors = rb_intern("n_processors");

  mWhisper = rb_define_module("Whisper");
  mVAD = rb_define_module_under(mWhisper, "VAD");

  rb_define_const(mWhisper, "VERSION", rb_str_new2(whisper_version()));
  rb_define_const(mWhisper, "LOG_LEVEL_NONE", INT2NUM(GGML_LOG_LEVEL_NONE));
  rb_define_const(mWhisper, "LOG_LEVEL_INFO", INT2NUM(GGML_LOG_LEVEL_INFO));
  rb_define_const(mWhisper, "LOG_LEVEL_WARN", INT2NUM(GGML_LOG_LEVEL_WARN));
  rb_define_const(mWhisper, "LOG_LEVEL_ERROR", INT2NUM(GGML_LOG_LEVEL_ERROR));
  rb_define_const(mWhisper, "LOG_LEVEL_DEBUG", INT2NUM(GGML_LOG_LEVEL_DEBUG));
  rb_define_const(mWhisper, "LOG_LEVEL_CONT", INT2NUM(GGML_LOG_LEVEL_CONT));

  rb_define_singleton_method(mWhisper, "lang_max_id", ruby_whisper_s_lang_max_id, 0);
  rb_define_singleton_method(mWhisper, "lang_id", ruby_whisper_s_lang_id, 1);
  rb_define_singleton_method(mWhisper, "lang_str", ruby_whisper_s_lang_str, 1);
  rb_define_singleton_method(mWhisper, "lang_str_full", ruby_whisper_s_lang_str_full, 1);
  rb_define_singleton_method(mWhisper, "system_info_str", ruby_whisper_s_system_info_str, 0);
  rb_define_singleton_method(mWhisper, "log_set", ruby_whisper_s_log_set, 2);
  rb_define_private_method(rb_singleton_class(mWhisper), "finalize_log_callback", ruby_whisper_s_finalize_log_callback, 1);

  init_ruby_whisper_context(&mWhisper);
  init_ruby_whisper_params(&mWhisper);
  init_ruby_whisper_error(&mWhisper);
  init_ruby_whisper_segment(&mWhisper, &cContext);
  init_ruby_whisper_model(&mWhisper);
  init_ruby_whisper_vad_params(&mVAD);

  rb_require("whisper/context");
  rb_require("whisper/segment");
  rb_require("whisper/model/uri");
}
