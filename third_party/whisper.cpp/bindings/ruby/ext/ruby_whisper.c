#include "ruby_whisper.h"

VALUE mWhisper;
VALUE mLogSettable;
VALUE mVAD;
VALUE mParakeet;
VALUE cContext;
VALUE cParams;
VALUE cVADContext;
VALUE cVADParams;
VALUE cVADSegments;
VALUE cVADSegment;
VALUE cParakeetContext;
VALUE cParakeetContextParams;
VALUE cParakeetParams;
VALUE cParakeetSegment;
VALUE cParakeetModel;
VALUE eError;

VALUE cSegment;
VALUE cToken;
VALUE cModel;

VALUE mOutputContext;
VALUE mOutputSegment;

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
ID id_extended;
ID id_start_log_callback_thread;
ID id_log_callback_thread;
ID id_alive_p;
ID id_join;

// High level API
extern VALUE ruby_whisper_segment_allocate(VALUE klass);

extern VALUE init_ruby_whisper_context(VALUE *mWhisper);
extern void init_ruby_whisper_context_params(VALUE *cContext);
extern void init_ruby_whisper_params(VALUE *mWhisper);
extern void init_ruby_whisper_error(VALUE *mWhisper);
extern void init_ruby_whisper_segment(VALUE *mWhisper);
extern void init_ruby_whisper_token(VALUE *mWhisper);
extern void init_ruby_whisper_model(VALUE *mWhisper);
extern void init_ruby_whisper_vad_params(VALUE *mVAD);
extern void init_ruby_whisper_vad_context(VALUE *mVAD);
extern void init_ruby_whisper_vad_segment(VALUE *mVAD);
extern void init_ruby_whisper_vad_segments(VALUE *mVAD);
extern void init_ruby_whisper_parakeet(VALUE *mWhisper);
extern void register_callbacks(ruby_whisper_params *rwp, VALUE *context);

static ruby_whisper_log_queue whisper_log_queue;

LOG_SETTABLE_SETUP(whisper_log_queue, mWhisper, whisper_log_set)

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
  id_extended = rb_intern("extended");
  id_start_log_callback_thread = rb_intern("start_log_callback_thread");
  id_log_callback_thread = rb_intern("@log_callback_thread");
  id_alive_p = rb_intern("alive?");
  id_join = rb_intern("join");

  mWhisper = rb_define_module("Whisper");
  rb_require("whisper/log_settable");
  mLogSettable = rb_path2class("Whisper::LogSettable");
  mVAD = rb_define_module_under(mWhisper, "VAD");
  rb_require("whisper/output");
  mOutputContext = rb_path2class("Whisper::Output::Context");
  mOutputSegment = rb_path2class("Whisper::Output::Segment");

  rb_define_const(mWhisper, "VERSION", rb_str_new2(whisper_version()));
  rb_define_const(mWhisper, "LOG_LEVEL_NONE", INT2NUM(GGML_LOG_LEVEL_NONE));
  rb_define_const(mWhisper, "LOG_LEVEL_INFO", INT2NUM(GGML_LOG_LEVEL_INFO));
  rb_define_const(mWhisper, "LOG_LEVEL_WARN", INT2NUM(GGML_LOG_LEVEL_WARN));
  rb_define_const(mWhisper, "LOG_LEVEL_ERROR", INT2NUM(GGML_LOG_LEVEL_ERROR));
  rb_define_const(mWhisper, "LOG_LEVEL_DEBUG", INT2NUM(GGML_LOG_LEVEL_DEBUG));
  rb_define_const(mWhisper, "LOG_LEVEL_CONT", INT2NUM(GGML_LOG_LEVEL_CONT));

  rb_define_const(mWhisper, "AHEADS_NONE", INT2NUM(WHISPER_AHEADS_NONE));
  rb_define_const(mWhisper, "AHEADS_N_TOP_MOST", INT2NUM(WHISPER_AHEADS_N_TOP_MOST));
  rb_define_const(mWhisper, "AHEADS_CUSTOM", INT2NUM(WHISPER_AHEADS_CUSTOM));
  rb_define_const(mWhisper, "AHEADS_TINY_EN", INT2NUM(WHISPER_AHEADS_TINY_EN));
  rb_define_const(mWhisper, "AHEADS_TINY", INT2NUM(WHISPER_AHEADS_TINY));
  rb_define_const(mWhisper, "AHEADS_BASE_EN", INT2NUM(WHISPER_AHEADS_BASE_EN));
  rb_define_const(mWhisper, "AHEADS_BASE", INT2NUM(WHISPER_AHEADS_BASE));
  rb_define_const(mWhisper, "AHEADS_SMALL_EN", INT2NUM(WHISPER_AHEADS_SMALL_EN));
  rb_define_const(mWhisper, "AHEADS_SMALL", INT2NUM(WHISPER_AHEADS_SMALL));
  rb_define_const(mWhisper, "AHEADS_MEDIUM_EN", INT2NUM(WHISPER_AHEADS_MEDIUM_EN));
  rb_define_const(mWhisper, "AHEADS_MEDIUM", INT2NUM(WHISPER_AHEADS_MEDIUM));
  rb_define_const(mWhisper, "AHEADS_LARGE_V1", INT2NUM(WHISPER_AHEADS_LARGE_V1));
  rb_define_const(mWhisper, "AHEADS_LARGE_V2", INT2NUM(WHISPER_AHEADS_LARGE_V2));
  rb_define_const(mWhisper, "AHEADS_LARGE_V3", INT2NUM(WHISPER_AHEADS_LARGE_V3));
  rb_define_const(mWhisper, "AHEADS_LARGE_V3_TURBO", INT2NUM(WHISPER_AHEADS_LARGE_V3_TURBO));

  rb_define_singleton_method(mWhisper, "lang_max_id", ruby_whisper_s_lang_max_id, 0);
  rb_define_singleton_method(mWhisper, "lang_id", ruby_whisper_s_lang_id, 1);
  rb_define_singleton_method(mWhisper, "lang_str", ruby_whisper_s_lang_str, 1);
  rb_define_singleton_method(mWhisper, "lang_str_full", ruby_whisper_s_lang_str_full, 1);
  rb_define_singleton_method(mWhisper, "system_info_str", ruby_whisper_s_system_info_str, 0);

  LOG_SETTABLE_INIT(whisper_log_queue, mWhisper)

  cContext = init_ruby_whisper_context(&mWhisper);
  init_ruby_whisper_context_params(&cContext);
  init_ruby_whisper_params(&mWhisper);
  init_ruby_whisper_error(&mWhisper);
  init_ruby_whisper_segment(&mWhisper);
  init_ruby_whisper_token(&mWhisper);
  init_ruby_whisper_model(&mWhisper);
  init_ruby_whisper_vad_params(&mVAD);
  init_ruby_whisper_vad_segment(&mVAD);
  init_ruby_whisper_vad_segments(&mVAD);
  init_ruby_whisper_vad_context(&mVAD);
  init_ruby_whisper_parakeet(&mWhisper);

  rb_require("whisper/model/uri");

  rb_include_module(cContext, mOutputContext);
  rb_include_module(cSegment, mOutputSegment);
}
