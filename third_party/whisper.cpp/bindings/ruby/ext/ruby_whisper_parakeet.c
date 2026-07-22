#include "ruby_whisper.h"
#include <stdio.h>
#include <unistd.h>

extern VALUE mParakeet;
extern VALUE mLogSettable;
extern VALUE cParakeetContext;
extern VALUE cParakeetSegment;
extern VALUE mOutputContext;
extern VALUE mOutputSegment;

extern void init_ruby_whisper_parakeet_params(VALUE *mParakeet);
extern void init_ruby_whisper_parakeet_token(VALUE *mParakeet);
extern void init_ruby_whisper_parakeet_segment(VALUE *mParakeet);
extern VALUE init_ruby_whisper_parakeet_context(VALUE *mParakeet);
extern void init_ruby_whisper_parakeet_context_params(VALUE *cParakeetContext);
extern void init_ruby_whisper_parakeet_model(VALUE *mParakeet);

static ruby_whisper_log_queue parakeet_log_queue;

LOG_SETTABLE_SETUP(parakeet_log_queue, mParakeet, parakeet_log_set)

static VALUE
ruby_whisper_parakeet_s_system_info_str(VALUE self)
{
  return rb_str_new2(parakeet_print_system_info());
}

void
init_ruby_whisper_parakeet(VALUE *mWhisper)
{
  mParakeet = rb_define_module_under(*mWhisper, "Parakeet");

  rb_define_const(mParakeet, "VERSION", rb_str_new2(parakeet_version()));

  LOG_SETTABLE_INIT(parakeet_log_queue, mParakeet)

  rb_define_singleton_method(mParakeet, "system_info_str", ruby_whisper_parakeet_s_system_info_str, 0);

  init_ruby_whisper_parakeet_params(&mParakeet);
  init_ruby_whisper_parakeet_token(&mParakeet);
  init_ruby_whisper_parakeet_segment(&mParakeet);
  cParakeetContext = init_ruby_whisper_parakeet_context(&mParakeet);
  init_ruby_whisper_parakeet_context_params(&cParakeetContext);
  init_ruby_whisper_parakeet_model(&mParakeet);

  rb_include_module(cParakeetContext, mOutputContext);
  rb_include_module(cParakeetSegment, mOutputSegment);
}
