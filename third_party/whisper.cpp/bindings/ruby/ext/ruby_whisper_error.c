#include <ruby.h>

extern VALUE eError;

VALUE ruby_whisper_error_initialize(VALUE self, VALUE code)
{
  const int c_code = NUM2INT(code);
  const char *raw_message;
  switch (c_code) {
  case -2:
    raw_message = "failed to compute log mel spectrogram";
    break;
  case -3:
    raw_message = "failed to auto-detect language";
    break;
  case -4:
    raw_message = "too many decoders requested";
    break;
  case -5:
    raw_message = "audio_ctx is larger than the maximum allowed";
    break;
  case -6:
    raw_message = "failed to encode";
    break;
  case -7:
    raw_message = "whisper_kv_cache_init() failed for self-attention cache";
    break;
  case -8:
    raw_message = "failed to decode";
    break;
  case -9:
    raw_message = "failed to decode";
    break;
  default:
    raw_message = "unknown error";
    break;
  }
  const VALUE message = rb_str_new2(raw_message);
  rb_call_super(1, &message);
  rb_iv_set(self, "@code", code);

  return self;
}

void
init_ruby_whisper_error(VALUE *mWhisper)
{
  eError = rb_define_class_under(*mWhisper, "Error", rb_eStandardError);

  rb_define_attr(eError, "code", true, false);
  rb_define_method(eError, "initialize", ruby_whisper_error_initialize, 1);
}
