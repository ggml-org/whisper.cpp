#ifndef RUBY_WHISPER_LOG_SETTABLE_H
#define RUBY_WHISPER_LOG_SETTABLE_H

#define LOG_SETTABLE_SETUP(log_queue, mod, log_set) \
  static VALUE \
  ruby_whisper_##log_queue##_s_drain_logs(VALUE self) \
  { \
    return ruby_whisper_log_queue_drain(&log_queue); \
  } \
  static void \
  ruby_whisper_##log_queue##_log_callback(enum ggml_log_level level, const char *text, void *user_data) \
  { \
    ruby_whisper_log_queue_enqueue(&log_queue, level, text);   \
  } \
  static VALUE \
  ruby_whisper_##log_queue##_s_log_set(VALUE self, VALUE log_callback, VALUE user_data) \
  { \
    rb_iv_set(self, "@log_callback", log_callback); \
    rb_iv_set(self, "@log_callback_user_data", user_data); \
    if (NIL_P(log_callback)) { \
      log_set(NULL, NULL); \
    } else { \
      ruby_whisper_log_queue_open(&log_queue); \
      rb_funcall((mod), id_start_log_callback_thread, 0); \
      log_set(ruby_whisper_##log_queue##_log_callback, NULL); \
    } \
    return Qnil; \
  } \
  static void \
  ruby_whisper_##log_queue##_end_proc(VALUE args) \
  { \
    ruby_whisper_log_queue_close(&log_queue); \
    VALUE log_callback_thread = rb_ivar_get(mod, id_log_callback_thread); \
    if (!NIL_P(log_callback_thread) && RTEST(rb_funcall(log_callback_thread, id_alive_p, 0))) { \
      rb_funcall(log_callback_thread, id_join, 0); \
    } \
  }

#define LOG_SETTABLE_INIT(log_queue, mod) \
  ruby_whisper_log_queue_initialize(&log_queue); \
  rb_define_singleton_method(mod, "drain_logs", ruby_whisper_##log_queue##_s_drain_logs, 0); \
  rb_define_singleton_method(mod, "log_set", ruby_whisper_##log_queue##_s_log_set, 2); \
  rb_set_end_proc(ruby_whisper_##log_queue##_end_proc, Qnil); \
  rb_extend_object(mod, mLogSettable); \
  rb_funcall(mLogSettable, id_extended, 1, mod);

#endif
