#include "ruby_whisper.h"
#include <stdio.h>
#include <unistd.h>

extern VALUE mParakeet;
extern VALUE mLogSettable;
extern ID id_extended;
extern ID id_log_callback_thread;
extern ID id_start_log_callback_thread;
extern ID id_alive;
extern ID id_join;

extern void ruby_whisper_log_queue_initialize(ruby_whisper_log_queue *log_queue);
extern void ruby_whisper_log_queue_open(ruby_whisper_log_queue *log_queue);
extern void ruby_whisper_log_queue_close(ruby_whisper_log_queue *log_queue);
extern void ruby_whisper_log_queue_enqueue(ruby_whisper_log_queue *log_queue, enum ggml_log_level level, const char *text);
extern VALUE ruby_whisper_log_queue_drain(ruby_whisper_log_queue *log_queue);

static ruby_whisper_log_queue parakeet_log_queue;

static VALUE
ruby_whisper_parakeet_s_drain_logs(VALUE self)
{
  return ruby_whisper_log_queue_drain(&parakeet_log_queue);
}

static void
ruby_whisper_parakeet_log_callback(enum ggml_log_level level, const char *text, void *user_data)
{
  ruby_whisper_log_queue_enqueue(&parakeet_log_queue, level, text);
}

static VALUE
ruby_whisper_parakeet_s_log_set(VALUE self, VALUE log_callback, VALUE user_data)
{
  rb_iv_set(self, "@log_callback", log_callback);
  rb_iv_set(self, "@log_callback_user_data", user_data);
  if (NIL_P(log_callback)) {
    parakeet_log_set(NULL, NULL);
  } else {
    ruby_whisper_log_queue_open(&parakeet_log_queue);
    rb_funcall(mParakeet, id_start_log_callback_thread, 0);
    parakeet_log_set(ruby_whisper_parakeet_log_callback, NULL);
  }

  return Qnil;
}

static void
ruby_whisper_parakeet_end_proc(VALUE args)
{
  ruby_whisper_log_queue_close(&parakeet_log_queue);

  VALUE log_callback_thread = rb_ivar_get(mParakeet, id_log_callback_thread);
  if (!NIL_P(log_callback_thread) && RTEST(rb_funcall(log_callback_thread, id_alive, 0))) {
    rb_funcall(log_callback_thread, id_join, 0);
  }
}

void
init_ruby_whisper_parakeet()
{
  ruby_whisper_log_queue_initialize(&parakeet_log_queue);

  rb_define_singleton_method(mParakeet, "log_set", ruby_whisper_parakeet_s_log_set, 2);
  rb_define_private_method(rb_singleton_class(mParakeet), "drain_logs", ruby_whisper_parakeet_s_drain_logs, 0);

  rb_set_end_proc(ruby_whisper_parakeet_end_proc, Qnil);
  rb_extend_object(mParakeet, mLogSettable);
  rb_funcall(mLogSettable, id_extended, 1, mParakeet);
}
