#include "ruby_whisper.h"

#define LOG_QUEUE_CAPACITY 256
#define LOG_DEFAULT_CAPACITY 1024

void
ruby_whisper_log_queue_initialize(ruby_whisper_log_queue *log_queue)
{
  rb_nativethread_lock_initialize(&log_queue->lock);
  rb_native_cond_initialize(&log_queue->cond);
  log_queue->head = 0;
  log_queue->tail = 0;
  log_queue->size = 0;
  log_queue->is_open = true;
  log_queue->logs = ALLOC_N(ruby_whisper_log, LOG_QUEUE_CAPACITY);
  for (size_t i = 0; i < LOG_QUEUE_CAPACITY; i++) {
    // we cannot call Ruby API like ALLOC_N because this slot may be realloced without GVL
    // this doesn't be freed because log queue lives until the end of process
    char *slot = malloc(sizeof(char) * LOG_QUEUE_CAPACITY);
    if (!slot) {
      rb_raise(rb_eRuntimeError, "Could not allocate memory for log text");
    }
    ruby_whisper_log log = {
      0,
      slot,
      0,
      LOG_QUEUE_CAPACITY,
    };
    log_queue->logs[i] = log;
  }
}

void
ruby_whisper_log_queue_open(ruby_whisper_log_queue *log_queue)
{
  rb_nativethread_lock_lock(&log_queue->lock);

  log_queue->is_open = true;

  rb_native_cond_signal(&log_queue->cond);

  rb_nativethread_lock_unlock(&log_queue->lock);
}

void
ruby_whisper_log_queue_close(ruby_whisper_log_queue *log_queue)
{
  rb_nativethread_lock_lock(&log_queue->lock);

  log_queue->is_open = false;
  rb_native_cond_broadcast(&log_queue->cond);

  rb_nativethread_lock_unlock(&log_queue->lock);
}

static size_t
calc_enough_cap(size_t len)
{
  size_t quot = len / LOG_DEFAULT_CAPACITY;
  size_t rem = len % LOG_DEFAULT_CAPACITY;

  return sizeof(char) * (rem == 0 ? quot : quot + 1) * LOG_DEFAULT_CAPACITY;
}

void
ruby_whisper_log_queue_enqueue(ruby_whisper_log_queue *log_queue, enum ggml_log_level level, const char *text)
{
  rb_nativethread_lock_lock(&log_queue->lock);

  if (!log_queue->is_open) {
    rb_nativethread_lock_unlock(&log_queue->lock);
    return;
  }

  size_t len = strlen(text);
  ruby_whisper_log *log = &log_queue->logs[log_queue->head];
  if (len > log->capacity) {
    size_t new_cap = calc_enough_cap(len);
    // we cannot call Ruby API like REALLOC_N because this function is called without GVL
    char *slot = realloc(log->text, new_cap);
    if (!slot) {
      rb_nativethread_lock_unlock(&log_queue->lock);
      return;
    }
    log->text = slot;
    log->capacity = new_cap;
  }
  // we cannot call Ruby API like MEMCPY because this function is called without GVL
  memcpy(log->text, text, sizeof(char) * len);
  log->length = len;
  log->level = level;
  log_queue->head = (log_queue->head + 1) % LOG_QUEUE_CAPACITY;
  bool is_full = log_queue->size >= LOG_QUEUE_CAPACITY;
  log_queue->size = is_full ? LOG_QUEUE_CAPACITY : log_queue->size + 1;
  if (is_full) {
    log_queue->tail = log_queue->head;
  }

  rb_native_cond_signal(&log_queue->cond);
  rb_nativethread_lock_unlock(&log_queue->lock);
}

static void*
ruby_whisper_log_queue_wait(void *args)
{
  ruby_whisper_log_queue *log_queue = (ruby_whisper_log_queue *)args;

  rb_native_cond_wait(&log_queue->cond, &log_queue->lock);
  rb_nativethread_lock_unlock(&log_queue->lock);

  return NULL;
}

static void
ruby_whisper_log_queue_wait_ubf(void *args)
{
  ruby_whisper_log_queue *log_queue = (ruby_whisper_log_queue *)args;

  rb_native_cond_broadcast(&log_queue->cond);
}

typedef struct {
  enum ggml_log_level level;
  size_t length;
  char *text;
} log_snapshot;

VALUE
ruby_whisper_log_queue_drain(ruby_whisper_log_queue *log_queue)
{
  log_snapshot logs[LOG_QUEUE_CAPACITY];

  rb_nativethread_lock_lock(&log_queue->lock);

  while (log_queue->size == 0 && log_queue->is_open) {
    rb_thread_call_without_gvl(ruby_whisper_log_queue_wait, (void *)log_queue, ruby_whisper_log_queue_wait_ubf, (void *)log_queue);
    rb_nativethread_lock_lock(&log_queue->lock);
  }

  if (log_queue->size == 0 && !log_queue->is_open) {
    rb_native_cond_broadcast(&log_queue->cond);
    rb_nativethread_lock_unlock(&log_queue->lock);
    return Qnil;
  }

  size_t size = log_queue->size;
  ruby_whisper_log *log;
  size_t i;
  for (i = 0; i < size; i++) {
    log = &log_queue->logs[(log_queue->tail + i) % LOG_QUEUE_CAPACITY];
    logs[i].level = log->level;
    logs[i].length = log->length;
    char *text = malloc(log->length);
    if (!text) {
      logs[i].text = NULL;
      continue;
    }
    logs[i].text = text;
    memcpy(logs[i].text, log->text, log->length);
  }
  log_queue->size = 0;
  log_queue->tail = log_queue->head;

  rb_native_cond_signal(&log_queue->cond);

  rb_nativethread_lock_unlock(&log_queue->lock);

  VALUE rb_logs = rb_ary_new2(size);
  VALUE rb_text;
  for (i = 0; i < size; i++) {
    if (!logs[i].text) {
      continue;
    }
    rb_text = rb_str_new(logs[i].text, logs[i].length);
    free(logs[i].text);
    rb_ary_push(rb_logs, rb_ary_new3(2, INT2NUM(logs[i].level), rb_text));
  }

  return rb_logs;
}
