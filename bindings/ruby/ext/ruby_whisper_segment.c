#include <ruby.h>
#include "ruby_whisper.h"

#define N_KEY_NAMES 5

static VALUE sym_start_time;
static VALUE sym_end_time;
static VALUE sym_text;
static VALUE sym_no_speech_prob;
static VALUE sym_speaker_turn_next;
static VALUE key_names;

extern const rb_data_type_t ruby_whisper_type;

extern VALUE cSegment;

static void
rb_whisper_segment_mark(void *p)
{
  ruby_whisper_segment *rws = (ruby_whisper_segment *)p;
  rb_gc_mark(rws->context);
}

static size_t
ruby_whisper_segment_memsize(const void *p)
{
  const ruby_whisper_segment *rws = (const ruby_whisper_segment *)p;
  size_t size = sizeof(rws);
  if (!rws) {
    return 0;
  }
  return size;
}

static const rb_data_type_t ruby_whisper_segment_type = {
  "ruby_whisper_segment",
  {rb_whisper_segment_mark, RUBY_DEFAULT_FREE, ruby_whisper_segment_memsize,},
  0, 0,
  0
};

VALUE
ruby_whisper_segment_allocate(VALUE klass)
{
  ruby_whisper_segment *rws;
  return TypedData_Make_Struct(klass, ruby_whisper_segment, &ruby_whisper_segment_type, rws);
}

VALUE
rb_whisper_segment_s_new(VALUE context, int index)
{
  ruby_whisper_segment *rws;
  const VALUE segment = ruby_whisper_segment_allocate(cSegment);
  TypedData_Get_Struct(segment, ruby_whisper_segment, &ruby_whisper_segment_type, rws);
  rws->context = context;
  rws->index = index;
  return segment;
};

/*
 * Start time in milliseconds.
 *
 * call-seq:
 *   start_time -> Integer
 */
static VALUE
ruby_whisper_segment_get_start_time(VALUE self)
{
  ruby_whisper_segment *rws;
  TypedData_Get_Struct(self, ruby_whisper_segment, &ruby_whisper_segment_type, rws);
  ruby_whisper *rw;
  TypedData_Get_Struct(rws->context, ruby_whisper, &ruby_whisper_type, rw);
  const int64_t t0 = whisper_full_get_segment_t0(rw->context, rws->index);
  // able to multiply 10 without overflow because to_timestamp() in whisper.cpp does it
  return LONG2NUM(t0 * 10);
}

/*
 * End time in milliseconds.
 *
 * call-seq:
 *   end_time -> Integer
 */
static VALUE
ruby_whisper_segment_get_end_time(VALUE self)
{
  ruby_whisper_segment *rws;
  TypedData_Get_Struct(self, ruby_whisper_segment, &ruby_whisper_segment_type, rws);
  ruby_whisper *rw;
  TypedData_Get_Struct(rws->context, ruby_whisper, &ruby_whisper_type, rw);
  const int64_t t1 = whisper_full_get_segment_t1(rw->context, rws->index);
  // able to multiply 10 without overflow because to_timestamp() in whisper.cpp does it
  return LONG2NUM(t1 * 10);
}

/*
 * Whether the next segment is predicted as a speaker turn.
 *
 * call-seq:
 *   speaker_turn_next? -> bool
 */
static VALUE
ruby_whisper_segment_get_speaker_turn_next(VALUE self)
{
  ruby_whisper_segment *rws;
  TypedData_Get_Struct(self, ruby_whisper_segment, &ruby_whisper_segment_type, rws);
  ruby_whisper *rw;
  TypedData_Get_Struct(rws->context, ruby_whisper, &ruby_whisper_type, rw);
  return whisper_full_get_segment_speaker_turn_next(rw->context, rws->index) ? Qtrue : Qfalse;
}

/*
 * call-seq:
 *   text -> String
 */
static VALUE
ruby_whisper_segment_get_text(VALUE self)
{
  ruby_whisper_segment *rws;
  TypedData_Get_Struct(self, ruby_whisper_segment, &ruby_whisper_segment_type, rws);
  ruby_whisper *rw;
  TypedData_Get_Struct(rws->context, ruby_whisper, &ruby_whisper_type, rw);
  const char * text = whisper_full_get_segment_text(rw->context, rws->index);
  return rb_str_new2(text);
}

/*
 * call-seq:
 *   no_speech_prob -> Float
 */
static VALUE
ruby_whisper_segment_get_no_speech_prob(VALUE self)
{
  ruby_whisper_segment *rws;
  TypedData_Get_Struct(self, ruby_whisper_segment, &ruby_whisper_segment_type, rws);
  ruby_whisper *rw;
  TypedData_Get_Struct(rws->context, ruby_whisper, &ruby_whisper_type, rw);
  return DBL2NUM(whisper_full_get_segment_no_speech_prob(rw->context, rws->index));
}

/*
 * call-seq:
 *   deconstruct_keys(keys) -> hash
 *
 *  Possible keys: :start_time, :end_time, :text, :no_speech_prob, :speaker_turn_next
 *
 *   whisper.each_segment do |segment|
 *     segment => {start_time:, end_time:, text:, no_speech_prob:, speaker_turn_next:}
 *
 *     puts "[#{start_time} --> #{end_time}] #{text} (no speech prob: #{no_speech_prob}#{speaker_turn_next ? ', speaker turns next' : ''})"
 *   end
 */
static VALUE
ruby_whisper_segment_deconstruct_keys(VALUE self, VALUE keys)
{
  ruby_whisper_segment *rws;
  TypedData_Get_Struct(self, ruby_whisper_segment, &ruby_whisper_segment_type, rws);
  ruby_whisper *rw;
  TypedData_Get_Struct(rws->context, ruby_whisper, &ruby_whisper_type, rw);

  VALUE hash = rb_hash_new();
  long n_keys;
  if (NIL_P(keys)) {
    keys = key_names;
    n_keys = N_KEY_NAMES;
  } else {
    n_keys = RARRAY_LEN(keys);
    if (n_keys > N_KEY_NAMES) {
      return hash;
    }
  }
  for (int i = 0; i < n_keys; i++) {
    VALUE key = rb_ary_entry(keys, i);
    if (key == sym_start_time) {
      rb_hash_aset(hash, key, ruby_whisper_segment_get_start_time(self));
    }
    if (key == sym_end_time) {
      rb_hash_aset(hash, key, ruby_whisper_segment_get_end_time(self));
    }
    if (key == sym_text) {
      rb_hash_aset(hash, key, ruby_whisper_segment_get_text(self));
    }
    if (key == sym_no_speech_prob) {
      rb_hash_aset(hash, key, ruby_whisper_segment_get_no_speech_prob(self));
    }
    if (key == sym_speaker_turn_next) {
      rb_hash_aset(hash, key, ruby_whisper_segment_get_speaker_turn_next(self));
    }
  }

  return hash;
}

void
init_ruby_whisper_segment(VALUE *mWhisper, VALUE *cContext)
{
  cSegment  = rb_define_class_under(*mWhisper, "Segment", rb_cObject);

  sym_start_time = ID2SYM(rb_intern("start_time"));
  sym_end_time = ID2SYM(rb_intern("end_time"));
  sym_text = ID2SYM(rb_intern("text"));
  sym_no_speech_prob = ID2SYM(rb_intern("no_speech_prob"));
  sym_speaker_turn_next = ID2SYM(rb_intern("speaker_turn_next"));
  key_names = rb_ary_new3(
    N_KEY_NAMES,
    sym_start_time,
    sym_end_time,
    sym_text,
    sym_no_speech_prob,
    sym_speaker_turn_next
  );

  rb_define_alloc_func(cSegment, ruby_whisper_segment_allocate);
  rb_define_method(cSegment, "start_time", ruby_whisper_segment_get_start_time, 0);
  rb_define_method(cSegment, "end_time", ruby_whisper_segment_get_end_time, 0);
  rb_define_method(cSegment, "speaker_turn_next?", ruby_whisper_segment_get_speaker_turn_next, 0);
  rb_define_method(cSegment, "text", ruby_whisper_segment_get_text, 0);
  rb_define_method(cSegment, "no_speech_prob", ruby_whisper_segment_get_no_speech_prob, 0);
  rb_define_method(cSegment, "deconstruct_keys", ruby_whisper_segment_deconstruct_keys, 1);
}
