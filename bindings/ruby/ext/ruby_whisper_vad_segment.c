#include <ruby.h>
#include "ruby_whisper.h"

#define N_KEY_NAMES 2

extern VALUE cVADSegment;

extern const rb_data_type_t ruby_whisper_vad_segments_type;

static VALUE sym_start_time;
static VALUE sym_end_time;

static void
rb_whisper_vad_segment_mark(void *p)
{
  ruby_whisper_vad_segment *rwvs = (ruby_whisper_vad_segment *)p;
  rb_gc_mark(rwvs->segments);
}

static size_t
ruby_whisper_vad_segment_memsize(const void *p)
{
  const ruby_whisper_vad_segment *rwvs = p;
  size_t size = sizeof(rwvs);
  if (!rwvs) {
    return 0;
  }
  if (rwvs->index) {
    size += sizeof(rwvs->index);
  }
  return size;
}

static const rb_data_type_t ruby_whisper_vad_segment_type = {
    "ruby_whisper_vad_segment",
    {rb_whisper_vad_segment_mark, RUBY_DEFAULT_FREE, ruby_whisper_vad_segment_memsize,},
    0, 0,
    0
};

static VALUE
ruby_whisper_vad_segment_s_allocate(VALUE klass)
{
  ruby_whisper_vad_segment *rwvs;
  VALUE obj = TypedData_Make_Struct(klass, ruby_whisper_vad_segment, &ruby_whisper_vad_segment_type, rwvs);
  rwvs->segments = Qnil;
  rwvs->index = -1;
  return obj;
}

VALUE
rb_whisper_vad_segment_s_new(VALUE segments, int index)
{
  ruby_whisper_vad_segment *rwvs;
  const VALUE segment = ruby_whisper_vad_segment_s_allocate(cVADSegment);
  TypedData_Get_Struct(segment, ruby_whisper_vad_segment, &ruby_whisper_vad_segment_type, rwvs);
  rwvs->segments = segments;
  rwvs->index = index;
  return segment;
}

static VALUE
ruby_whisper_vad_segment_get_start_time(VALUE self)
{
  ruby_whisper_vad_segment *rwvs;
  ruby_whisper_vad_segments *rwvss;
  float t0;

  TypedData_Get_Struct(self, ruby_whisper_vad_segment, &ruby_whisper_vad_segment_type, rwvs);
  TypedData_Get_Struct(rwvs->segments, ruby_whisper_vad_segments, &ruby_whisper_vad_segments_type, rwvss);
  t0 = whisper_vad_segments_get_segment_t0(rwvss->segments, rwvs->index);
  return DBL2NUM(t0 * 10);
}

static VALUE
ruby_whisper_vad_segment_get_end_time(VALUE self)
{
  ruby_whisper_vad_segment *rwvs;
  ruby_whisper_vad_segments *rwvss;
  float t1;

  TypedData_Get_Struct(self, ruby_whisper_vad_segment, &ruby_whisper_vad_segment_type, rwvs);
  TypedData_Get_Struct(rwvs->segments, ruby_whisper_vad_segments, &ruby_whisper_vad_segments_type, rwvss);
  t1 = whisper_vad_segments_get_segment_t1(rwvss->segments, rwvs->index);
  return DBL2NUM(t1 * 10);
}

static VALUE
ruby_whisper_vad_segment_deconstruct_keys(VALUE self, VALUE keys)
{
  ruby_whisper_vad_segment *rwvs;
  ruby_whisper_vad_segments *rwvss;
  VALUE hash, key;
  long n_keys;
  int i;

  TypedData_Get_Struct(self, ruby_whisper_vad_segment, &ruby_whisper_vad_segment_type, rwvs);
  TypedData_Get_Struct(rwvs->segments, ruby_whisper_vad_segments, &ruby_whisper_vad_segments_type, rwvss);

  hash = rb_hash_new();
  if (NIL_P(keys)) {
    keys = rb_ary_new3(
      N_KEY_NAMES,
      sym_start_time,
      sym_end_time
    );
    n_keys = N_KEY_NAMES;
  } else {
    n_keys = RARRAY_LEN(keys);
    if (n_keys > N_KEY_NAMES) {
      return hash;
    }
  }
  for (i = 0; i < n_keys; i++) {
    key = rb_ary_entry(keys, i);
    if (key == sym_start_time) {
      rb_hash_aset(hash, key, ruby_whisper_vad_segment_get_start_time(self));
    }
    if (key == sym_end_time) {
      rb_hash_aset(hash, key, ruby_whisper_vad_segment_get_end_time(self));
    }
  }

  return hash;
}

void
init_ruby_whisper_vad_segment(VALUE *mVAD)
{
  cVADSegment = rb_define_class_under(*mVAD, "Segment", rb_cObject);

  sym_start_time = ID2SYM(rb_intern("start_time"));
  sym_end_time = ID2SYM(rb_intern("end_time"));

  rb_define_alloc_func(cVADSegment, ruby_whisper_vad_segment_s_allocate);
  rb_define_method(cVADSegment, "start_time", ruby_whisper_vad_segment_get_start_time, 0);
  rb_define_method(cVADSegment, "end_time", ruby_whisper_vad_segment_get_end_time, 0);
  rb_define_method(cVADSegment, "deconstruct_keys", ruby_whisper_vad_segment_deconstruct_keys, 1);
}
