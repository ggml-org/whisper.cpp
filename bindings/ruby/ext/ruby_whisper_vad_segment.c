#include <ruby.h>
#include "ruby_whisper.h"

extern VALUE cVADSegment;

extern const rb_data_type_t ruby_whisper_vad_segments_type;

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

VALUE
ruby_whisper_vad_segment_s_allocate(VALUE klass)
{
  ruby_whisper_vad_segment *rwvs;
  return TypedData_Make_Struct(klass, ruby_whisper_vad_segment, &ruby_whisper_vad_segment_type, rwvs);
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

VALUE
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

VALUE
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

void
init_ruby_whisper_vad_segment(VALUE *mVAD)
{
  cVADSegment = rb_define_class_under(*mVAD, "Segment", rb_cObject);
  rb_define_alloc_func(cVADSegment, ruby_whisper_vad_segment_s_allocate);
  rb_define_method(cVADSegment, "start_time", ruby_whisper_vad_segment_get_start_time, 0);
  rb_define_method(cVADSegment, "end_time", ruby_whisper_vad_segment_get_end_time, 0);
}
