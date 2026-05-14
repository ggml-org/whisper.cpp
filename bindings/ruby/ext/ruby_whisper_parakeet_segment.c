#include "ruby_whisper.h"

extern VALUE cParakeetSegment;

static void
rb_whisper_parakeet_segment_mark(void *p)
{
  ruby_whisper_parakeet_segment *rwps = (ruby_whisper_parakeet_segment *)p;
  rb_gc_mark(rwps->context);
}

static size_t
ruby_whisper_parakeet_segment_memsize(const void *p)
{
  const ruby_whisper_parakeet_segment *rwps = (const ruby_whisper_parakeet_segment *)p;
  if (!rwps) {
    return 0;
  }
  size_t size = sizeof(*rwps);
  if (rwps->index) {
    size += sizeof(rwps->index);
  }
  return size;
}

static const rb_data_type_t ruby_whisper_parakeet_segment_type = {
  "ruby_whisper_segment",
  {rb_whisper_parakeet_segment_mark, RUBY_DEFAULT_FREE, ruby_whisper_parakeet_segment_memsize,},
  0, 0,
  0
};

static VALUE
ruby_whisper_parakeet_segment_s_allocate(VALUE klass)
{
  ruby_whisper_parakeet_segment *rwps;
  return TypedData_Make_Struct(klass, ruby_whisper_parakeet_segment, &ruby_whisper_parakeet_segment_type, rwps);
}

VALUE
ruby_whisper_parakeet_segment_init(VALUE context, int index)
{
  ruby_whisper_parakeet_segment *rwps;

  const VALUE segment = ruby_whisper_parakeet_segment_s_allocate(cParakeetSegment);
  TypedData_Get_Struct(segment, ruby_whisper_parakeet_segment, &ruby_whisper_parakeet_segment_type, rwps);
  rwps->context = context;
  rwps->index = index;

  return segment;
}

void
init_ruby_whisper_parakeet_segment(VALUE *mParakeet)
{
  cParakeetSegment = rb_define_class_under(*mParakeet, "Segment", rb_cObject);

  rb_define_alloc_func(cParakeetSegment, ruby_whisper_parakeet_segment_s_allocate);
}
