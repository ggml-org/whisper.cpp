#include <ruby.h>
#include "ruby_whisper.h"

VALUE cVADSegments;
extern VALUE cVADSegments;

static size_t
ruby_whisper_vad_segments_memsize(const void *p)
{
  const ruby_whisper_vad_segments *rwvss = p;
  size_t size = sizeof(rwvss);
  if (!rwvss) {
    return 0;
  }
  if (rwvss->segments) {
    size += sizeof(rwvss->segments);
  }
  return size;
}

void
ruby_whisper_vad_segments_free(void *p)
{
  ruby_whisper_vad_segments *rwvss = (ruby_whisper_vad_segments *)p;
  if (rwvss->segments) {
    whisper_vad_free_segments(rwvss->segments);
    rwvss->segments = NULL;
  }
  xfree(rwvss);
}

const rb_data_type_t ruby_whisper_vad_segments_type = {
  "ruby_whisper_vad_segments",
  {0, ruby_whisper_vad_segments_free, ruby_whisper_vad_segments_memsize,},
  0, 0,
  0
};

VALUE
ruby_whisper_vad_segments_s_allocate(VALUE klass)
{
  ruby_whisper_vad_segments *rwvss;
  VALUE obj = TypedData_Make_Struct(klass, ruby_whisper_vad_segments, &ruby_whisper_vad_segments_type, rwvss);
  rwvss->segments = NULL;
  return obj;
}

void
init_ruby_whisper_vad_segments(VALUE *mVAD)
{
  cVADSegments = rb_define_class_under(*mVAD, "Segments", rb_cObject);
  rb_define_alloc_func(cVADSegments, ruby_whisper_vad_segments_s_allocate);
}
