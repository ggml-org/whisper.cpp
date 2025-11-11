#include <ruby.h>
#include "ruby_whisper.h"

extern ID id___method__;
extern ID id_to_enum;

extern VALUE cVADSegments;

extern VALUE rb_whisper_vad_segment_s_new(VALUE segments, int index);

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

static VALUE
ruby_whisper_vad_segments_each(VALUE self)
{
  ruby_whisper_vad_segments *rwvss;
  VALUE method_name;
  int n_segments, i;

  if (!rb_block_given_p()) {
    method_name = rb_funcall(self, id___method__, 0);
    return rb_funcall(self, id_to_enum, 1, method_name);
  }

  TypedData_Get_Struct(self, ruby_whisper_vad_segments, &ruby_whisper_vad_segments_type, rwvss);
  n_segments = whisper_vad_segments_n_segments(rwvss->segments);
  for (i = 0; i < n_segments; ++i) {
    rb_yield(rb_whisper_vad_segment_s_new(self, i));
  }

  return self;
}

void
init_ruby_whisper_vad_segments(VALUE *mVAD)
{
  cVADSegments = rb_define_class_under(*mVAD, "Segments", rb_cObject);
  rb_define_alloc_func(cVADSegments, ruby_whisper_vad_segments_s_allocate);
  rb_define_method(cVADSegments, "each", ruby_whisper_vad_segments_each, 0);
  rb_include_module(cVADSegments, rb_path2class("Enumerable"));
}
