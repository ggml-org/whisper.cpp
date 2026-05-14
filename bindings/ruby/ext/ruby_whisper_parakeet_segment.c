#include "ruby_whisper.h"

#define ITERATE_ATTRS(ITERATOR) \
  ITERATOR(start_time, t0, TIME) \
  ITERATOR(end_time, t1, TIME) \
  ITERATOR(text, text, STRING)

#define VAL_FROM_TIME(v) (LONG2NUM((v) * 10))
#define VAL_FROM_STRING(v) (rb_str_new2(v))
#define READER(type) VAL_FROM_##type
#define DEF_ATTR(rb_name, c_name, type) \
  static VALUE \
  ruby_whisper_parakeet_get_##rb_name(VALUE self) \
  { \
    ruby_whisper_parakeet_segment *rwps; \
    GetParakeetSegment(self, rwps); \
    ruby_whisper_parakeet_context *rwpc; \
    GetParakeetContext(rwps->context, rwpc); \
    return READER(type)(parakeet_full_get_segment_##c_name(rwpc->context, rwps->index)); \
  }

extern VALUE cParakeetSegment;
extern const rb_data_type_t ruby_whisper_parakeet_context_type;

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

ITERATE_ATTRS(DEF_ATTR)

void
init_ruby_whisper_parakeet_segment(VALUE *mParakeet)
{
  cParakeetSegment = rb_define_class_under(*mParakeet, "Segment", rb_cObject);

  rb_define_alloc_func(cParakeetSegment, ruby_whisper_parakeet_segment_s_allocate);

#define REGISTER_ATTR(rb_name, c_name, type) \
  rb_define_method(cParakeetSegment, #rb_name, ruby_whisper_parakeet_get_##rb_name, 0);

  ITERATE_ATTRS(REGISTER_ATTR)

#undef REGISTER_ATTR
}

#undef DEF_ATTR
#undef READER
#undef VAL_FROM_STRING
#undef VAL_FROM_TIME
#undef ITERATE_ATTRS
