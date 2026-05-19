#include "ruby_whisper.h"

#define ITERATE_ATTRS(ITERATOR) \
  ITERATOR(start_time, t0, TIME) \
  ITERATOR(end_time, t1, TIME) \
  ITERATOR(text, text, STRING)

enum {
#define DEF_IDX(name, c_name, type) RUBY_WHISPER_PARAKEET_SEGMENT_##name,

  ITERATE_ATTRS(DEF_IDX)
  RUBY_WHISPER_PARAKEET_SEGMENT_NUM_ATTRS,
};

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

extern ID id___method__;
extern ID id_to_enum;
extern VALUE cParakeetSegment;
extern VALUE sym_start_time;
extern VALUE sym_end_time;
extern VALUE sym_text;
extern const rb_data_type_t ruby_whisper_parakeet_context_type;
extern VALUE ruby_whisper_parakeet_token_s_from_index(struct parakeet_context *context, int i_segment, int i_token);

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
  return sizeof(*rwps);
}

static const rb_data_type_t ruby_whisper_parakeet_segment_type = {
  "ruby_whisper_parakeet_segment",
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

static VALUE
ruby_whisper_parakeet_segment_each_token(VALUE self)
{
  if (!rb_block_given_p()) {
    const VALUE method_name = rb_funcall(self, id___method__, 0);
    return rb_funcall(self, id_to_enum, 1, method_name);
  }

  ruby_whisper_parakeet_segment *rwps;
  GetParakeetSegment(self, rwps);
  ruby_whisper_parakeet_context *rwpc;
  GetParakeetContext(rwps->context, rwpc);

  const int n_tokens = parakeet_full_n_tokens(rwpc->context, rwps->index);
  for (int i = 0; i < n_tokens; i++) {
    rb_yield(ruby_whisper_parakeet_token_s_from_index(rwpc->context, rwps->index, i));
  }

  return self;
}

static VALUE
ruby_whisper_parakeet_segment_deconstruct_keys(VALUE self, VALUE keys)
{
  ruby_whisper_parakeet_segment *rwps;
  GetParakeetSegment(self, rwps);
  ruby_whisper_parakeet_context *rwpc;
  GetParakeetContext(rwps->context, rwpc);

  VALUE hash = rb_hash_new();
  long n_keys;
  if (NIL_P(keys)) {
    keys = rb_ary_new3(
      RUBY_WHISPER_PARAKEET_SEGMENT_NUM_ATTRS,
      sym_start_time,
      sym_end_time,
      sym_text
    );
    n_keys = RUBY_WHISPER_PARAKEET_SEGMENT_NUM_ATTRS;
  } else {
    n_keys = RARRAY_LEN(keys);
    if (n_keys > RUBY_WHISPER_PARAKEET_SEGMENT_NUM_ATTRS) {
      return hash;
    }
  }
  for (int i = 0; i < n_keys; i++) {
    VALUE key = rb_ary_entry(keys, i);

#define CHECK_AND_SET_KEY(rb_name, c_name, type) \
    if (key == sym_##rb_name) { \
      rb_hash_aset(hash, key, ruby_whisper_parakeet_get_##rb_name(self)); \
    }

    ITERATE_ATTRS(CHECK_AND_SET_KEY)
  }

  return hash;
}

void
init_ruby_whisper_parakeet_segment(VALUE *mParakeet)
{
  cParakeetSegment = rb_define_class_under(*mParakeet, "Segment", rb_cObject);

  rb_define_alloc_func(cParakeetSegment, ruby_whisper_parakeet_segment_s_allocate);

#define REGISTER_ATTR(rb_name, c_name, type) \
  rb_define_method(cParakeetSegment, #rb_name, ruby_whisper_parakeet_get_##rb_name, 0);

  ITERATE_ATTRS(REGISTER_ATTR)

  rb_define_method(cParakeetSegment, "each_token", ruby_whisper_parakeet_segment_each_token, 0);
  rb_define_method(cParakeetSegment, "deconstruct_keys", ruby_whisper_parakeet_segment_deconstruct_keys, 1);
}
