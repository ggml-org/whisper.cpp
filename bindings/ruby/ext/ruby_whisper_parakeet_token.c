#include "ruby_whisper.h"

#define ITERATE_MEMBERS(ITERATOR) \
  ITERATOR(id, id, id, id, INT) \
  ITERATOR(duration_idx, duration_idx, duration_idx, duration_idx, INT) \
  ITERATOR(duration_value, duration_value, duration_value, duration_value, INT) \
  ITERATOR(frame_index, frame_index, frame_index, frame_index, INT) \
  ITERATOR(probability, probability, p, p, FLOAT) \
  ITERATOR(log_probability, log_probability, plog, plog, FLOAT) \
  ITERATOR(start_time, start_time, start_time, t0, TIME) \
  ITERATOR(end_time, end_time, end_time, t1, TIME) \
  ITERATOR(word_start?, word_start, word_start_p, is_word_start, BOOL)

#define ITERATE_ATTRS(ITERATOR) \
  ITERATOR(text, text, text, text, STRING)

enum {
#define DEF_IDX(rb_name, s_key, c_name, p_name, type) RUBY_WHISPER_PARAKEET_TOKEN_##c_name,

  ITERATE_MEMBERS(DEF_IDX)
  ITERATE_ATTRS(DEF_IDX)
  RUBY_WHISPER_PARAKEET_TOKEN_NUM_ATTRS,
};

#define VAL_FROM_INT(v) (INT2NUM(v))
#define VAL_FROM_FLOAT(v) (DBL2NUM(v))
#define VAL_FROM_TIME(v) (LONG2NUM(v * 10))
#define VAL_FROM_BOOL(v) ((v) ? Qtrue : Qfalse)
#define VAL_FROM_STRING(v) (rb_str_new2(v))

#define READER(type) VAL_FROM_##type
#define MEMBER_NAME(name) name
#define DEF_MEMBER_ATTR(rb_name, s_key, c_name, p_name, type) \
  static VALUE \
  ruby_whisper_parakeet_token_get_##c_name(VALUE self) \
  { \
    ruby_whisper_parakeet_token *rwpt; \
    GetParakeetToken(self, rwpt); \
    return READER(type)(rwpt->token_data->MEMBER_NAME(p_name)); \
  }

#define DEF_ATTR(rb_name, s_key, c_name, p_name, type) \
  static VALUE \
  ruby_whisper_parakeet_token_get_##c_name(VALUE self) \
  { \
    ruby_whisper_parakeet_token *rwpt; \
    GetParakeetToken(self, rwpt); \
    return rwpt->p_name; \
  }

VALUE cParakeetToken;

#define DEC_ATTR_SYMS(rb_name, s_key, c_name, p_name, type) static VALUE sym_##s_key;

ITERATE_MEMBERS(DEC_ATTR_SYMS)
ITERATE_ATTRS(DEC_ATTR_SYMS)

static void
ruby_whisper_parakeet_token_mark(void *p)
{
  ruby_whisper_parakeet_token *rwpt = (ruby_whisper_parakeet_token *)p;
  rb_gc_mark(rwpt->text);
}

static void
ruby_whisper_parakeet_token_free(void *p)
{
  ruby_whisper_parakeet_token *rwpt = (ruby_whisper_parakeet_token *)p;
  if (rwpt->token_data) {
    xfree(rwpt->token_data);
    rwpt->token_data = NULL;
  }
  xfree(rwpt);
}

static size_t
ruby_whisper_parakeet_token_memsize(const void *p)
{
  ruby_whisper_parakeet_token *rwpt = (ruby_whisper_parakeet_token *)p;
  if (!rwpt) {
    return 0;
  }
  size_t size = sizeof(*rwpt);
  if (rwpt->token_data) {
    size += sizeof(*rwpt->token_data);
  }

  return size;
}

static const rb_data_type_t ruby_whisper_parakeet_token_type = {
  "ruby_whisper_parakeet_token",
  {ruby_whisper_parakeet_token_mark, ruby_whisper_parakeet_token_free, ruby_whisper_parakeet_token_memsize},
  0, 0,
  0,
};

static VALUE
ruby_whisper_parakeet_token_s_allocate(VALUE klass)
{
  ruby_whisper_parakeet_token *rwpt;
  VALUE token = TypedData_Make_Struct(klass, ruby_whisper_parakeet_token, &ruby_whisper_parakeet_token_type, rwpt);

  rwpt->token_data = NULL;
  rwpt->text = Qnil;

  return token;
}

VALUE
ruby_whisper_parakeet_token_s_from_token_data(struct parakeet_context *context, const parakeet_token_data *token_data)
{
  const VALUE token = ruby_whisper_parakeet_token_s_allocate(cParakeetToken);
  ruby_whisper_parakeet_token *rwpt;
  TypedData_Get_Struct(token, ruby_whisper_parakeet_token, &ruby_whisper_parakeet_token_type, rwpt);

  rwpt->token_data = ALLOC(parakeet_token_data);
  *rwpt->token_data = *token_data;
  rwpt->text = rb_utf8_str_new_cstr(parakeet_token_to_str(context, token_data->id));

  return token;
}

VALUE
ruby_whisper_parakeet_token_s_from_index(struct parakeet_context *context, int i_segment, int i_token)
{
  parakeet_token_data token_data = parakeet_full_get_token_data(context, i_segment, i_token);
  return ruby_whisper_parakeet_token_s_from_token_data(context, &token_data);
}

ITERATE_MEMBERS(DEF_MEMBER_ATTR)
// Define #text using parakeet_token_to_str or parakeet_token_to_text
ITERATE_ATTRS(DEF_ATTR)

static VALUE
ruby_whisper_parakeet_token_deconstruct_keys(VALUE self, VALUE keys)
{
  ruby_whisper_parakeet_token *rwpt;
  GetParakeetToken(self, rwpt);

  VALUE hash = rb_hash_new();
  long n_keys = 0;

  if (NIL_P(keys)) {
    VALUE attrs[] = {
#define LIST_SYMS(rb_name, s_key, c_name, p_name, type) sym_##s_key,

      ITERATE_MEMBERS(LIST_SYMS)
      ITERATE_ATTRS(LIST_SYMS)
    };
    keys = rb_ary_new_from_values(RUBY_WHISPER_PARAKEET_TOKEN_NUM_ATTRS, attrs);
    n_keys = RUBY_WHISPER_PARAKEET_TOKEN_NUM_ATTRS;
  } else {
    n_keys = RARRAY_LEN(keys);
    if (n_keys > RUBY_WHISPER_PARAKEET_TOKEN_NUM_ATTRS) {
      return hash;
    }
  }
  for (long i = 0; i < n_keys; i++) {
    VALUE key = rb_ary_entry(keys, i);

#define CHECK_AND_SET_KEY(rb_name, s_key, c_name, p_name, type) \
  if (key == sym_##s_key) { \
    rb_hash_aset(hash, key, ruby_whisper_parakeet_token_get_##c_name(self)); \
  }

    ITERATE_MEMBERS(CHECK_AND_SET_KEY)
    ITERATE_ATTRS(CHECK_AND_SET_KEY)
  }

  return hash;
}

void
init_ruby_whisper_parakeet_token(VALUE *mParakeet)
{
  cParakeetToken = rb_define_class_under(*mParakeet, "Token", rb_cObject);
  rb_define_alloc_func(cParakeetToken, ruby_whisper_parakeet_token_s_allocate);

#define REGISTER_ATTR(rb_name, s_key, c_name, p_name, type) \
  sym_##s_key = ID2SYM(rb_intern(#s_key)); \
  rb_define_method(cParakeetToken, #rb_name, ruby_whisper_parakeet_token_get_##c_name, 0);

  ITERATE_MEMBERS(REGISTER_ATTR)
  ITERATE_ATTRS(REGISTER_ATTR)

  rb_define_method(cParakeetToken, "deconstruct_keys", ruby_whisper_parakeet_token_deconstruct_keys, 1);
}
