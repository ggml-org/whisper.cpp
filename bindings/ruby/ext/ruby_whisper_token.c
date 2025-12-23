#include <ruby.h>
#include "ruby_whisper.h"

#define N_KEY_NAMES 11

extern VALUE cToken;
extern const rb_data_type_t ruby_whisper_type;

static VALUE key_names;
static VALUE sym_id;
static VALUE sym_tid;
static VALUE sym_probability;
static VALUE sym_log_probability;
static VALUE sym_pt;
static VALUE sym_ptsum;
static VALUE sym_t_dtw;
static VALUE sym_voice_length;
static VALUE sym_start_time;
static VALUE sym_end_time;
static VALUE sym_text;

static size_t
ruby_whisper_token_memsize(const void *p)
{
  const ruby_whisper_token *rwt = (const ruby_whisper_token *)p;
  if (!rwt) {
    return 0;
  }
  return sizeof(rwt);
}

static const rb_data_type_t ruby_whisper_token_type = {
  "ruby_whisper_token",
  {0, RUBY_DEFAULT_FREE, ruby_whisper_token_memsize,},
  0, 0,
  0
};

static VALUE
ruby_whisper_token_allocate(VALUE klass)
{
  ruby_whisper_token *rwt;
  VALUE token = TypedData_Make_Struct(klass, ruby_whisper_token, &ruby_whisper_token_type, rwt);
  rwt->token_data = NULL;
  rwt->text = NULL;
  return token;
}

VALUE
ruby_whisper_token_s_init(struct whisper_context *context, int i_segment, int i_token)
{
  whisper_token_data token_data = whisper_full_get_token_data(context, i_segment, i_token);
  const VALUE token = ruby_whisper_token_allocate(cToken);
  ruby_whisper_token *rwt;
  TypedData_Get_Struct(token, ruby_whisper_token, &ruby_whisper_token_type, rwt);
  rwt->token_data = &token_data;
  rwt->text = whisper_full_get_token_text(context, i_segment, i_token);
  return token;
}

/*
 * Token ID.
 *
 * call-seq:
 *   id -> Integer
 */
static VALUE
ruby_whisper_token_get_id(VALUE self)
{
  ruby_whisper_token *rwt;
  GetToken(self, rwt);
  return INT2NUM(rwt->token_data->id);
}

/*
 * Forced timestamp token ID.
 *
 * call-seq:
 *   tid -> Integer
 */
static VALUE
ruby_whisper_token_get_tid(VALUE self)
{
  ruby_whisper_token *rwt;
  GetToken(self, rwt);
  return INT2NUM(rwt->token_data->tid);
}

/*
 * Probability of the token.
 *
 * call-seq:
 *   probability -> Float
 */
static VALUE
ruby_whisper_token_get_p(VALUE self)
{
  ruby_whisper_token *rwt;
  GetToken(self, rwt);
  return DBL2NUM(rwt->token_data->p);
}

/*
 * Log probability of the token.
 *
 * call-seq:
 *   log_probability -> Float
 */
static VALUE
ruby_whisper_token_get_plog(VALUE self)
{
  ruby_whisper_token *rwt;
  GetToken(self, rwt);
  return DBL2NUM(rwt->token_data->plog);
}

/*
 * Probability of the timestamp token.
 *
 * call-seq:
 *   pt -> Float
 */
static VALUE
ruby_whisper_token_get_pt(VALUE self)
{
  ruby_whisper_token *rwt;
  GetToken(self, rwt);
  return DBL2NUM(rwt->token_data->pt);
}

/*
 * Sum of probability of all timestamp tokens.
 *
 * call-seq:
 *   ptsum -> Float
 */
static VALUE
ruby_whisper_token_get_ptsum(VALUE self)
{
  ruby_whisper_token *rwt;
  GetToken(self, rwt);
  return DBL2NUM(rwt->token_data->ptsum);
}

/*
 * [EXPERIMENTAL] Token-level timestamps with DTW
 *
 * Do not use if you haven't computed token-level timestamps with dtw.
 * Roughly corresponds to the moment in audio in which the token was output.
 *
 * call-seq:
 *   t_dtw -> Integer
 */
static VALUE
ruby_whisper_token_get_t_dtw(VALUE self)
{
  ruby_whisper_token *rwt;
  GetToken(self, rwt);
  return LONG2NUM(rwt->token_data->t_dtw);
}

/*
 * Voice length of the token.
 *
 * call-seq:
 *   voice_length -> Float
 */
static VALUE
ruby_whisper_token_get_vlen(VALUE self)
{
  ruby_whisper_token *rwt;
  GetToken(self, rwt);
  return DBL2NUM(rwt->token_data->vlen);
}

/*
 * Get the token text of the token.
 *
 * call-seq:
 *   text -> String
 */
static VALUE
ruby_whisper_token_get_text(VALUE self)
{
  ruby_whisper_token *rwt;
  GetToken(self, rwt);
  return rb_str_new2(rwt->text);
}


/*
 * Start time of the token.
 *
 * Token-level timestamp data.
 * Do not use if you haven't computed token-level timestamps.
 *
 * call-seq:
 *   start_time -> Integer
 */
static VALUE
ruby_whisper_token_get_start_time(VALUE self)
{
  ruby_whisper_token *rwt;
  GetToken(self, rwt);
  return LONG2NUM(rwt->token_data->t0 * 10);
}

/*
 * End time of the token.
 *
 * Token-level timestamp data.
 * Do not use if you haven't computed token-level timestamps.
 *
 * call-seq:
 *   end_time -> Integer
 */
static VALUE
ruby_whisper_token_get_end_time(VALUE self)
{
  ruby_whisper_token *rwt;
  GetToken(self, rwt);
  return LONG2NUM(rwt->token_data->t1 * 10);
}

/*
 * call-seq:
 *   deconstruct_keys(keys) -> hash
 *
 *  Possible keys: :id, :tid, :probability, :log_probability, :pt, :ptsum,
 *                 :t_dtw, :voice_length, :start_time, :end_time, :text
 *    segment.each_token do |token|
 *      token => {text:, probability:}
        puts "#{text} (#{probability})"
 *    end
 */
static VALUE ruby_whisper_token_deconstruct_keys(VALUE self, VALUE keys)
{
  ruby_whisper_token *rwt;
  GetToken(self, rwt);
  VALUE hash = rb_hash_new();
  long n_keys = 0;

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
      rb_hash_aset(hash, key, ruby_whisper_token_get_start_time(self));
      continue;
    }
    if (key == sym_end_time) {
      rb_hash_aset(hash, key, ruby_whisper_token_get_end_time(self));
      continue;
    }
    if (key == sym_text) {
      rb_hash_aset(hash, key, ruby_whisper_token_get_text(self));
      continue;
    }
    if (key == sym_probability) {
      rb_hash_aset(hash, key, ruby_whisper_token_get_p(self));
      continue;
    }
    if (key == sym_id) {
      rb_hash_aset(hash, key, ruby_whisper_token_get_id(self));
      continue;
    }
    if (key == sym_tid) {
      rb_hash_aset(hash, key, ruby_whisper_token_get_tid(self));
      continue;
    }
    if (key == sym_log_probability) {
      rb_hash_aset(hash, key, ruby_whisper_token_get_plog(self));
      continue;
    }
    if (key == sym_pt) {
      rb_hash_aset(hash, key, ruby_whisper_token_get_pt(self));
      continue;
    }
    if (key == sym_ptsum) {
      rb_hash_aset(hash, key, ruby_whisper_token_get_ptsum(self));
      continue;
    }
    if (key == sym_t_dtw) {
      rb_hash_aset(hash, key, ruby_whisper_token_get_t_dtw(self));
      continue;
    }
    if (key == sym_voice_length) {
      rb_hash_aset(hash, key, ruby_whisper_token_get_vlen(self));
      continue;
    }
  }

  return hash;
}


void
init_ruby_whisper_token(VALUE *mWhisper)
{
  cToken = rb_define_class_under(*mWhisper, "Token", rb_cObject);

  rb_define_alloc_func(cToken, ruby_whisper_token_allocate);

  sym_id = ID2SYM(rb_intern("id"));
  sym_tid = ID2SYM(rb_intern("tid"));
  sym_probability = ID2SYM(rb_intern("probability"));
  sym_log_probability = ID2SYM(rb_intern("log_probability"));
  sym_pt = ID2SYM(rb_intern("pt"));
  sym_ptsum = ID2SYM(rb_intern("ptsum"));
  sym_t_dtw = ID2SYM(rb_intern("t_dtw"));
  sym_voice_length = ID2SYM(rb_intern("voice_length"));
  sym_start_time = ID2SYM(rb_intern("start_time"));
  sym_end_time = ID2SYM(rb_intern("end_time"));
  sym_text = ID2SYM(rb_intern("text"));
  key_names = rb_ary_new3(
    N_KEY_NAMES,
    sym_id,
    sym_tid,
    sym_probability,
    sym_log_probability,
    sym_pt,
    sym_ptsum,
    sym_t_dtw,
    sym_voice_length,
    sym_start_time,
    sym_end_time,
    sym_text
  );

  rb_define_method(cToken, "id", ruby_whisper_token_get_id, 0);
  rb_define_method(cToken, "tid", ruby_whisper_token_get_tid, 0);
  rb_define_method(cToken, "probability", ruby_whisper_token_get_p, 0);
  rb_define_method(cToken, "log_probability", ruby_whisper_token_get_plog, 0);
  rb_define_method(cToken, "pt", ruby_whisper_token_get_pt, 0);
  rb_define_method(cToken, "ptsum", ruby_whisper_token_get_ptsum, 0);
  rb_define_method(cToken, "t_dtw", ruby_whisper_token_get_t_dtw, 0);
  rb_define_method(cToken, "voice_length", ruby_whisper_token_get_vlen, 0);
  rb_define_method(cToken, "start_time", ruby_whisper_token_get_start_time, 0);
  rb_define_method(cToken, "end_time", ruby_whisper_token_get_end_time, 0);
  rb_define_method(cToken, "text", ruby_whisper_token_get_text, 0);

  rb_define_method(cToken, "deconstruct_keys", ruby_whisper_token_deconstruct_keys, 1);
}

#undef N_KEY_NAMES
