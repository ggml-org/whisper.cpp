#include <ruby.h>
#include "ruby_whisper.h"

extern VALUE cToken;
extern const rb_data_type_t ruby_whisper_type;

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

void
init_ruby_whisper_token(VALUE *mWhisper)
{
  cToken = rb_define_class_under(*mWhisper, "Token", rb_cObject);

  rb_define_alloc_func(cToken, ruby_whisper_token_allocate);
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
}
