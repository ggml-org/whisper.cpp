#include "ruby_whisper.h"
#include "common-whisper.h"
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

extern const rb_data_type_t ruby_whisper_parakeet_context_type;
extern const rb_data_type_t ruby_whisper_parakeet_params_type;

extern void ruby_whisper_parakeet_prepare_transcription(ruby_whisper_parakeet_params *rwpp, VALUE *context, ruby_whisper_parakeet_abort_callback_user_data *abort_callback_user_data);

extern ID id_to_s;
extern ID id_to_path;
extern ID id_new;

extern VALUE eError;

typedef struct transcribe_without_gvl_args {
  struct parakeet_context *context;
  struct parakeet_full_params params;
  float *samples;
  size_t n_samples;
  int result;
} transcribe_without_gvl_args;

typedef struct {
  ruby_whisper_parakeet_abort_callback_user_data *abort_callback_user_data;
} ruby_whisper_parakeet_transcribe_ubf_args;

static void
ruby_whisper_parakeet_transcribe_ubf(void *rb_args)
{
  ruby_whisper_parakeet_transcribe_ubf_args *args = (ruby_whisper_parakeet_transcribe_ubf_args *)rb_args;

  RUBY_ATOMIC_SET(args->abort_callback_user_data->is_interrupted, 1);
}

static void*
transcribe_without_gvl(void *rb_args)
{
  struct transcribe_without_gvl_args *args = (struct transcribe_without_gvl_args *)rb_args;
  args->result = parakeet_full(args->context, args->params, args->samples, args->n_samples);

  return NULL;
}

VALUE
ruby_whisper_parakeet_transcribe(VALUE self, VALUE audio_path, VALUE params)
{
  if (rb_respond_to(audio_path, id_to_path)) {
    audio_path = rb_funcall(audio_path, id_to_path, 0);
  }

  std::string fname = StringValueCStr(audio_path);
  std::vector<float> pcmf32;
  std::vector<std::vector<float>> pcmf32s;

  if (!read_audio_data(fname, pcmf32, pcmf32s, false)) {
    rb_raise(rb_eRuntimeError, "Failed to open %s", fname.c_str());
    return Qnil;
  }

  ruby_whisper_parakeet_context *rwpc;
  ruby_whisper_parakeet_params *rwpp;
  GetParakeetContext(self, rwpc);
  GetParakeetParams(params, rwpp);

  ruby_whisper_parakeet_abort_callback_user_data abort_callback_user_data = {
    0,
    NULL,
  };
  ruby_whisper_parakeet_prepare_transcription(rwpp, &self, &abort_callback_user_data);

  struct transcribe_without_gvl_args args = {
    rwpc->context,
    rwpp->params,
    pcmf32.data(),
    pcmf32.size(),
    0,
  };

  ruby_whisper_parakeet_transcribe_ubf_args ubf_args = {
    &abort_callback_user_data,
  };

  rb_thread_call_without_gvl(transcribe_without_gvl, (void *)&args, ruby_whisper_parakeet_transcribe_ubf, (void *)&ubf_args);
  if (args.result == 0) {
    return self;
  } else {
    rb_exc_raise(rb_funcall(eError, id_new, 1, args.result));
  }
}

#ifdef __cplusplus
}
#endif
