#include "ruby_whisper.h"
#include "common-whisper.h"
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

extern const rb_data_type_t ruby_whisper_parakeet_context_type;
extern const rb_data_type_t ruby_whisper_parakeet_params_type;

extern ID id_to_s;
extern ID id_to_path;

static struct transcribe_without_gvl_args {
  struct parakeet_context *context;
  struct parakeet_full_params params;
  float *samples;
  size_t n_samples;
  int result;
} transcribe_without_gvl_args;

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

  struct transcribe_without_gvl_args args = {
    rwpc->context,
    rwpp->params,
    pcmf32.data(),
    pcmf32.size(),
    0,
  };

  rb_thread_call_without_gvl(transcribe_without_gvl, (void *)&args, NULL, NULL);
  if (args.result != 0) {
    rb_raise(rb_eRuntimeError, "Failed to process audio");
    return Qnil;
  }

  return self;
}

#ifdef __cplusplus
}
#endif
