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

  if (parakeet_full(rwpc->context, rwpp->params, pcmf32.data(), pcmf32.size()) != 0) {
    rb_raise(rb_eRuntimeError, "Failed to process audio");
    return Qnil;
  }

  return self;
}

#ifdef __cplusplus
}
#endif
