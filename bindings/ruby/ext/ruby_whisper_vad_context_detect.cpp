#include <ruby.h>
#include "ruby_whisper.h"
#include "common-whisper.h"
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

extern VALUE cVADSegments;

extern const rb_data_type_t ruby_whisper_vad_context_type;
extern const rb_data_type_t ruby_whisper_vad_params_type;
extern const rb_data_type_t ruby_whisper_vad_segments_type;

extern VALUE ruby_whisper_vad_segments_s_init(struct whisper_vad_segments *segments);

VALUE
ruby_whisper_vad_detect(VALUE self, VALUE file_path, VALUE params) {
  ruby_whisper_vad_context *rwvc;
  ruby_whisper_vad_params *rwvp;
  std::string cpp_file_path;
  std::vector<float> pcmf32;
  std::vector<std::vector<float>> pcmf32s;
  whisper_vad_segments *segments;

  TypedData_Get_Struct(self, ruby_whisper_vad_context, &ruby_whisper_vad_context_type, rwvc);
  if (rwvc->context == NULL) {
    rb_raise(rb_eRuntimeError, "Doesn't have referenxe to context internally");
  }
  TypedData_Get_Struct(params, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);

  cpp_file_path = StringValueCStr(file_path);

  if (!read_audio_data(cpp_file_path, pcmf32, pcmf32s, false)) {
    rb_raise(rb_eRuntimeError, "Failed to open '%s' as WAV file\n", cpp_file_path.c_str());
  }

  segments = whisper_vad_segments_from_samples(rwvc->context, rwvp->params, pcmf32.data(), pcmf32.size());
  if (segments == nullptr) {
    rb_raise(rb_eRuntimeError, "Failed to process audio\n");
  }

  return ruby_whisper_vad_segments_s_init(segments);
}

#ifdef __cplusplus
}
#endif
