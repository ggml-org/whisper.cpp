#include "ruby_whisper.h"
#include "common-whisper.h"
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

extern ID id_to_path;

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

  GetVADContext(self, rwvc);
  TypedData_Get_Struct(params, ruby_whisper_vad_params, &ruby_whisper_vad_params_type, rwvp);

  if (rb_respond_to(file_path, id_to_path)) {
    file_path = rb_funcall(file_path, id_to_path, 0);
  }
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
