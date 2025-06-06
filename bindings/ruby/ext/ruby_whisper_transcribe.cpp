#include <ruby.h>
#include "ruby_whisper.h"
#include "common-whisper.h"
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

extern const rb_data_type_t ruby_whisper_type;
extern const rb_data_type_t ruby_whisper_params_type;

extern ID id_to_s;
extern ID id_call;
extern ID transcribe_option_names[1];

extern void
prepare_transcription(ruby_whisper_params * rwp, VALUE * self);

/*
 * transcribe a single file
 * can emit to a block results
 *
 *   params = Whisper::Params.new
 *   params.duration = 60_000
 *   whisper.transcribe "path/to/audio.wav", params do |text|
 *     puts text
 *   end
 *
 * call-seq:
 *   transcribe(path_to_audio, params) {|text| ...}
 **/
VALUE
ruby_whisper_transcribe(int argc, VALUE *argv, VALUE self) {
  ruby_whisper *rw;
  ruby_whisper_params *rwp;
  VALUE wave_file_path, blk, params, kws;
  VALUE opts[1];

  rb_scan_args_kw(RB_SCAN_ARGS_LAST_HASH_KEYWORDS, argc, argv, "2:&", &wave_file_path, &params, &kws, &blk);
  rb_get_kwargs(kws, transcribe_option_names, 0, 1, opts);

  int n_processors = opts[0] == Qundef ? 1 : NUM2INT(opts[0]);

  TypedData_Get_Struct(self, ruby_whisper, &ruby_whisper_type, rw);
  TypedData_Get_Struct(params, ruby_whisper_params, &ruby_whisper_params_type, rwp);

  if (!rb_respond_to(wave_file_path, id_to_s)) {
    rb_raise(rb_eRuntimeError, "Expected file path to wave file");
  }

  std::string fname_inp = StringValueCStr(wave_file_path);

  std::vector<float> pcmf32; // mono-channel F32 PCM
  std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

  if (!read_audio_data(fname_inp, pcmf32, pcmf32s, rwp->diarize)) {
    fprintf(stderr, "error: failed to open '%s' as WAV file\n", fname_inp.c_str());
    return self;
  }
  // Commented out because it is work in progress
  // {
  //   static bool is_aborted = false; // NOTE: this should be atomic to avoid data race

  //   rwp->params.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void * user_data) {
  //     bool is_aborted = *(bool*)user_data;
  //     return !is_aborted;
  //   };
  //   rwp->params.encoder_begin_callback_user_data = &is_aborted;
  // }

  prepare_transcription(rwp, &self);

  if (whisper_full_parallel(rw->context, rwp->params, pcmf32.data(), pcmf32.size(), n_processors) != 0) {
    fprintf(stderr, "failed to process audio\n");
    return self;
  }
  if (NIL_P(blk)) {
    return self;
  }
  const int n_segments = whisper_full_n_segments(rw->context);
  VALUE output = rb_str_new2("");
  for (int i = 0; i < n_segments; ++i) {
    const char * text = whisper_full_get_segment_text(rw->context, i);
    output = rb_str_concat(output, rb_str_new2(text));
  }
  rb_funcall(blk, id_call, 1, output);
  return self;
}
#ifdef __cplusplus
}
#endif
