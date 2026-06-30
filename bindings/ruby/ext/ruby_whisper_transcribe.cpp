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
extern ID id_to_path;
extern ID transcribe_option_names[1];

extern void prepare_transcription(ruby_whisper_params * rwp, VALUE * self, int n_processors);
extern VALUE full_body(VALUE rb_args);
extern VALUE full_parallel_body(VALUE rb_args);

typedef struct{
  struct whisper_context *context;
  struct whisper_full_params *params;
  float *samples;
  size_t n_samples;
  int n_processors;
  int result;
} transcribe_without_gvl_args;

static void*
transcribe_without_gvl(void *rb_args)
{
  transcribe_without_gvl_args *args = (transcribe_without_gvl_args *)rb_args;
  args->result = whisper_full_parallel(args->context, *args->params, args->samples, args->n_samples, args->n_processors);

  return NULL;
}

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

  GetContext(self, rw);
  TypedData_Get_Struct(params, ruby_whisper_params, &ruby_whisper_params_type, rwp);

  if (!rb_respond_to(wave_file_path, id_to_s)) {
    rb_raise(rb_eRuntimeError, "Expected file path to wave file");
  }

  if (rb_respond_to(wave_file_path, id_to_path)) {
    wave_file_path = rb_funcall(wave_file_path, id_to_path, 0);
  }
  std::string fname_inp = StringValueCStr(wave_file_path);

  std::vector<float> pcmf32; // mono-channel F32 PCM
  std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

  if (!read_audio_data(fname_inp, pcmf32, pcmf32s, rwp->diarize)) {
    fprintf(stderr, "error: failed to open '%s' as WAV file\n", fname_inp.c_str());
    return self;
  }

  VALUE rb_result;
  if (n_processors == 1) {
    ruby_whisper_full_args args = {
      &self,
      &params,
      pcmf32.data(),
      (int)pcmf32.size(),
    };
    rb_result = full_body((VALUE)&args);
  } else {
    ruby_whisper_full_parallel_args parallel_args = {
      &self,
      &params,
      pcmf32.data(),
      (int)pcmf32.size(),
      n_processors,
    };
    rb_result = full_parallel_body((VALUE)&parallel_args);
  }
  const int result = NUM2INT(rb_result);
  if (result != 0) {
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
