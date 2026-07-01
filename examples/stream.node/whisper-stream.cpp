// whisper-stream.cpp 
#include "whisper-stream.h"
#include "common-whisper.h"
#include "common.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

WhisperStream::WhisperStream(const StreamParams &stream_params)
    : params(stream_params) {}

WhisperStream::~WhisperStream() {
  if (ctx) {
    whisper_print_timings(ctx);
    whisper_free(ctx);
    ctx = nullptr;
  }
  pcmf32_old.clear();
  pcmf32_new.clear();
  pcmf32.clear();
  prompt_tokens.clear();
}

bool WhisperStream::init() {

  // ensure keep/length constraints
  params.keep_ms = std::min(params.keep_ms, params.step_ms);
  params.length_ms = std::max(params.length_ms, params.step_ms);

  // store sample counts as members (SAMPLES, not bytes)
  n_samples_step = int((1e-3 * params.step_ms) * WHISPER_SAMPLE_RATE);
  n_samples_len = int((1e-3 * params.length_ms) * WHISPER_SAMPLE_RATE);
  n_samples_keep = int((1e-3 * params.keep_ms) * WHISPER_SAMPLE_RATE);
  n_samples_30s = int((1e-3 * 30000.0) * WHISPER_SAMPLE_RATE);

  use_vad = (n_samples_step <= 0);

  n_new_line =
      !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1;

  params.no_timestamps = !use_vad;
  params.no_context |= use_vad;

  // language check
  if (params.language != "auto" &&
      whisper_lang_id(params.language.c_str()) == -1) {
    fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
    throw std::runtime_error("unknown language");
  }

  struct whisper_context_params cparams = whisper_context_default_params();
  cparams.use_gpu = params.use_gpu;
  cparams.flash_attn = params.flash_attn;

  // assign member ctx 
  ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
  if (ctx == nullptr) {
    fprintf(stderr, "error: failed to initialize whisper context\n");
    throw std::runtime_error("failed to initialize whisper context");
  }

  // reserve buffers
  pcmf32_new.clear();
  pcmf32_new.reserve(n_samples_30s);
  pcmf32.clear();
  pcmf32_old.clear();
  prompt_tokens.clear();

  {
    fprintf(stderr, "\n");
    if (!whisper_is_multilingual(ctx)) {
      if (params.language != "en" || params.translate) {
        params.language = "en";
        params.translate = false;
        fprintf(stderr,
                "%s: WARNING: model is not multilingual, ignoring language and "
                "translation options\n",
                __func__);
      }
    }
    fprintf(
        stderr,
        "%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = "
        "%.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
        __func__, n_samples_step, float(n_samples_step) / WHISPER_SAMPLE_RATE,
        float(n_samples_len) / WHISPER_SAMPLE_RATE,
        float(n_samples_keep) / WHISPER_SAMPLE_RATE, params.n_threads,
        params.language.c_str(), params.translate ? "translate" : "transcribe",
        params.no_timestamps ? 0 : 1);

    if (!use_vad) {
      fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__,
              n_new_line, params.no_context);
    } else {
      fprintf(stderr, "%s: using VAD, will transcribe on speech activity\n",
              __func__);
    }
    fprintf(stderr, "\n");
  }

  n_iter = 0;
  return true;
}

TranscriptionResult WhisperStream::process(const std::vector<float> &pcmf32_chunk) {

  t_last = std::chrono::high_resolution_clock::now();
  t_start = t_last;
  // append incoming samples
  pcmf32_new.insert(pcmf32_new.end(), pcmf32_chunk.begin(), pcmf32_chunk.end());

  // Not VAD mode: require at least one step worth of samples
  if (!use_vad) {
    if ((int)pcmf32_new.size() < n_samples_step) {
      return TranscriptionResult(); // not enough samples yet
    }

    const int n_samples_new = (int)pcmf32_new.size();

    // take up to params.length_ms audio from previous iteration
    const int n_samples_take =
        std::min((int)pcmf32_old.size(),
                 std::max(0, n_samples_keep + n_samples_len - n_samples_new));

    pcmf32.resize(n_samples_new + n_samples_take);

    // copy tail of old
    for (int i = 0; i < n_samples_take; ++i) {
      pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
    }

    // copy new samples
    memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(),
           n_samples_new * sizeof(float));

    // consume new buffer for next iteration
    pcmf32_old = pcmf32;
    pcmf32_new.clear();

  } else {
    const auto t_now = std::chrono::high_resolution_clock::now();
    // VAD mode: require at least 2 seconds of audio (example); caller can tune
    if ((int)pcmf32_new.size() < 2 * WHISPER_SAMPLE_RATE) {
      return TranscriptionResult();
    }

    if (!::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold,
                      params.freq_thold, false)) {
      pcmf32_new.clear();
      return TranscriptionResult(); // no speech detected
    }

    // take last length_ms worth of samples
    const int take = std::min((int)pcmf32_new.size(), n_samples_len);
    pcmf32.assign(pcmf32_new.end() - take, pcmf32_new.end());
    pcmf32_new.clear();
    t_last = t_now;
  }

  // run the inference
  whisper_full_params wparams = whisper_full_default_params(
      params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH
                           : WHISPER_SAMPLING_GREEDY);

  wparams.print_progress = false;
  wparams.print_special = params.print_special;
  wparams.print_realtime = false;
  wparams.print_timestamps = !params.no_timestamps;
  wparams.translate = params.translate;
  wparams.single_segment = !use_vad;
  wparams.max_tokens = params.max_tokens;
  wparams.language = params.language.c_str();
  wparams.n_threads = params.n_threads;
  wparams.beam_search.beam_size = params.beam_size;

  wparams.audio_ctx = params.audio_ctx;
  wparams.tdrz_enable = params.tinydiarize;
  wparams.temperature_inc = params.no_fallback ? 0.0f : wparams.temperature_inc;

  wparams.prompt_tokens =
      params.no_context
          ? nullptr
          : (prompt_tokens.empty() ? nullptr : prompt_tokens.data());
  wparams.prompt_n_tokens = params.no_context ? 0 : (int)prompt_tokens.size();

  if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
    fprintf(stderr, "%s: failed to process audio\n", __func__);
    return TranscriptionResult();
  }

  // Build result as structured segments (we return a simple concatenated string
  // here; you can change it to JSON or an array of structs for the JS wrapper)
  std::string plain;
  if (use_vad) {
    const int64_t t1 = (t_last - t_start).count() / 1000000;
    const int64_t t0 =
        std::max(0.0, t1 - pcmf32.size() * 1000.0 / WHISPER_SAMPLE_RATE);

    plain += "\n";
    plain += "### Transcription " + std::to_string(n_iter) +
             " START | t0 = " + std::to_string(t0) +
             " ms | t1 = " + std::to_string(t1) + " ms\n";
    plain += "\n";
  }

  const int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
    const char *text = whisper_full_get_segment_text(ctx, i);

    if (params.no_timestamps) {
      plain += text;
    } else {
      const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
      const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
      // append in safe steps to avoid operator precedence issues
      plain += "[";
      plain += to_timestamp(t0, false);
      plain += " --> ";
      plain += to_timestamp(t1, false);
      plain += "]  ";
      plain += text;
      if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
        plain += " [SPEAKER_TURN]";
      }
      plain += "\n";
    }
  }

  if (use_vad) {
    plain += "\n";
    plain += "### Transcription n_iter END\n";
  }

  ++n_iter;

  bool will_commit = false;
  if (!use_vad && (n_iter % n_new_line) == 0) {
    plain += "\n";
    will_commit = true;
    // guard slicing: ensure pcmf32 has enough samples
    if ((int)pcmf32.size() >= n_samples_keep && n_samples_keep > 0) {
      pcmf32_old.assign(pcmf32.end() - n_samples_keep, pcmf32.end());
    } else {
      pcmf32_old = pcmf32;
    }

    // update prompt tokens safely
    if (!params.no_context) {
      prompt_tokens.clear();
      const int n_segments_after = whisper_full_n_segments(ctx);
      for (int si = 0; si < n_segments_after; ++si) {
        const int token_count = whisper_full_n_tokens(ctx, si);
        for (int ti = 0; ti < token_count; ++ti) {
          prompt_tokens.push_back(whisper_full_get_token_id(ctx, si, ti));
        }
      }
    }
  }

  const bool is_final = use_vad || will_commit;

  return TranscriptionResult{ plain, is_final};
}
