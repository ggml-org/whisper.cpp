# whisper.cpp/examples/bench

A very basic tool for benchmarking the inference performance on your device. The tool simply runs the Encoder part of
the transformer on some random audio data and records the execution time. This way we can have an objective comparison
of the performance of the model for various setups.

Benchmark results are tracked in the following Github issue: https://github.com/ggml-org/whisper.cpp/issues/89

```bash
# run the bench too on the small.en model using 4 threads
$ ./build/bin/whisper-bench -m ./models/ggml-small.en.bin -t 4

whisper_model_load: loading model from './models/ggml-small.en.bin'
whisper_model_load: n_vocab       = 51864
whisper_model_load: n_audio_ctx   = 1500
whisper_model_load: n_audio_state = 768
whisper_model_load: n_audio_head  = 12
whisper_model_load: n_audio_layer = 12
whisper_model_load: n_text_ctx    = 448
whisper_model_load: n_text_state  = 768
whisper_model_load: n_text_head   = 12
whisper_model_load: n_text_layer  = 12
whisper_model_load: n_mels        = 80
whisper_model_load: f16           = 1
whisper_model_load: type          = 3
whisper_model_load: mem_required  = 1048.00 MB
whisper_model_load: adding 1607 extra tokens
whisper_model_load: ggml ctx size = 533.05 MB
whisper_model_load: memory size =    68.48 MB 
whisper_model_load: model size  =   464.44 MB

whisper_print_timings:     load time =   240.82 ms
whisper_print_timings:      mel time =     0.00 ms
whisper_print_timings:   sample time =     0.00 ms
whisper_print_timings:   encode time =  1062.21 ms / 88.52 ms per layer
whisper_print_timings:   decode time =     0.00 ms / 0.00 ms per layer
whisper_print_timings:    total time =  1303.04 ms

system_info: n_threads = 4 | AVX2 = 0 | AVX512 = 0 | NEON = 1 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 | 

If you wish, you can submit these results here:

  https://github.com/ggml-org/whisper.cpp/issues/89

Please include the following information:

  - CPU model
  - Operating system
  - Compiler

```
