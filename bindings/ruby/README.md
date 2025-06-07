whispercpp
==========

![whisper.cpp](https://user-images.githubusercontent.com/1991296/235238348-05d0f6a4-da44-4900-a1de-d0707e75b763.jpeg)

Ruby bindings for [whisper.cpp][], an interface of automatic speech recognition model.

Installation
------------

Install the gem and add to the application's Gemfile by executing:

    $ bundle add whispercpp

If bundler is not being used to manage dependencies, install the gem by executing:

    $ gem install whispercpp

You can pass build options for whisper.cpp, for instance:

    $ bundle config build.whispercpp --enable-ggml-cuda

or,

    $ gem install whispercpp -- --enable-ggml-cuda

See whisper.cpp's [README](https://github.com/ggml-org/whisper.cpp/blob/master/README.md) for available options. You need convert options present the README to Ruby-style options, for example:

Boolean options:

* `-DGGML_BLAS=1` -> `--enable-ggml-blas`
* `-DWHISER_COREML=OFF` -> `--disable-whisper-coreml`

Argument options:

* `-DGGML_CUDA_COMPRESSION_MODE=size` -> `--ggml-cuda-compression-mode=size`

Combination:

* `-DGGML_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES="86"` -> `--enable-ggml-cuda --cmake_cuda-architectures="86"`

For boolean options like `GGML_CUDA`, the README says `-DGGML_CUDA=1`. You need strip `-D`, prepend `--enable-` for `1` or `ON` (`--disable-` for `0` or `OFF`) and make it kebab-case: `--enable-ggml-cuda`.  
For options which require arguments like `CMAKE_CUDA_ARCHITECTURES`, the README says `-DCMAKE_CUDA_ARCHITECTURES="86"`. You need strip `-D`, prepend `--`, make it kebab-case, append `=` and append argument: `--cmake-cuda-architectures="86"`.

Usage
-----

```ruby
require "whisper"

whisper = Whisper::Context.new("base")

params = Whisper::Params.new(
  language: "en",
  offset: 10_000,
  duration: 60_000,
  max_text_tokens: 300,
  translate: true,
  print_timestamps: false,
  initial_prompt: "Initial prompt here."
)

whisper.transcribe("path/to/audio.wav", params) do |whole_text|
  puts whole_text
end

```

### Preparing model ###

Some models are prepared up-front:

You also can use shorthand for pre-converted models:

```ruby
whisper = Whisper::Context.new("base.en")
```

You can see the list of prepared model names by `Whisper::Model.pre_converted_models.keys`:

```ruby
puts Whisper::Model.pre_converted_models.keys
# tiny
# tiny.en
# tiny-q5_1
# tiny.en-q5_1
# tiny-q8_0
# base
# base.en
# base-q5_1
# base.en-q5_1
# base-q8_0
#   :
#   :
```

You can also retrieve each model:

```ruby
base_en = Whisper::Model.pre_converted_models["base.en"]
whisper = Whisper::Context.new(base_en)
```

At first time you use a model, it is downloaded automatically. After that, downloaded cached file is used. To clear cache, call `#clear_cache`:

```ruby
Whisper::Model.pre_converted_models["base"].clear_cache
```

You can also use local model files you prepared:

```ruby
whisper = Whisper::Context.new("path/to/your/model.bin")
```

Or, you can download model files:

```ruby
whisper = Whisper::Context.new("https://example.net/uri/of/your/model.bin")
# Or
whisper = Whisper::Context.new(URI("https://example.net/uri/of/your/model.bin"))
```

See [models][] page for details.

### Preparing audio file ###

Currently, whisper.cpp accepts only 16-bit WAV files.

### Voice Activity Detection (VAD) ###

Support for Voice Activity Detection (VAD) can be enabled by setting `Whisper::Params`'s `vad` argument to `true` and specifying VAD model:

```ruby
Whisper::Params.new(
  vad: true,
  vad_model_path: "silero-v5.1.2",
  # other arguments...
)
```

When you pass the model name (`"silero-v5.1.2"`) or URI (`https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin`), it will be downloaded automatically.
Currently, "silero-v5.1.2" is registered as pre-converted model like ASR models. You also specify file path or URI of model.

If you need configure VAD behavior, pass params for that:

```ruby
Whisper::Params.new(
  vad: true,
  vad_model_path: "silero-v5.1.2",
  vad_params: Whisper::VAD::Params.new(
    threshold: 1.0, # defaults to 0.5
    min_speech_duration_ms: 500, # defaults to 250
    min_silence_duration_ms: 200, # defaults to 100
    max_speech_duration_s: 30000, # default is FLT_MAX,
    speech_pad_ms: 50, # defaults to 30
    samples_overlap: 0.5 # defaults to 0.1
  ),
  # other arguments...
)
```

For details on VAD, see [whisper.cpp's README](https://github.com/ggml-org/whisper.cpp?tab=readme-ov-file#voice-activity-detection-vad).

### Output ###

whispercpp supports SRT and WebVTT output:

```ruby
puts whisper.transcribe("path/to/audio.wav", Whisper::Params.new).to_webvtt
# =>
WEBVTT

1
00:00:00.000 --> 00:00:03.860
 My thought I have nobody by a beauty and will as you poured.

2
00:00:03.860 --> 00:00:09.840
 Mr. Rochester is sub in that so-don't find simplest, and devoted about, to let might in

3
00:00:09.840 --> 00:00:09.940
 a

```

You may call `#to_srt`, too


API
---

### Transcription ###

By default, `Whisper::Context#transcribe` works in a single thread. You can make it work in parallel by passing `n_processors` option:

```ruby
whisper.transcribe("path/to/audio.wav", params, n_processors: Etc.nprocessors)
```

Note that transcription occasionally might be low accuracy when it works in parallel.

### Segments ###

Once `Whisper::Context#transcribe` called, you can retrieve segments by `#each_segment`:

```ruby
def format_time(time_ms)
  sec, decimal_part = time_ms.divmod(1000)
  min, sec = sec.divmod(60)
  hour, min = min.divmod(60)
  "%02d:%02d:%02d.%03d" % [hour, min, sec, decimal_part]
end

whisper
  .transcribe("path/to/audio.wav", params)
  .each_segment.with_index do |segment, index|
    line = "[%{nth}: %{st} --> %{ed}] %{text}" % {
      nth: index + 1,
      st: format_time(segment.start_time),
      ed: format_time(segment.end_time),
      text: segment.text
    }
    line << " (speaker turned)" if segment.speaker_turn_next?
    puts line
  end

```

You can also add hook to params called on new segment:

```ruby
# Add hook before calling #transcribe
params.on_new_segment do |segment|
  line = "[%{st} --> %{ed}] %{text}" % {
    st: format_time(segment.start_time),
    ed: format_time(segment.end_time),
    text: segment.text
  }
  line << " (speaker turned)" if segment.speaker_turn_next?
  puts line
end

whisper.transcribe("path/to/audio.wav", params)

```

### Models ###

You can see model information:

```ruby
whisper = Whisper::Context.new("base")
model = whisper.model

model.n_vocab # => 51864
model.n_audio_ctx # => 1500
model.n_audio_state # => 512
model.n_audio_head # => 8
model.n_audio_layer # => 6
model.n_text_ctx # => 448
model.n_text_state # => 512
model.n_text_head # => 8
model.n_text_layer # => 6
model.n_mels # => 80
model.ftype # => 1
model.type # => "base"

```

### Logging ###

You can set log callback:

```ruby
prefix = "[MyApp] "
log_callback = ->(level, buffer, user_data) {
  case level
  when Whisper::LOG_LEVEL_NONE
    puts "#{user_data}none: #{buffer}"
  when Whisper::LOG_LEVEL_INFO
    puts "#{user_data}info: #{buffer}"
  when Whisper::LOG_LEVEL_WARN
    puts "#{user_data}warn: #{buffer}"
  when Whisper::LOG_LEVEL_ERROR
    puts "#{user_data}error: #{buffer}"
  when Whisper::LOG_LEVEL_DEBUG
    puts "#{user_data}debug: #{buffer}"
  when Whisper::LOG_LEVEL_CONT
    puts "#{user_data}same to previous: #{buffer}"
  end
}
Whisper.log_set log_callback, prefix
```

Using this feature, you are also able to suppress log:

```ruby
Whisper.log_set ->(level, buffer, user_data) {
  # do nothing
}, nil
Whisper::Context.new("base")
```

### Low-level API to transcribe ###

You can also call `Whisper::Context#full` and `#full_parallel` with a Ruby array as samples. Although `#transcribe` with audio file path is recommended because it extracts PCM samples in C++ and is fast, `#full` and `#full_parallel` give you flexibility.

```ruby
require "whisper"
require "wavefile"

reader = WaveFile::Reader.new("path/to/audio.wav", WaveFile::Format.new(:mono, :float, 16000))
samples = reader.enum_for(:each_buffer).map(&:samples).flatten

whisper = Whisper::Context.new("base")
whisper
  .full(Whisper::Params.new, samples)
  .each_segment do |segment|
    puts segment.text
  end
```

The second argument `samples` may be an array, an object with `length` and `each` method, or a MemoryView. If you can prepare audio data as C array and export it as a MemoryView, whispercpp accepts and works with it with zero copy.

Development
-----------

    % git clone https://github.com/ggml-org/whisper.cpp.git
    % cd whisper.cpp/bindings/ruby
    % rake test

First call of `rake test` builds an extension and downloads a model for testing. After that, you add tests in `tests` directory and modify `ext/ruby_whisper.cpp`.

If something seems wrong on build, running `rake clean` solves some cases.

### Need help ###

* Windows support
* Refinement of C/C++ code, especially memory management

License
-------

The same to [whisper.cpp][].

[whisper.cpp]: https://github.com/ggml-org/whisper.cpp
[models]: https://github.com/ggml-org/whisper.cpp/tree/master/models
