require_relative "helper"
require "pathname"

class TestModel < TestBase
  def test_model
    whisper = Whisper::Context.new("base.en")
    assert_instance_of Whisper::Model, whisper.model
  end

  def test_attributes
    whisper = Whisper::Context.new("base.en")
    model = whisper.model

    assert_equal 51864, model.n_vocab
    assert_equal 1500, model.n_audio_ctx
    assert_equal 512, model.n_audio_state
    assert_equal 8, model.n_audio_head
    assert_equal 6, model.n_audio_layer
    assert_equal 448, model.n_text_ctx
    assert_equal 512, model.n_text_state
    assert_equal 8, model.n_text_head
    assert_equal 6, model.n_text_layer
    assert_equal 80, model.n_mels
    assert_equal 1, model.ftype
    assert_equal "base", model.type
  end

  def test_gc
    model = Whisper::Context.new("base.en").model
    GC.start

    assert_equal 51864, model.n_vocab
    assert_equal 1500, model.n_audio_ctx
    assert_equal 512, model.n_audio_state
    assert_equal 8, model.n_audio_head
    assert_equal 6, model.n_audio_layer
    assert_equal 448, model.n_text_ctx
    assert_equal 512, model.n_text_state
    assert_equal 8, model.n_text_head
    assert_equal 6, model.n_text_layer
    assert_equal 80, model.n_mels
    assert_equal 1, model.ftype
    assert_equal "base", model.type
  end

  def test_pathname
    path = Pathname(Whisper::Model.pre_converted_models["base.en"].to_path)
    whisper = Whisper::Context.new(path)
    model = whisper.model

    assert_equal 51864, model.n_vocab
    assert_equal 1500, model.n_audio_ctx
    assert_equal 512, model.n_audio_state
    assert_equal 8, model.n_audio_head
    assert_equal 6, model.n_audio_layer
    assert_equal 448, model.n_text_ctx
    assert_equal 512, model.n_text_state
    assert_equal 8, model.n_text_head
    assert_equal 6, model.n_text_layer
    assert_equal 80, model.n_mels
    assert_equal 1, model.ftype
    assert_equal "base", model.type
  end

  def test_auto_download
    path = Whisper::Model.pre_converted_models["base.en"].to_path

    assert_path_exist path
    assert_equal 147964211, File.size(path)
  end

  def test_uri_string
    path = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
    whisper = Whisper::Context.new(path)
    model = whisper.model

    assert_equal 51864, model.n_vocab
    assert_equal 1500, model.n_audio_ctx
    assert_equal 512, model.n_audio_state
    assert_equal 8, model.n_audio_head
    assert_equal 6, model.n_audio_layer
    assert_equal 448, model.n_text_ctx
    assert_equal 512, model.n_text_state
    assert_equal 8, model.n_text_head
    assert_equal 6, model.n_text_layer
    assert_equal 80, model.n_mels
    assert_equal 1, model.ftype
    assert_equal "base", model.type
  end

  def test_uri
    path = URI("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin")
    whisper = Whisper::Context.new(path)
    model = whisper.model

    assert_equal 51864, model.n_vocab
    assert_equal 1500, model.n_audio_ctx
    assert_equal 512, model.n_audio_state
    assert_equal 8, model.n_audio_head
    assert_equal 6, model.n_audio_layer
    assert_equal 448, model.n_text_ctx
    assert_equal 512, model.n_text_state
    assert_equal 8, model.n_text_head
    assert_equal 6, model.n_text_layer
    assert_equal 80, model.n_mels
    assert_equal 1, model.ftype
    assert_equal "base", model.type
  end

  def test_coreml_model_auto_download
    uri = Whisper::Model.coreml_compiled_models[Whisper::Model.pre_converted_models["tiny"]]
    model_path = Pathname(uri.to_path).sub_ext("")
    model_path.rmtree if model_path.exist?

    uri.cache
    assert_path_exist model_path
  end
end
