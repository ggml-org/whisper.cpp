require_relative "helper"

class TestParams < TestBase
  PARAM_NAMES = [
    :language,
    :translate,
    :no_context,
    :single_segment,
    :print_special,
    :print_progress,
    :print_realtime,
    :print_timestamps,
    :suppress_blank,
    :suppress_nst,
    :token_timestamps,
    :split_on_word,
    :initial_prompt,
    :diarize,
    :offset,
    :duration,
    :max_text_tokens,
    :temperature,
    :max_initial_ts,
    :length_penalty,
    :temperature_inc,
    :entropy_thold,
    :logprob_thold,
    :no_speech_thold,
    :new_segment_callback,
    :new_segment_callback_user_data,
    :progress_callback,
    :progress_callback_user_data,
    :abort_callback,
    :abort_callback_user_data,
    :vad,
    :vad_model_path,
    :vad_params,
  ]

  def setup
    @params  = Whisper::Params.new
  end

  def test_language
    @params.language = "en"
    assert_equal @params.language, "en"
    @params.language = "auto"
    assert_equal @params.language, "auto"
  end

  def test_offset
    @params.offset = 10_000
    assert_equal @params.offset, 10_000
    @params.offset = 0
    assert_equal @params.offset, 0
  end

  def test_duration
    @params.duration = 60_000
    assert_equal @params.duration, 60_000
    @params.duration = 0
    assert_equal @params.duration, 0
  end

  def test_max_text_tokens
    @params.max_text_tokens = 300
    assert_equal @params.max_text_tokens, 300
    @params.max_text_tokens = 0
    assert_equal @params.max_text_tokens, 0
  end

  def test_translate
    @params.translate = true
    assert @params.translate
    @params.translate = false
    assert !@params.translate
  end

  def test_no_context
    @params.no_context = true
    assert @params.no_context
    @params.no_context = false
    assert !@params.no_context
  end

  def test_single_segment
    @params.single_segment = true
    assert @params.single_segment
    @params.single_segment = false
    assert !@params.single_segment
  end

  def test_print_special
    @params.print_special = true
    assert @params.print_special
    @params.print_special = false
    assert !@params.print_special
  end

  def test_print_progress
    @params.print_progress = true
    assert @params.print_progress
    @params.print_progress = false
    assert !@params.print_progress
  end

  def test_print_realtime
    @params.print_realtime = true
    assert @params.print_realtime
    @params.print_realtime = false
    assert !@params.print_realtime
  end

  def test_print_timestamps
    @params.print_timestamps = true
    assert @params.print_timestamps
    @params.print_timestamps = false
    assert !@params.print_timestamps
  end

  def test_suppress_blank
    @params.suppress_blank = true
    assert @params.suppress_blank
    @params.suppress_blank = false
    assert !@params.suppress_blank
  end

  def test_suppress_nst
    @params.suppress_nst = true
    assert @params.suppress_nst
    @params.suppress_nst = false
    assert !@params.suppress_nst
  end

  def test_token_timestamps
    @params.token_timestamps = true
    assert @params.token_timestamps
    @params.token_timestamps = false
    assert !@params.token_timestamps
  end

  def test_split_on_word
    @params.split_on_word = true
    assert @params.split_on_word
    @params.split_on_word = false
    assert !@params.split_on_word
  end

  def test_initial_prompt
    assert_nil @params.initial_prompt
    @params.initial_prompt = "You are a polite person."
    assert_equal "You are a polite person.", @params.initial_prompt
  end

  def test_temperature
    assert_equal 0.0, @params.temperature
    @params.temperature = 0.5
    assert_equal 0.5, @params.temperature
  end

  def test_max_initial_ts
    assert_equal 1.0, @params.max_initial_ts
    @params.max_initial_ts = 600.0
    assert_equal 600.0, @params.max_initial_ts
  end

  def test_length_penalty
    assert_equal(-1.0, @params.length_penalty)
    @params.length_penalty = 0.5
    assert_equal 0.5, @params.length_penalty
  end

  def test_temperature_inc
    assert_in_delta 0.2, @params.temperature_inc
    @params.temperature_inc = 0.5
    assert_in_delta 0.5, @params.temperature_inc
  end

  def test_entropy_thold
    assert_in_delta 2.4, @params.entropy_thold
    @params.entropy_thold = 3.0
    assert_in_delta 3.0, @params.entropy_thold
  end

  def test_logprob_thold
    assert_in_delta(-1.0, @params.logprob_thold)
    @params.logprob_thold = -0.5
    assert_in_delta(-0.5, @params.logprob_thold)
  end

  def test_no_speech_thold
    assert_in_delta 0.6, @params.no_speech_thold
    @params.no_speech_thold = 0.2
    assert_in_delta 0.2, @params.no_speech_thold
  end

  def test_vad
    assert_false @params.vad
    @params.vad = true
    assert_true @params.vad
  end

  def test_vad_model_path
    assert_nil @params.vad_model_path
    @params.vad_model_path = "silero-v5.1.2"
    assert_equal Whisper::Model.pre_converted_models["silero-v5.1.2"].to_path, @params.vad_model_path
  end

  def test_vad_model_path_with_nil
    @params.vad_model_path = "silero-v5.1.2"
    @params.vad_model_path = nil
    assert_nil @params.vad_model_path
  end

  def test_vad_model_path_with_invalid
    assert_raise TypeError do
      @params.vad_model_path = Object.new
    end
  end

  def test_vad_model_path_with_URI_string
    @params.vad_model_path = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin"
    assert_equal @params.vad_model_path, Whisper::Model.pre_converted_models["silero-v5.1.2"].to_path
  end

  def test_vad_model_path_with_URI
    @params.vad_model_path = URI("https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin")
    assert_equal @params.vad_model_path, Whisper::Model.pre_converted_models["silero-v5.1.2"].to_path
  end

  def test_vad_params
    assert_kind_of Whisper::VAD::Params, @params.vad_params
    default_params = @params.vad_params
    assert_same default_params, @params.vad_params
    assert_equal 0.5, default_params.threshold
    new_params = Whisper::VAD::Params.new
    @params.vad_params = new_params
    assert_same new_params, @params.vad_params
  end

  def test_new_with_kw_args
    params = Whisper::Params.new(language: "es")
    assert_equal "es", params.language
    assert_equal 1.0, params.max_initial_ts
  end

  def test_new_with_kw_args_non_existent
    assert_raise ArgumentError do
      Whisper::Params.new(non_existent: "value")
    end
  end

  def test_new_with_kw_args_wrong_type
    assert_raise TypeError do
      Whisper::Params.new(language: 3)
    end
  end

  data(PARAM_NAMES.collect {|param| [param, param]}.to_h)
  def test_new_with_kw_args_default_values(param)
    default_value = @params.send(param)
    value = case [param, default_value]
            in [*, true | false]
              !default_value
            in [*, Integer | Float]
              default_value + 1
            in [:language, *]
              "es"
            in [:initial_prompt, *]
              "Initial prompt"
            in [/_callback\Z/, *]
              proc {}
            in [/_user_data\Z/, *]
              Object.new
            in [:vad_model_path, *]
              Whisper::Model.pre_converted_models["silero-v5.1.2"].to_path
            in [:vad_params, *]
              Whisper::VAD::Params.new
            end
    params = Whisper::Params.new(param => value)
    if Float === value
      assert_in_delta value, params.send(param)
    else
      assert_equal value, params.send(param)
    end

    PARAM_NAMES.reject {|name| name == param}.each do |name|
      expected = @params.send(name)
      actual = params.send(name)
      if Float === expected
        assert_in_delta expected, actual
      else
        assert_equal expected, actual
      end
    end
  end
end
