require_relative "helper"

class TestVADParams < TestBase
  PARAM_NAMES = [
    :threshold,
    :min_speech_duration_ms,
    :min_silence_duration_ms,
    :max_speech_duration_s,
    :speech_pad_ms,
    :samples_overlap
  ]

  def setup
    @params = Whisper::VAD::Params.new
  end

  def test_new
    params = Whisper::VAD::Params.new
    assert_kind_of Whisper::VAD::Params, params
  end

  def test_threshold
    assert_in_delta @params.threshold, 0.5
    @params.threshold = 0.7
    assert_in_delta @params.threshold, 0.7
  end

  def test_min_speech_duration
    pend
  end

  def test_min_speech_duration_ms
    assert_equal 250, @params.min_speech_duration_ms
    @params.min_speech_duration_ms = 500
    assert_equal 500, @params.min_speech_duration_ms
  end

  def test_min_silence_duration_ms
    assert_equal 100, @params.min_silence_duration_ms
    @params.min_silence_duration_ms = 200
    assert_equal 200, @params.min_silence_duration_ms
  end

  def test_max_speech_duration
    pend
  end

  def test_max_speech_duration_s
    assert @params.max_speech_duration_s >= 10e37 # Defaults to FLT_MAX
    @params.max_speech_duration_s = 60.0
    assert_equal 60.0, @params.max_speech_duration_s
  end

  def test_speech_pad_ms
    assert_equal 30, @params.speech_pad_ms
    @params.speech_pad_ms = 50
    assert_equal 50, @params.speech_pad_ms
  end

  def test_samples_overlap
    assert_in_delta @params.samples_overlap, 0.1
    @params.samples_overlap = 0.5
    assert_in_delta @params.samples_overlap, 0.5
  end

  def test_equal
    assert_equal @params, Whisper::VAD::Params.new
  end

  def test_new_with_kw_args
    params = Whisper::VAD::Params.new(threshold: 0.7)
    assert_in_delta params.threshold, 0.7
    assert_equal 250, params.min_speech_duration_ms
  end

  def test_new_with_kw_args_non_existent
    assert_raise ArgumentError do
      Whisper::VAD::Params.new(non_existent: "value")
    end
  end

  data(PARAM_NAMES.collect {|param| [param, param]}.to_h)
  def test_new_with_kw_args_default_values(param)
    default_value = @params.send(param)
    value = default_value + 1
    params = Whisper::VAD::Params.new(param => value)
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
