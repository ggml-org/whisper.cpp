require_relative "helper"

class TestVAD < TestBase
  def setup
    @whisper = Whisper::Context.new("base.en")
    vad_params = Whisper::VAD::Params.new
    @params = Whisper::Params.new(
      vad: true,
      vad_model_path: "silero-v6.2.0",
      vad_params:
    )
  end

  def test_transcribe
    @whisper.transcribe(TestBase::AUDIO, @params) do |text|
      assert_match(/ask not what your country can do for you[,.] ask what you can do for your country/i, text)
    end
  end

  def test_vad_segments_api
    @whisper.transcribe TestBase::AUDIO, @params

    assert_equal 4, @whisper.full_n_vad_segments
    @whisper.full_n_vad_segments.times do |i|
      assert @whisper.full_get_vad_segment_t0(i) < @whisper.full_get_vad_segment_t1(i)
    end
  end
end
