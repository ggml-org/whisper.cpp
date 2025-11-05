require_relative "helper"

class TestVAD < TestBase
  def setup
    @whisper = Whisper::Context.new("base.en")
    vad_params = Whisper::VAD::Params.new
    @params = Whisper::Params.new(
      vad: true,
      vad_model_path: "silero-v5.1.2",
      vad_params:
    )
  end

  def test_transcribe
    @whisper.transcribe(TestBase::AUDIO, @params) do |text|
      assert_match(/ask not what your country can do for you[,.] ask what you can do for your country/i, text)
    end
  end
end
