require_relative "helper"

class TestVADContext < Test::Unit::TestCase
  def test_initialize
    context = Whisper::VAD::Context.new("silero-v5.1.2")
    assert_instance_of Whisper::VAD::Context, context
  end
end
