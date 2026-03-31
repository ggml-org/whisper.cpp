require_relative "helper"

class TestVADSegments < TestBase
  def test_initialize
    segments = Whisper::VAD::Segments.new

    assert_raise do
      segments.each do |segment|
      end
    end

    assert_raise do
      segments.length
    end
  end
end
