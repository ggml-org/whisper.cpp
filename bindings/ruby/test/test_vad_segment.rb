require_relative "helper"

class TestVADSegment < TestBase
  def test_initialize
    segment = Whisper::VAD::Segment.new

    assert_raise do
      segment.start_time
    end

    assert_raise do
      segments.end_time
    end

    assert_raise do
      segment => {start_time:, end_time:}
    end
  end
end
