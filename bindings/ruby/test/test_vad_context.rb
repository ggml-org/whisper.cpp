require_relative "helper"

class TestVADContext < TestBase
  def test_initialize
    context = Whisper::VAD::Context.new("silero-v6.2.0")
    assert_instance_of Whisper::VAD::Context, context
  end

  def test_detect
    context = Whisper::VAD::Context.new("silero-v6.2.0")
    segments = context.detect(AUDIO, Whisper::VAD::Params.new)
    assert_instance_of Whisper::VAD::Segments, segments

    i = 0
    segments.each do |segment|
      i += 1
      assert_instance_of Whisper::VAD::Segment, segment
    end
    assert i > 0

    segments.each_with_index do |segment, index|
      assert_instance_of Integer, index
    end

    assert_instance_of Enumerator, segments.each

    segment = segments.each.first
    assert_instance_of Float, segment.start_time
    assert_instance_of Float, segment.end_time

    segment => {start_time:, end_time:}
    assert_equal segment.start_time, start_time
    assert_equal segment.end_time, end_time

    assert_equal 4, segments.length
  end

  def test_invalid_model_type
    assert_raise TypeError do
      Whisper::VAD::Context.new(Object.new)
    end
  end

  def test_allocate
    vad = Whisper::VAD::Context.allocate
    assert_raise do
      vad.detect(AUDIO, Whisper::VAD::Params.new)
    end
  end
end
