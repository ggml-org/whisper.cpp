require_relative "helper"

class TestVADContext < TestBase
  def test_initialize
    context = Whisper::VAD::Context.new("silero-v6.2.0")
    assert_instance_of Whisper::VAD::Context, context
  end

  def test_detect
    context = Whisper::VAD::Context.new("silero-v6.2.0")
    segments = context.detect(AUDIO, Whisper::VAD::Params.new)
    assert_segments segments
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

  private

  def assert_segments(segments)
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

  sub_test_case "from samples" do
    def setup
      super
      @vad = Whisper::VAD::Context.new("silero-v6.2.0")
      @samples = File.read(AUDIO, nil, 78).unpack("s<*").collect {|i| i.to_f / 2**15}
    end

    def test_segments_from_samples
      segments = @vad.segments_from_samples(Whisper::VAD::Params.new, @samples, @samples.length)
      assert_segments segments
    end

    def test_segments_from_samples_without_length
      segments = @vad.segments_from_samples(Whisper::VAD::Params.new, @samples)
      assert_segments segments
    end

    def test_segments_from_samples_enumerator
      samples = @samples.each
      segments = @vad.segments_from_samples(Whisper::VAD::Params.new, samples, @samples.length)
      assert_segments segments
    end

    def test_segments_from_samples_enumerator_without_length
      samples = @samples.each
      assert_raise ArgumentError do
        @vad.segments_from_samples(Whisper::VAD::Params.new, samples)
      end
    end

    def test_segments_from_samples_enumerator_with_too_large_length
      samples = @samples.each.take(10).to_enum
      assert_raise StopIteration do
        @vad.segments_from_samples(Whisper::VAD::Params.new, samples, 11)
      end
    end

    def test_segments_from_samples_with_memory_view
      samples = JFKReader.new(AUDIO)
      segments = @vad.segments_from_samples(Whisper::VAD::Params.new, samples)
      assert_segments segments
    end
  end
end
