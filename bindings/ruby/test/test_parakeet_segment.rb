require_relative "helper"

class TestParakeetSegment < TestBase
  def setup
    omit "Skip not to download large model" if ENV["CI"]

    @parakeet = Parakeet::Context.new("parakeet-tdt-0.6b-v3-q4_0")
    @parakeet.transcribe AUDIO, Parakeet::Params.new
  end

  def test_segment
    whole_text = ""
    @parakeet.each_segment do |segment|
      assert_instance_of Parakeet::Segment, segment
      assert_kind_of Integer, segment.start_time
      assert segment.end_time >= segment.start_time
      assert_kind_of String, segment.text
      whole_text << segment.text
    end
    assert_match(/ask not what your country can do for you, ask what you can do for your country/, whole_text)
  end

  def test_deconstruct_keys
    segment = @parakeet.each_segment.first
    expected = {
      start_time: segment.start_time,
      end_time: segment.end_time,
      text: segment.text
    }
    assert_equal expected, segment.deconstruct_keys([:start_time, :end_time, :text])
  end

  def test_deconstruct_keys_with_nil
    segment = @parakeet.each_segment.first
    expected = {
      start_time: segment.start_time,
      end_time: segment.end_time,
      text: segment.text
    }
    assert_equal expected, segment.deconstruct_keys(nil)
  end
end
