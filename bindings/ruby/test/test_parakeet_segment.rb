require_relative "helper"

class TestParakeetSegment < TestBase
  def setup
    @parakeet = Parakeet::Context.new(File.join(__dir__, "../../../models/parakeet-tdt-0.6b-v3.bin"))
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
end
