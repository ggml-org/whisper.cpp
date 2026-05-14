require_relative "helper"

class TestParakeetSegment < TestBase
  def setup
    @parakeet = Parakeet::Context.new(File.join(__dir__, "../../../models/parakeet-tdt-0.6b-v3.bin"))
    @parakeet.transcribe AUDIO, Parakeet::Params.new
  end

  def test_segment
    @parakeet.each_segment do |segment|
      assert_instance_of Parakeet::Segment, segment
    end
  end
end
