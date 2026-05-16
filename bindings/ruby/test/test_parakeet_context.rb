require_relative "helper"
require "stringio"

class TestParakeetContext < TestBase
  def setup
    Whisper.instance_variable_set "@whisper", nil
    GC.start

    @parakeet = Parakeet::Context.new(File.join(__dir__, "../../../models/parakeet-tdt-0.6b-v3.bin"))
    @params = Parakeet::Params.new
  end

  def test_new
    assert_instance_of Parakeet::Context, @parakeet
  end

  def test_transcribe
    assert_nothing_raised do
      @parakeet.transcribe AUDIO, @params
    end
  end

  def test_transcribe_with_pathname
    assert_nothing_raised do
      @parakeet.transcribe Pathname(AUDIO), @params
    end
  end

  def test_transcribe_with_nothing
    assert_raise_message(/open/) do
      @parakeet.transcribe "nothing", @params
    end
  end
end
