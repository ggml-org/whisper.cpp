require "test/unit"
require "whisper"
require_relative "jfk_reader/jfk_reader"

class TestBase < Test::Unit::TestCase
  AUDIO = File.join(__dir__, "fixtures", "jfk.wav")

  class << self
    def whisper
      return @whisper if @whisper

      @whisper = Whisper::Context.new("base.en")
      params = Whisper::Params.new
      params.print_timestamps = false
      @whisper.transcribe(TestBase::AUDIO, params)
    end
  end

  private

  def whisper
    self.class.whisper
  end
end
