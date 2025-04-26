require "test/unit"
require "whisper"
require_relative "jfk_reader/jfk_reader"

class TestBase < Test::Unit::TestCase
  AUDIO = File.join(__dir__, "..", "..", "..", "samples", "jfk.wav")

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

  module BuildOptions
    load "ext/options.rb", self
    Options.include self

    def enable_config(name)
    end

    def arg_config(name)
    end
  end
end
