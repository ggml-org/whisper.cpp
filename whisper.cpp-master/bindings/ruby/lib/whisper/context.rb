module Whisper
  class Context
    def to_srt
      each_segment.with_index.reduce("") {|srt, (segment, index)|
        srt << "#{index + 1}\n#{segment.to_srt_cue}\n"
      }
    end

    def to_webvtt
      each_segment.with_index.reduce("WEBVTT\n\n") {|webvtt, (segment, index)|
        webvtt << "#{index + 1}\n#{segment.to_webvtt_cue}\n"
      }
    end
  end
end
