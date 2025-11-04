module Whisper
  class Segment
    SRT_ESCAPES = {
      "&" => "&amp;",
      "<" => "&lt;",
      ">" => "&gt;",
    }
    SRT_ESCAPES_RE = Regexp.union(SRT_ESCAPES.keys)
    private_constant :SRT_ESCAPES, :SRT_ESCAPES_RE

    def to_srt_cue
      "#{srt_start_time} --> #{srt_end_time}\n#{srt_text}\n"
    end

    def to_webvtt_cue
      "#{webvtt_start_time} --> #{webvtt_end_time}\n#{webvtt_text}\n"
    end

    private

    def time_to_a(time)
      sec, decimal_part = time.divmod(1000)
      min, sec = sec.divmod(60)
      hour, min = min.divmod(60)
      [hour, min, sec, decimal_part]
    end

    def srt_time(time)
      "%02d:%02d:%02d,%03d" % time_to_a(time)
    end

    def srt_start_time
      srt_time(start_time)
    end

    def srt_end_time
      srt_time(end_time)
    end

    def srt_text
      text.gsub(SRT_ESCAPES_RE, SRT_ESCAPES)
    end

    def webvtt_time(time)
      "%02d:%02d:%02d.%03d" % time_to_a(time)
    end

    def webvtt_start_time
      webvtt_time(start_time)
    end

    def webvtt_end_time
      webvtt_time(end_time)
    end

    alias webvtt_text srt_text
  end
end
