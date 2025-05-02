require "uri"
require "net/http"
require "time"
require "pathname"
require "io/console/size"

module Whisper
  class Model
    class URI
      def initialize(uri)
        @uri = URI(uri)
      end

      def to_path
        cache
        cache_path.to_path
      end

      def clear_cache
        path = cache_path
        path.delete if path.exist?
      end

      private

      def cache_path
        base_cache_dir/@uri.host/@uri.path[1..]
      end

      def base_cache_dir
        base = case RUBY_PLATFORM
               when /mswin|mingw/
                 ENV.key?("LOCALAPPDATA") ? Pathname(ENV["LOCALAPPDATA"]) : Pathname(Dir.home)/"AppData/Local"
               when /darwin/
                 Pathname(Dir.home)/"Library/Caches"
               else
                 ENV.key?("XDG_CACHE_HOME") ? Pathname(ENV["XDG_CACHE_HOME"]) : Pathname(Dir.home)/".cache"
               end
        base/"whisper.cpp"
      end

      def cache
        path = cache_path
        headers = {}
        headers["if-modified-since"] = path.mtime.httpdate if path.exist?
        request @uri, headers
        path
      end

      def request(uri, headers)
        Net::HTTP.start uri.host, uri.port, use_ssl: uri.scheme == "https" do |http|
          request = Net::HTTP::Get.new(uri, headers)
          http.request request do |response|
            case response
            when Net::HTTPNotModified
              # noop
            when Net::HTTPOK
              return if !response.key?("last-modified") && cache_path.exist?

              download response
            when Net::HTTPRedirection
              request URI(response["location"]), headers
            else
              return if headers.key?("if-modified-since") # Use cache file

              raise "#{response.code} #{response.message}\n#{response.body}"
            end
          end
        end
      rescue => err
        if cache_path.exist?
          warn err
          # Use cache file
        else
          raise
        end
      end

      def download(response)
        path = cache_path
        path.dirname.mkpath unless path.dirname.exist?
        downloading_path = Pathname("#{path}.downloading")
        size = response.content_length
        downloading_path.open "wb" do |file|
          downloaded = 0
          response.read_body do |chunk|
            file << chunk
            downloaded += chunk.bytesize
            show_progress downloaded, size
          end
          $stderr.puts
        end
        downloading_path.rename path
      end

      def show_progress(current, size)
        progress_rate_available = size && $stderr.tty?

        unless @prev
          @prev = Time.now
          $stderr.puts "Downloading #{@uri} to #{cache_path}"
        end

        now = Time.now

        if progress_rate_available
          return if now - @prev < 1 && current < size

          progress_width = 20
          progress = current.to_f / size
          arrow_length = progress * progress_width
          arrow = "=" * (arrow_length - 1) + ">" + " " * (progress_width - arrow_length)
          line = "[#{arrow}] (#{format_bytesize(current)} / #{format_bytesize(size)})"
          padding = ' ' * ($stderr.winsize[1] - line.size)
          $stderr.print "\r#{line}#{padding}"
        else
          return if now - @prev < 1

          $stderr.print "."
        end
        @prev = now
      end

      def format_bytesize(bytesize)
        return "0.0 B" if bytesize.zero?

        units = %w[B KiB MiB GiB TiB]
        exp = (Math.log(bytesize) / Math.log(1024)).to_i
        format("%.1f %s", bytesize.to_f / 1024 ** exp, units[exp])
      end
    end

    @pre_converted_models = %w[
      tiny
      tiny.en
      tiny-q5_1
      tiny.en-q5_1
      tiny-q8_0
      base
      base.en
      base-q5_1
      base.en-q5_1
      base-q8_0
      small
      small.en
      small.en-tdrz
      small-q5_1
      small.en-q5_1
      small-q8_0
      medium
      medium.en
      medium-q5_0
      medium.en-q5_0
      medium-q8_0
      large-v1
      large-v2
      large-v2-q5_0
      large-v2-q8_0
      large-v3
      large-v3-q5_0
      large-v3-turbo
      large-v3-turbo-q5_0
      large-v3-turbo-q8_0
    ].each_with_object({}) {|name, models|
      models[name] = URI.new("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-#{name}.bin")
    }

    class << self
      attr_reader :pre_converted_models
    end
  end
end
