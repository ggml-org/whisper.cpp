require "mutex_m"

module Whisper
  module LogSettable
    class << self
      def extended(base)
        base.extend Mutex_m
      end
    end

    private

    def start_log_callback_thread
      return if @log_callback_thread&.alive?

      @log_callback_thread = Thread.new {
        begin
          while logs = drain_logs
            begin
              callback, user_data = synchronize {[@log_callback, @log_callback_user_data]}
              next if callback.nil?

              logs.each do |(level, text)|
                callback.call level, text, user_data
              end
            rescue => err
              $stderr.puts err
            end
          end
        rescue => err
          $stderr.puts err
        end
      }
    end
  end
end
