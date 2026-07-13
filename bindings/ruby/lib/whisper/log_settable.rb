module Whisper
  module LogSettable
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

    def synchronize(&block)
      @mutex ||= Thread::Mutex.new
      @mutex.synchronize &block 
    end
  end
end
