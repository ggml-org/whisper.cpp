require_relative "helper"
require "stringio"

class TestParakeet < TestBase
  def test_log_set
    log_callback = Parakeet.instance_variable_get("@log_callback")
    user_data = Parakeet.instance_variable_get("@log_callback_user_data")

    $stdout = StringIO.new
    Parakeet.log_set proc {|level, message, _| puts [level, message].join(": ")}, nil
    Parakeet::Context.new("parakeet-tdt-0.6b-v3")
    sleep 0.1
    $stdout.rewind
    logs = $stdout.string
    assert_match /loading model from/, logs
  ensure
    $stdout = STDOUT
    Parakeet.log_set log_callback, user_data
  end
end
