require_relative "helper"

class TestError < TestBase
  def test_error
    error = Whisper::Error.new(-2)
    assert_equal "failed to compute log mel spectrogram", error.message
    assert_equal(-2, error.code)
  end

  def test_unknown_error
    error = Whisper::Error.new(-20)
    assert_equal "unknown error", error.message
  end

  def test_non_int_code
    assert_raise TypeError do
      _error = Whisper::Error.new("non int")
    end
  end
end
