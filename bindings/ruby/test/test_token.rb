require_relative "helper"

class TestToken < TestBase
  def setup
    @segment = whisper.each_segment.first
    @token = @segment.each_token.first
  end

  def test_n_tokens
    assert_equal 27, @segment.n_tokens
  end

  def test_allocate
    token = Whisper::Token.allocate
    assert_raise  do
      token.id
    end
  end

  def test_each_token
    i = 0
    @segment.each_token do |token|
      i += 1
      assert_instance_of Whisper::Token, token
    end
    assert_equal 27, i
  end

  def test_each_token_without_block
    assert_instance_of Enumerator, @segment.each_token
  end

  def test_token
    assert_instance_of Whisper::Token, @token

    assert_instance_of Integer, @token.id
    assert_instance_of Float, @token.p
    assert_equal @token.p, @token.probability
    assert_instance_of Float, @token.plog
    assert_equal @token.plog, @token.log_probability

    assert_instance_of Integer, @token.tid
    assert_instance_of Float, @token.pt
    assert_instance_of Float, @token.ptsum

    assert_instance_of Integer, @token.t0
    assert_instance_of Integer, @token.t1
    assert_equal @token.t0 * 10, @token.start_time
    assert_instance_of Integer, @token.start_time
    assert_equal @token.t1 * 10, @token.end_time

    assert_instance_of Integer, @token.t_dtw

    assert_instance_of Float, @token.vlen
  end
end
