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
    assert_instance_of Float, @token.probability
    assert_instance_of Float, @token.log_probability

    assert_instance_of Integer, @token.tid
    assert_instance_of Float, @token.pt
    assert_instance_of Float, @token.ptsum

    assert_instance_of Integer, @token.start_time
    assert_instance_of Integer, @token.end_time

    assert_instance_of Integer, @token.t_dtw

    assert_instance_of Float, @token.voice_length

    assert_instance_of String, @token.text
  end

  def test_text
    assert_equal ["[_BEG_]", " And", " so", " my", " fellow", " Americans", ",", " ask", " not", " what", " your", " country", " can", " do", " for", " you", ",", " ask", " what", " you", " can", " do", " for", " your", " country", ".", "[_TT_550]"],
                 @segment.each_token.collect(&:text)
  end

  def test_deconstruct_keys_with_nil
    assert_equal({}, @token.deconstruct_keys(nil))
  end

  def test_deconstruct_keys_with_keys
    keys = %i[id tid probability log_probability pt ptsum t_dtw voice_length start_time end_time text]
    expected = keys.collect {|key| [key, @token.send(key)] }.to_h
    assert_equal expected, @token.deconstruct_keys(keys)
  end
end
