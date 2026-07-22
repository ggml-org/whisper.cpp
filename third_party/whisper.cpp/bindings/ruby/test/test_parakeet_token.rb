require_relative "helper"

class TestParakeetToken < TestBase
  ATTRS = %i[
    id
    duration_idx
    duration_value
    frame_index
    probability
    log_probability
    start_time
    end_time
    word_start?
    text
  ]

  def setup
    omit "Skip not to download large model" if ENV["CI"]

    Whisper.instance_variable_set "@whisper", nil
    GC.start

    parakeet = Parakeet::Context.new("parakeet-tdt-0.6b-v3-q4_0")
    params = Parakeet::Params.new
    parakeet.transcribe AUDIO, params
    @segment = parakeet.each_segment.first
  end

  def test_each_token
    i = 0
    @segment.each_token do |token|
      i += 1
      assert_instance_of Parakeet::Token, token
    end
    assert_equal 38, i
  end

  def test_each_token_without_block
    assert_instance_of Enumerator, @segment.each_token
  end

  def test_token
    token = @segment.each_token.first

    assert_instance_of Parakeet::Token, token
    assert_instance_of Integer, token.id
    assert_instance_of Integer, token.duration_idx
    assert_instance_of Integer, token.duration_value
    assert_instance_of Integer, token.frame_index
    assert_instance_of Float, token.probability
    assert_instance_of Float, token.log_probability
    assert_instance_of Integer, token.start_time
    assert_instance_of Integer, token.end_time
    assert_instance_of String, token.text
  end

  def test_text
    assert_equal ["▁And", "▁so", ",", "▁my", "▁f", "ell", "ow", "▁Amer", "ic", "ans", ",", "▁a", "sk", "▁not", "▁what", "▁your", "▁co", "un", "tr", "y", "▁can", "▁do", "▁for", "▁you", ",", "▁a", "sk", "▁what", "▁you", "▁can", "▁do", "▁for", "▁your", "▁co", "un", "tr", "y", "."],
                 @segment.each_token.collect(&:text)
  end

  def test_deconstruct_keys_with_nil
    token = @segment.each_token.first
    expected = ATTRS.collect {|attr| [attr.to_s.sub(/\?\z/, "").intern, token.send(attr)]}.to_h
    assert_equal expected, token.deconstruct_keys(nil)
  end

  def test_deconstruct_keys_with_keys
    token = @segment.each_token.first
    expected = ATTRS.collect {|attr| [attr.to_s.sub(/\?\z/, "").intern, token.send(attr)]}.to_h
    assert_equal expected, token.deconstruct_keys(expected.keys)
  end
end
