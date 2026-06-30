require_relative "helper"

class TestParakeetCallback < TestBase
  def setup
    omit "Skip not to download large model" if ENV["CI"]

    Whisper.instance_variable_set "@whisper", nil
    GC.start
    @params = Parakeet::Params.new
    @parakeet = Parakeet::Context.new("parakeet-tdt-0.6b-v3-q4_0")
  end

  def test_new_segment_callback
    @params.new_segment_callback = ->(context, state, n_new, user_data) {
      assert_kind_of Integer, n_new
      assert n_new > 0
      assert_same @parakeet, context

      n_segments = context.full_n_segments
      n_new.times do |i|
        i_segment = n_segments - 1 + i
        start_time = context.full_get_segment_t0(i_segment) * 10
        end_time = context.full_get_segment_t1(i_segment) * 10
        text = context.full_get_segment_text(i_segment)

        assert_kind_of Integer, start_time
        assert start_time >= 0
        assert_kind_of Integer, end_time
        assert end_time > 0
        assert_match(/ask not what your country can do for you, ask what you can do for your/, text) if i_segment == 0
      end
    }

    @parakeet.transcribe AUDIO, @params
  end

  def test_on_new_segment
    seg = nil
    index = 0
    @params.on_new_segment do |segment|
      assert_instance_of Parakeet::Segment, segment
      if index == 0
        seg = segment
        assert_equal 0, segment.start_time
        assert_match(/ask not what your country can do for you, ask what you can do for your/, segment.text)
      end
      index += 1
    end
    @parakeet.transcribe AUDIO, @params
    assert_equal 0, seg.start_time
    assert_match /ask not what your country can do for you, ask what you can do for your/, seg.text
  end

  def test_on_new_token
    index = 0
    @params.on_new_token do |token|
      assert_instance_of Parakeet::Token, token
      if index == 0
        assert_instance_of Integer, token.start_time
        assert_match "▁And", token.text
      end
      index += 1
    end

    @parakeet.transcribe AUDIO, @params
  end

  def test_on_progress
    first = nil
    @params.on_progress do |progress|
      assert_kind_of Integer, progress
      assert 0 <= progress && progress <= 100
      first = progress if first.nil?
    end

    @parakeet.transcribe AUDIO, @params

    assert_equal 0, first
  end

  def test_on_encoder_begin
    i = 0
    @params.on_encoder_begin do
      i += 1
    end

    @parakeet.transcribe AUDIO, @params

    assert i > 0
  end

  def test_abort_on
    do_abort = false
    @params.on_new_segment do |segment|
      do_abort = true if segment.text.match?(/ask/)
    end
    i = 0
    @params.abort_on do
      i += 1
      do_abort
    end

    @parakeet.transcribe(AUDIO, @params) rescue nil

    assert i > 0
  end
end
