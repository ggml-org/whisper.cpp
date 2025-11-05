require_relative "helper"

class TestCallback < TestBase
  def setup
    GC.start
    @params = Whisper::Params.new
    @whisper = Whisper::Context.new("base.en")
    @audio = File.join(AUDIO)
  end

  def test_new_segment_callback
    @params.new_segment_callback = ->(context, state, n_new, user_data) {
      assert_kind_of Integer, n_new
      assert n_new > 0
      assert_same @whisper, context

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
        assert_match(/ask not what your country can do for you, ask what you can do for your country/, text) if i_segment == 0
      end
    }

    @whisper.transcribe(@audio, @params)
  end

  def test_new_segment_callback_closure
    search_word = "what"
    @params.new_segment_callback = ->(context, state, n_new, user_data) {
      n_segments = context.full_n_segments
      n_new.times do |i|
        i_segment = n_segments - 1 + i
        text = context.full_get_segment_text(i_segment)
        if text.include?(search_word)
          t0 = context.full_get_segment_t0(i_segment)
          t1 = context.full_get_segment_t1(i_segment)
          raise "search word '#{search_word}' found at between #{t0} and #{t1}"
        end
      end
    }

    assert_raise RuntimeError do
      @whisper.transcribe(@audio, @params)
    end
  end

  def test_new_segment_callback_user_data
    udata = Object.new
    @params.new_segment_callback_user_data = udata
    @params.new_segment_callback = ->(context, state, n_new, user_data) {
      assert_same udata, user_data
    }

    @whisper.transcribe(@audio, @params)
  end

  def test_new_segment_callback_user_data_gc
    @params.new_segment_callback_user_data = "My user data"
    @params.new_segment_callback = ->(context, state, n_new, user_data) {
      assert_equal "My user data", user_data
    }
    GC.start

    assert_same @whisper, @whisper.transcribe(@audio, @params)
  end

  def test_progress_callback
    first = nil
    last = nil
    @params.progress_callback = ->(context, state, progress, user_data) {
      assert_kind_of Integer, progress
      assert 0 <= progress && progress <= 100
      assert_same @whisper, context
      first = progress if first.nil?
      last = progress
    }
    @whisper.transcribe(@audio, @params)
    assert_equal 0, first
    assert_equal 100, last
  end

  def test_progress_callback_user_data
    udata = Object.new
    @params.progress_callback_user_data = udata
    @params.progress_callback = ->(context, state, n_new, user_data) {
      assert_same udata, user_data
    }

    @whisper.transcribe(@audio, @params)
  end

  def test_on_progress
    first = nil
    last = nil
    @params.on_progress do |progress|
      assert_kind_of Integer, progress
      assert 0 <= progress && progress <= 100
      first = progress if first.nil?
      last = progress
    end
    @whisper.transcribe(@audio, @params)
    assert_equal 0, first
    assert_equal 100, last
  end

  def test_encoder_begin_callback
    i = 0
    @params.encoder_begin_callback = ->(context, state, user_data) {
      i += 1
    }
    @whisper.transcribe(@audio, @params)
    assert i > 0
  end

  def test_encoder_begin_callback_abort
    logs = []
    Whisper.log_set -> (level, buffer, user_data) {
      logs << buffer if level == Whisper::LOG_LEVEL_ERROR
    }, logs
    @params.encoder_begin_callback = ->(context, state, user_data) {
      return false
    }
    @whisper.transcribe(@audio, @params)
    assert_match(/encoder_begin_callback returned false - aborting/, logs.join)
    Whisper.log_set ->(level, buffer, user_data) {}, nil
  end

  def test_encoder_begin_callback_user_data
    udata = Object.new
    @params.encoder_begin_callback_user_data = udata
    yielded = nil
    @params.encoder_begin_callback = ->(context, state, user_data) {
      yielded = user_data
    }
    @whisper.transcribe(@audio, @params)
    assert_same udata, yielded
  end

  def test_on_encoder_begin
    i = 0
    @params.on_encoder_begin do
      i += 1
    end
    @whisper.transcribe(@audio, @params)
    assert i > 0
  end

  def test_abort_callback
    i = 0
    @params.abort_callback = ->(user_data) {
      assert_nil user_data
      i += 1
      return false
    }
    @whisper.transcribe(@audio, @params)
    assert i > 0
  end

  def test_abort_callback_abort
    i = 0
    @params.abort_callback = ->(user_data) {
      i += 1
      return i == 3
    }
    @whisper.transcribe(@audio, @params)
    assert_equal 3, i
  end

  def test_abort_callback_user_data
    udata = Object.new
    @params.abort_callback_user_data = udata
    yielded = nil
    @params.abort_callback = ->(user_data) {
      yielded = user_data
    }
    @whisper.transcribe(@audio, @params)
    assert_same udata, yielded
  end

  def test_abort_on
    do_abort = false
    _aborted_from_callback = false
    @params.on_new_segment do |segment|
      do_abort = true if segment.text.match?(/ask/)
    end
    i = 0
    @params.abort_on do
      i += 1
      do_abort
    end
    @whisper.transcribe(@audio, @params)
    assert i > 0
  end
end
