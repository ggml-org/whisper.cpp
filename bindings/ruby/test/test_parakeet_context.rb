require_relative "helper"
require "stringio"

class TestParakeetContext < TestBase
  def setup
    Whisper.instance_variable_set "@whisper", nil
    GC.start

    @parakeet = Parakeet::Context.new("parakeet-tdt-0.6b-v3")
    @params = Parakeet::Params.new
  end

  def test_new
    assert_instance_of Parakeet::Context, @parakeet
  end

  sub_test_case "full" do
    def setup
      super
      @samples = File.read(AUDIO, nil, 78).unpack("s<*").collect {|i| i.to_f / 2**15}
    end

    def test_full
      @parakeet.full @params, @samples, @samples.length

      segments = @parakeet.each_segment.to_a
      assert_equal 2, segments.length
      assert_match /ask not what your country can do for you, ask what you can do for your/, segments.first.text
    end

    def test_full_without_length
      @parakeet.full(@params, @samples)

      segments = @parakeet.each_segment.to_a
      assert_equal 2, segments.length
      assert_match /ask not what your country can do for you, ask what you can do for your/, @parakeet.each_segment.first.text
    end

    def test_full_enumerator
      samples = @samples.each
      @parakeet.full @params, samples, @samples.length

      segments = @parakeet.each_segment.to_a
      assert_equal 2, segments.length
      assert_match /ask not what your country can do for you, ask what you can do for your/, @parakeet.each_segment.first.text
    end

    def test_full_enumerator_without_length
      samples = @samples.each
      assert_raise ArgumentError do
        @parakeet.full @params, samples
      end
    end

    def test_full_enumerator_with_too_large_length
      samples = @samples.each.take(10).to_enum
      assert_raise StopIteration do
        @parakeet.full @params, samples, 11
      end
    end

    def test_full_with_memory_view
      samples = JFKReader.new(AUDIO)
      @parakeet.full @params, samples

      segments = @parakeet.each_segment.to_a
      assert_equal 2, segments.length
      assert_match /ask not what your country can do for you, ask what you can do for your/, @parakeet.each_segment.first.text
    end

    def test_full_with_memroy_view_gc
      samples = JFKReader.new(AUDIO)
      @parakeet.full(@params, samples)
      GC.start
      require "fiddle"
      Fiddle::MemoryView.export samples do |view|
        assert_equal 176000, view.to_s.unpack("#{view.format}*").length
      end
    end
  end

  def test_transcribe
    assert_nothing_raised do
      @parakeet.transcribe AUDIO, @params
    end
  end

  def test_transcribe_with_pathname
    assert_nothing_raised do
      @parakeet.transcribe Pathname(AUDIO), @params
    end
  end

  def test_transcribe_with_nothing
    assert_raise_message(/open/) do
      @parakeet.transcribe "nothing", @params
    end
  end
end
