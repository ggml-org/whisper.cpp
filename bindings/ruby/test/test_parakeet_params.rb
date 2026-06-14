require_relative "helper"
require "etc"

class TestParakeetParams < TestBase
  PARAM_NAMES = [
    :n_threads,
    :offset_ms,
    :duration_ms,
    :no_context,
    :audio_ctx
  ]

  def setup
    @params = Parakeet::Params.new
  end

  def test_new
    assert_instance_of Parakeet::Params, @params
  end

  def test_n_threads
    assert_equal [4, Etc.nprocessors].min, @params.n_threads

    @params.n_threads = 1
    assert_equal 1, @params.n_threads
  end

  def test_offset_ms
    assert_equal 0, @params.offset_ms

    @params.offset_ms = 10_000
    assert_equal 10_000, @params.offset_ms
  end

  def test_duration_ms
    assert_equal 0, @params.duration_ms

    @params.duration_ms = 60_000
    assert_equal 60_000, @params.duration_ms
  end

  def test_no_context
    assert_equal true, @params.no_context

    @params.no_context = false
    assert_equal false, @params.no_context
  end

  def test_audio_ctx
    assert_equal 0, @params.audio_ctx

    @params.audio_ctx = 1
    assert_equal 1, @params.audio_ctx
  end

  def test_new_with_kw_args
    params = Parakeet::Params.new(n_threads: 1)
    assert_equal 1, params.n_threads
    assert_equal 0, params.offset_ms
  end

  data(PARAM_NAMES.collect {|param| [param, param]}.to_h)
  def test_new_with_kw_args_default_values(param)
    default_value = @params.send(param)
    value = case [param, default_value]
            in [*, true | false]
              !default_value
            in [*, Integer]
              default_value + 1
            end
    params = Parakeet::Params.new(param => value)
    assert_equal value, params.send(param)

    PARAM_NAMES.reject {|name| name == param}.each do |name|
      assert_equal @params.send(name), params.send(name)
    end
  end
end
