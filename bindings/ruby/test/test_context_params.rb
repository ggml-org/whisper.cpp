require_relative "helper"

class TestContextParams < TestBase
  PARAM_NAMES = [
    :use_gpu,
    :flash_attn,
    :gpu_device,
    :dtw_token_timestamps,
    :dtw_aheads_preset,
    :dtw_n_top
  ]

  def test_new
    params = Whisper::Context::Params.new
    assert_instance_of Whisper::Context::Params, params
  end

  def test_attributes
    params = Whisper::Context::Params.new

    assert_true params.use_gpu
    params.use_gpu = false
    assert_false params.use_gpu

    assert_true params.flash_attn
    params.flash_attn = false
    assert_false params.flash_attn

    assert_equal 0, params.gpu_device
    params.gpu_device = 1
    assert_equal 1, params.gpu_device

    assert_false params.dtw_token_timestamps
    params.dtw_token_timestamps = true
    assert_true params.dtw_token_timestamps

    assert_equal Whisper::AHEADS_NONE, params.dtw_aheads_preset
    params.dtw_aheads_preset =Whisper::AHEADS_BASE
    assert_equal Whisper::AHEADS_BASE, params.dtw_aheads_preset

    assert_nil params.dtw_n_top
    params.dtw_n_top = 6
    assert_equal 6, params.dtw_n_top
    params.dtw_n_top = nil
    assert_nil params.dtw_n_top
  end

  def test_new_with_kw_args
    params = Whisper::Context::Params.new(use_gpu: false)
    assert_false params.use_gpu
  end

  def test_new_with_kw_wargs_non_existent
    assert_raise ArgumentError do
      Whisper::Context::Params.new(non_existent: "value")
    end
  end

  data(PARAM_NAMES.collect {|param| [param, param]}.to_h)
  def test_new_with_kw_args_default_values(param)
    default_params = Whisper::Context::Params.new
    default_value = default_params.send(param)
    value = if param == :dtw_n_top
              6
            else
              case default_value
              in true | false
                !default_value
              in Integer
                default_value + 1
              end
            end
    params = Whisper::Context::Params.new(param => value)
    assert_equal value, params.send(param)

    PARAM_NAMES.reject {|name| name == param}.each do |name|
      expected = default_params.send(name)
      actual = params.send(name)
      assert_equal expected, actual
    end
  end
end
