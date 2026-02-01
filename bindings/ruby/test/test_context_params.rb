require_relative "helper"

class TestContextParams < TestBase
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

    # assert_equal , params.dtw_aheads_preset
    # params.dtw_aheads_preset = 
    # assert_equal , params.dtw_aheads_preset

    assert_nil params.dtw_n_top
    params.dtw_n_top = 6
    assert_equal 6, params.dtw_n_top

    # assert_equal 0, params.dtw_aheads.n_heads
    # params.dtw_aheads.n_heads = 6
    # assert_equal 6, params.dtw_aheads.n_heads

    # assert_nil params.dtw_aheads.heads
    # params.dtw_aheads.heads = 
    # assert_equal , params.dtw_aheads.heads
  end
end
