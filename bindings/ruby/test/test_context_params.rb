require_relative "helper"

class TestContextParams < TestBase
  def test_context_new_with_default_params
    whisper = Whisper::Context.new("base.en")
    assert_instance_of Whisper::Context, whisper
  end

  def test_context_new_with_use_gpu
    whisper = Whisper::Context.new("base.en", use_gpu: true)
    assert_instance_of Whisper::Context, whisper

    whisper = Whisper::Context.new("base.en", use_gpu: false)
    assert_instance_of Whisper::Context, whisper
  end

  def test_context_new_with_flash_attn
    whisper = Whisper::Context.new("base.en", flash_attn: true)
    assert_instance_of Whisper::Context, whisper

    whisper = Whisper::Context.new("base.en", flash_attn: false)
    assert_instance_of Whisper::Context, whisper
  end

  def test_context_new_with_gpu_device
    whisper = Whisper::Context.new("base.en", gpu_device: 0)
    assert_instance_of Whisper::Context, whisper

    whisper = Whisper::Context.new("base.en", gpu_device: 1)
    assert_instance_of Whisper::Context, whisper
  end

  def test_context_new_with_dtw_token_timestamps
    whisper = Whisper::Context.new("base.en", dtw_token_timestamps: true)
    assert_instance_of Whisper::Context, whisper

    whisper = Whisper::Context.new("base.en", dtw_token_timestamps: false)
    assert_instance_of Whisper::Context, whisper
  end

  def test_context_new_with_dtw_aheads_preset
    whisper = Whisper::Context.new("base.en", dtw_aheads_preset: 0)
    assert_instance_of Whisper::Context, whisper

    whisper = Whisper::Context.new("base.en", dtw_aheads_preset: 1)
    assert_instance_of Whisper::Context, whisper
  end

  def test_context_new_with_combined_params
    whisper = Whisper::Context.new("base.en",
      use_gpu: true,
      flash_attn: true,
      gpu_device: 0,
      dtw_token_timestamps: false,
      dtw_aheads_preset: 0
    )
    assert_instance_of Whisper::Context, whisper
  end
end
