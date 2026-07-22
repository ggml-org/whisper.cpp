require_relative "helper"

class TestParakeetContextParams < TestBase
  def setup
    @params = Parakeet::Context::Params.new
  end

  def test_new
    assert_instance_of Parakeet::Context::Params, @params
  end

  def test_attributes
    assert_true @params.use_gpu
    assert_instance_of Integer, @params.gpu_device
  end

  def test_attribute_writer
    @params.use_gpu = false
    assert_false @params.use_gpu

    @params.gpu_device = 2
    assert_equal 2, @params.gpu_device
  end
end
