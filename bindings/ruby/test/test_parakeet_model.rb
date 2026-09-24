require_relative "helper"

class TestParakeetModel < TestBase
  def test_model
    parakeet = Parakeet::Context.new("test/fixtures/for-tests-ggml-parakeet-tdt.bin")
    assert_instance_of Parakeet::Model, parakeet.model
  end

  def test_attributes
    parakeet = Parakeet::Context.new("test/fixtures/for-tests-ggml-parakeet-tdt.bin")
    model = parakeet.model

    assert_equal 10, model.n_vocab
    assert_equal 3200, model.n_audio_ctx
    assert_equal 8, model.n_audio_state
    assert_equal 2, model.n_audio_head
    assert_equal 1, model.n_audio_layer
    assert_equal 16, model.n_mels
    assert_equal 0, model.ftype
  end
end
