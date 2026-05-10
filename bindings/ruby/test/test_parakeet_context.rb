require_relative "helper"

class TestParakeetContext < TestBase
  def test_new
    assert_instance_of Parakeet::Context, Parakeet::Context.new(File.join(__dir__, "../../../models/parakeet-tdt-0.6b-v3.bin"))
  end
end
