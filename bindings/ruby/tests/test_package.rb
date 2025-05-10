require_relative "helper"
require 'tempfile'
require 'tmpdir'
require 'shellwords'

class TestPackage < TestBase
  def test_build
    Tempfile.create do |file|
      assert system("gem", "build", "whispercpp.gemspec", "--output", file.to_path.shellescape, exception: true)
      assert file.size > 0
      assert_path_exist file.to_path
    end
  end

  sub_test_case "Building binary on installation" do
    def setup
      system "rake", "build", exception: true
    end

    def test_install
      match_data = `rake -Tbuild`.match(/(whispercpp-(.+)\.gem)/)
      filename = match_data[1]
      version = match_data[2]
      Dir.mktmpdir do |dir|
        system "gem", "install", "--install-dir", dir.shellescape, "--no-document", "pkg/#{filename.shellescape}", exception: true
        assert_installed dir, version
      end
    end

    private

    def assert_installed(dir, version)
      assert_path_exist File.join(dir, "gems/whispercpp-#{version}/lib", "whisper.#{RbConfig::CONFIG["DLEXT"]}")
      assert_path_exist File.join(dir, "gems/whispercpp-#{version}/LICENSE")
      assert_path_not_exist File.join(dir, "gems/whispercpp-#{version}/ext/build")
    end
  end

  def test_build_options
    # This test is disabled as it currently fails when run locally on macOS and
    # Linux. We need to find a good way to handle the situation with build
    # options which varies between platforms.
    # Refs: https://github.com/ggml-org/whisper.cpp/pull/3132
    omit "Temporarily disabled locally as this test currently fails when run locally" unless ENV["CI"]
    options = BuildOptions::Options.new
    assert_empty options.missing_options
    unless ENV["CI"]
      assert_empty options.extra_options
    end
  end
end
