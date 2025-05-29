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
      gemspec = Gem::Specification.load("whispercpp.gemspec")
      Dir.mktmpdir do |dir|
        system "gem", "install", "--install-dir", dir.shellescape, "--no-document", "pkg/#{gemspec.file_name.shellescape}", exception: true
        assert_installed dir, gemspec.version
      end
    end

    private

    def assert_installed(dir, version)
      assert_path_exist File.join(dir, "gems/whispercpp-#{version}/lib", "whisper.#{RbConfig::CONFIG["DLEXT"]}")
      assert_path_exist File.join(dir, "gems/whispercpp-#{version}/LICENSE")
      assert_path_not_exist File.join(dir, "gems/whispercpp-#{version}/ext/build")
    end
  end
end
