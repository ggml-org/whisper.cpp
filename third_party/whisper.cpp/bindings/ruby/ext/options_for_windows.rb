require_relative "options"

class OptionsForWindows < Options
  def to_s
    command_line(*generator_args, *to_a)
  end

  def graphviz_cmake_args
    generator_args
  end

  private

  def arm?
    RbConfig::CONFIG["host_cpu"].to_s.downcase.match?(/\A(?:arm64|aarch64)\z/)
  end

  def cmake_options_output
    Dir.chdir(__dir__) do
      IO.popen([@cmake, "-S", "sources", "-B", "build", *generator_args, "-L"]) {|io| io.read}
    end
  end

  def generator_args
    generator = cmake_generator
    ["-G", generator] if generator && !generator.empty?
  end

  def cmake_generator
    return @cmake_generator if defined?(@cmake_generator)

    generator = ENV["CMAKE_GENERATOR"]
    abort "CMAKE_GENERATOR=#{generator} is unsupported for mingw/ucrt Ruby" if visual_studio_generator_name?(generator)
    return @cmake_generator = generator unless generator.nil? || generator.empty?

    ninja = find_executable("ninja")
    return @cmake_generator = "Ninja" if ninja

    make = find_executable("make")
    return @cmake_generator = "MSYS Makefiles" if make

    mingw32_make = find_executable("mingw32-make")
    return @cmake_generator = "MinGW Makefiles" if mingw32_make

    @cmake_generator = nil
  end

  def visual_studio_generator_name?(generator)
    generator && generator.start_with?("Visual Studio")
  end
end
