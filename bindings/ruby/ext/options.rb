class Options
  def initialize(cmake="cmake")
    @cmake = cmake
    @options = {}

    configure
  end

  def to_s
    @options
      .reject {|name, (type, value)| value.nil?}
      .collect {|name, (type, value)| "-D #{name}=#{value == true ? "ON" : value == false ? "OFF" : value.shellescape}"}
      .join(" ")
  end

  def cmake_options
    return @cmake_options if @cmake_options

    output = nil
    Dir.chdir __dir__ do
      output = `#{@cmake.shellescape} -S sources -B build -L`
    end
    @cmake_options = output.lines.drop_while {|line| line.chomp != "-- Cache values"}.drop(1)
                       .filter_map {|line|
                         option, value = line.chomp.split("=", 2)
                         name, type = option.split(":", 2)
                         [
                           name,
                           [
                             type,
                             type == "BOOL" ? value == "ON" : value
                           ]
                         ]
                       }.to_h
  end

  private

  def configure
    cmake_options.each_pair do |name, (type, default_value)|
      option = option_name(name)
      value = type == "BOOL" ? enable_config(option) : arg_config("--#{option}")
      @options[name] = [type, value]
    end

    configure_accelerate
    configure_metal
    configure_coreml
  end

  # See ggml/src/ggml-cpu/CMakeLists.txt
  def configure_accelerate
    if RUBY_PLATFORM.match?(/darwin/) && enabled?("GGML_ACCELERATE")
      $LDFLAGS << " -framework Accelerate"
    end
  end

  # See ggml/src/ggml-metal/CMakeLists.txt
  def configure_metal
    $LDFLAGS << " -framework Foundation -framework Metal -framework MetalKit" if enabled?("GGML_METAL")
  end

  # See src/CmakeLists.txt
  def configure_coreml
    if enabled?("WHISPER_COREML")
      $LDFLAGS << " -framework Foundation -framework CoreML"
      $defs << "-DRUBY_WHISPER_USE_COREML"
    end
  end

  def option_name(name)
    name.downcase.gsub("_", "-")
  end

  def enabled?(option)
    op = @options[option]
    raise "Option not exist: #{option}" unless op
    raise "Option not boolean: #{option}(#{op[0]})" unless op[0] == "BOOL"
    if op[1].nil?
      cmake_options[option][1]
    else
      op[1]
    end
  end
end
