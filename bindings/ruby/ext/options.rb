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

    configure_coreml
  end

  def configure_coreml
    use_coreml = if @options["WHISPER_COREML"][1].nil?
                   cmake_options["WHISPER_COREML"][1]
                 else
                   @options["WHISPER_COREML"][1]
                 end
    $CPPFLAGS << " -DRUBY_WHISPER_USE_COREML" if use_coreml
  end

  def option_name(name)
    name.downcase.gsub("_", "-")
  end
end
