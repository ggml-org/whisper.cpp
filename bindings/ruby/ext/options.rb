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
    started = false
    @cmake_options = output.lines.filter_map {|line|
      if line.chomp == "-- Cache values"
        started = true
        next
      end
      next unless started
      option, value = line.chomp.split("=", 2)
      name, type = option.split(":", 2)
      [name, type, value]
    }
  end

  private

  def configure
    cmake_options.each do |name, type, default_value|
      option = option_name(name)
      value = type == "BOOL" ? enable_config(option) : arg_config("--#{option}")
      @options[name] = [type, value]
    end
  end

  def option_name(name)
    name.downcase.gsub("_", "-")
  end
end
