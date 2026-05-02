require "mkmf"

if RUBY_PLATFORM.match? /mswin|mingw|ucrt/
  require_relative "options_for_windows"
  require_relative "dependencies_for_windows"

  Opts = OptionsForWindows
  Deps = DependenciesForWindows
else
  require_relative "options"
  require_relative "dependencies"

  Opts = Options
  Deps = Dependencies
end

cmake = find_executable("cmake") || abort
options = Opts.new(cmake)
have_library("gomp") rescue nil
libs = Deps.new(cmake, options)

append_cflags ["-O3", "-march=native"]
$INCFLAGS << " -Isources/include -Isources/ggml/include -Isources/examples"
$LOCAL_LIBS << " #{libs.local_libs}"
$cleanfiles << " build #{libs}"

create_makefile "whisper" do |conf|
  conf << <<~EOF
    $(TARGET_SO): #{libs}
    #{libs}: cmake-targets
    cmake-targets:
    #{"\t"}"#{cmake}" -S sources -B build #{options}
    #{"\t"}"#{cmake}" --build build --config Release --target common whisper
  EOF
end
