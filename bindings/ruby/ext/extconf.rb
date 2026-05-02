require "mkmf"
require_relative "options"
require_relative "dependencies"

cmake = find_executable("cmake") || abort
options = Options.new(cmake)
have_library("gomp") rescue nil
libs = Dependencies.new(cmake, options).to_s

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
