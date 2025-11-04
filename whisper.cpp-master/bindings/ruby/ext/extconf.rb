require "mkmf"
require_relative "options"
require_relative "dependencies"

cmake = find_executable("cmake") || abort
options = Options.new(cmake)
have_library("gomp") rescue nil
libs = Dependencies.new(cmake, options)

$INCFLAGS << " -Isources/include -Isources/ggml/include -Isources/examples"
$LOCAL_LIBS << " #{libs}"
$cleanfiles << " build #{libs}"

create_makefile "whisper" do |conf|
  conf << <<~EOF
    $(TARGET_SO): #{libs}
    #{libs}: cmake-targets
    cmake-targets:
    #{"\t"}#{cmake} -S sources -B build -D BUILD_SHARED_LIBS=OFF -D CMAKE_ARCHIVE_OUTPUT_DIRECTORY=#{__dir__} -D CMAKE_POSITION_INDEPENDENT_CODE=ON #{options}
    #{"\t"}#{cmake} --build build --config Release --target common whisper
  EOF
end
