cmake_minimum_required(VERSION 3.0...3.5)
project(ci_utils C CXX)

set(txt "CC=${CMAKE_C_COMPILER}
CXX=${CMAKE_CXX_COMPILER}
CFLAGS=${CMAKE_C_FLAGS}
CXXFLAGS=${CMAKE_CXX_FLAGS}
LDFLAGS=${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_C_STANDARD_LIBRARIES}
")

message("${txt}")

set(VAR_PATH "/tmp/env.txt" CACHE PATH "Where to write environment file")
message(STATUS "Writing CC/CXX/CFLAGS/CXXFLAGS/LDFLAGS environment to ${VAR_PATH}")

file(WRITE "${VAR_PATH}" "${txt}")
