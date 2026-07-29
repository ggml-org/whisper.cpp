# Downstream-consumer fixture for the installed qvac-parakeet package.
#
# Deliberately standalone (not part of the engine build): it is configured
# against an *install tree* by scripts/test-package-consumption.sh, which
# is what makes it a real test of the exported artifact contract
# (find_package name, imported target name, include dirs, link closure)
# rather than of the in-tree targets.
cmake_minimum_required(VERSION 3.20)
project(qvac-parakeet-consumer LANGUAGES CXX)

find_package(qvac-parakeet CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE qvac::parakeet)

# Fail loudly if the package ever stops providing the namespaced target
# (e.g. a future rename that updates the package name but not the alias).
if (NOT TARGET qvac::parakeet)
    message(FATAL_ERROR "qvac-parakeet package did not define the qvac::parakeet imported target")
endif()
