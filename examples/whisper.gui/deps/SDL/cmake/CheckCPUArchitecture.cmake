include(CheckCSourceCompiles)
include(CMakePushCheckState)

function(_internal_check_cpu_architecture macro_check NAME VARIABLE)
  cmake_push_check_state(RESET)
  string(TOUPPER "${NAME}" UPPER_NAME)
  set(CACHE_VARIABLE "CHECK_CPU_ARCHITECTURE_${UPPER_NAME}")
  set(test_src "
int main(int argc, char *argv[]) {
#if ${macro_check}
  return 0;
#else
  choke
#endif
}
")
  check_c_source_compiles("${test_src}" "${CACHE_VARIABLE}")
  cmake_pop_check_state()
  if(${CACHE_VARIABLE})
    set(${VARIABLE} "TRUE" PARENT_SCOPE)
  else()
    set(${VARIABLE} "FALSE" PARENT_SCOPE)
  endif()
endfunction()

function(check_cpu_architecture ARCH VARIABLE)
  if(ARCH STREQUAL "x86")
    _internal_check_cpu_architecture("defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) ||defined( __i386) || defined(_M_IX86)" x86 ${VARIABLE})
  elseif(ARCH STREQUAL "x64")
    _internal_check_cpu_architecture("defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) || defined(_M_AMD64)" x64 ${VARIABLE})
  elseif(ARCH STREQUAL "arm32")
    _internal_check_cpu_architecture("defined(__arm__) || defined(_M_ARM)" arm32 ${VARIABLE})
  elseif(ARCH STREQUAL "arm64")
    _internal_check_cpu_architecture("defined(__aarch64__) || defined(_M_ARM64)" arm64 ${VARIABLE})
  elseif(ARCH STREQUAL "loongarch64")
    _internal_check_cpu_architecture("defined(__loongarch64)" loongarch64 ${VARIABLE})
  else()
    message(WARNING "Unknown CPU architectures (${ARCH}).")
    set(${VARIABLE} FALSE)
  endif()
  set("${VARIABLE}" "${${VARIABLE}}" PARENT_SCOPE)
endfunction()
