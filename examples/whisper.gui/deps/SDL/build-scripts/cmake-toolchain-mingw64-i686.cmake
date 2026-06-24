set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86)

find_program(CMAKE_C_COMPILER NAMES i686-w64-mingw32-gcc)
find_program(CMAKE_CXX_COMPILER NAMES i686-w64-mingw32-g++)
find_program(CMAKE_RC_COMPILER NAMES i686-w64-mingw32-windres windres)

if(NOT CMAKE_C_COMPILER)
	message(FATAL_ERROR "Failed to find CMAKE_C_COMPILER.")
endif()

if(NOT CMAKE_CXX_COMPILER)
	message(FATAL_ERROR "Failed to find CMAKE_CXX_COMPILER.")
endif()

if(NOT CMAKE_RC_COMPILER)
        message(FATAL_ERROR "Failed to find CMAKE_RC_COMPILER.")
endif()
