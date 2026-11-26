cmake_minimum_required(VERSION 3.13)
project(stream_addon)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../../cmake ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DNAPI_VERSION=4)
include_directories(${CMAKE_JS_INC})

set(TARGET stream.node)

add_library(${TARGET} SHARED
    addon.cpp
    whisper-stream.cpp
)

set_target_properties(${TARGET} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)
include(DefaultTargetOptions)

execute_process(COMMAND node -p "require('node-addon-api').include"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NODE_ADD_ON_API_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\"" "" NODE_ADD_ON_API_DIR ${NODE_ADD_ON_API_DIR})
string(REPLACE "\\\\" "/" NODE_ADD_ON_API_DIR ${NODE_ADD_ON_API_DIR})
target_include_directories(${TARGET} PRIVATE ${NODE_ADD_ON_API_DIR})

target_link_libraries(${TARGET}
    whisper
    common
    ${CMAKE_THREAD_LIBS_INIT}
)

if(MSVC AND CMAKE_NODEJS_DEF AND CMAKE_JS_NODEJS_TARGET)
    target_link_libraries(${TARGET} ${CMAKE_JS_NODEJS_TARGET})
elseif(CMAKE_JS_LIB)
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_JS_LIB} ${CMAKE_NODEJS_LIB_TARGET})
    target_link_libraries(${TARGET} ${CMAKE_NODEJS_LIB_TARGET})
endif()