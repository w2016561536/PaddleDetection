cmake_minimum_required(VERSION 3.10)
project(paddlelite_conv_verify CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED PADDLE_LITE_DIR)
  if(DEFINED ENV{PADDLE_LITE_DIR})
    set(PADDLE_LITE_DIR $ENV{PADDLE_LITE_DIR})
  else()
    message(FATAL_ERROR "Set -DPADDLE_LITE_DIR=/path/to/paddle_lite_inference_lib or export PADDLE_LITE_DIR")
  endif()
endif()

include_directories(${PADDLE_LITE_DIR}/cxx/include)

add_executable(run_paddle_lite_conv
  run_paddle_lite_conv.cc
  input_data.cc
)

find_library(PADDLE_LIGHT_API
  NAMES paddle_light_api_shared
  PATHS ${PADDLE_LITE_DIR}/cxx/lib
  NO_DEFAULT_PATH
)

if(NOT PADDLE_LIGHT_API)
  message(FATAL_ERROR "Cannot find libpaddle_light_api_shared in ${PADDLE_LITE_DIR}/cxx/lib")
endif()

target_link_libraries(run_paddle_lite_conv
  ${PADDLE_LIGHT_API}
  dl
  pthread
)
