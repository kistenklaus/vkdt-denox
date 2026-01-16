cmake_minimum_required(VERSION 3.24)
project(vkdt-denox CXX)

option(VKDT_DENOX_SAN "Enables sanitizers" OFF)
option(VKDT_DENOX_STRICT_WARNINGS "Enables most warnings flags" ON)

option(VKDT_DENOX_USE_SYSTEM_FLATBUFFERS
  "Use system-installed FlatBuffers instead of FetchContent"
  ON)

if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "") 
  set(CMAKE_BUILD_TYPE "Release")
endif()

include(cmake/colorful.cmake)

include(cmake/cli11.cmake)
include(cmake/fmt.cmake)
include(cmake/spdlog.cmake)
include(cmake/flatbuffers.cmake)
include(cmake/dnx.cmake)

add_executable(vkdt-denox 
  # entry
  ${CMAKE_CURRENT_SOURCE_DIR}/codegen/main.cpp
  # utilitiy
  ${CMAKE_CURRENT_SOURCE_DIR}/codegen/io.cpp
  # preprocessing
  ${CMAKE_CURRENT_SOURCE_DIR}/codegen/symbolics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codegen/compress_weights.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codegen/shader_registry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codegen/compute_graph.cpp
  
  # code generation
  ${CMAKE_CURRENT_SOURCE_DIR}/codegen/denox_create_nodes.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codegen/denox_read_source.cpp

)

if (VKDT_DENOX_SAN)
  target_compile_options(vkdt-denox PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(vkdt-denox PRIVATE -fsanitize=undefined -fsanitize=address)
endif()

target_compile_features(vkdt-denox PUBLIC cxx_std_20)

set_target_properties(vkdt-denox PROPERTIES CXX_STANDARD 20)
set_target_properties(vkdt-denox PROPERTIES CXX_STANDARD_REQUIRED ON)
# set_target_properties(vkdt-denox PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(vkdt-denox
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/codegen
)

target_link_libraries(vkdt-denox
  PRIVATE
    denox::fmt
    denox::spdlog
    denox::dnx
)

foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(vkdt-denox PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${cfg} ${CMAKE_BINARY_DIR}/bin
    )
endforeach()

install(TARGETS vkdt-denox 
  RUNTIME DESTINATION bin
)

