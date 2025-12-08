cmake_minimum_required(VERSION 3.24)
project(vkdt-denox CXX)

option(VKDT_DENOX_SAN "Enables sanitizers" OFF)
option(VKDT_DENOX_STRICT_WARNINGS "Enables most warnings flags" ON)

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
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/io.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/symbolics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/create_nodes.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/compute_graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/compress_weights.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/shader_registry.cpp

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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(vkdt-denox
  PRIVATE
    denox::fmt
    denox::spdlog
    denox::dnx
)




