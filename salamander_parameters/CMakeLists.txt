cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(PLUGIN_NAME biorob_salamander_parameters_plugin)

cmake_policy(SET CMP0048 NEW)
project(
  biorob_salamander_parameters
  VERSION 0.0.1
  DESCRIPTION "Salamander parameters plugin")

# Find Gazebo
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

# Compilation flags
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

# Plugin compilation
add_library(
  biorob_salamander_parameters_plugin
  SHARED
  src/parameters.cc)
target_link_libraries(
  biorob_salamander_parameters_plugin
  ${GAZEBO_libraries}
  yaml-cpp)
set_target_properties(
  biorob_salamander_parameters_plugin
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
# set_target_properties(
#   biorob_salamander_parameters_plugin
#   PROPERTIES
#   PUBLIC_HEADER "parameters.hh")
# target_include_directories(
#   biorob_salamander_parameters_plugin
#   PUBLIC
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/parameters.hh>
#   ${CMAKE_BINARY_DIR}/include)

# Library version
set_target_properties(
  biorob_salamander_parameters_plugin
  PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(
  biorob_salamander_parameters_plugin
  PROPERTIES SOVERSION 0)

# Installation
include(GNUInstallDirs)
install(
  TARGETS biorob_salamander_parameters_plugin
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
