cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(PLUGIN_NAME biorob_salamander_control_plugin)

cmake_policy(SET CMP0048 NEW)
project(
  biorob_salamander_control
  VERSION 0.0.1
  DESCRIPTION "Salamander control plugin")

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
  biorob_salamander_control_plugin
  SHARED
  src/control_plugin.cc)
target_link_libraries(
  biorob_salamander_control_plugin
  ${GAZEBO_libraries})

# Library version
set_target_properties(
  biorob_salamander_control_plugin
  PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(
  biorob_salamander_control_plugin
  PROPERTIES SOVERSION 0)

# Installation
include(GNUInstallDirs)
install(
  TARGETS biorob_salamander_control_plugin
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
