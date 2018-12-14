cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

cmake_policy(SET CMP0048 NEW)
project(
  biorob_salamander_viscous_swimming
  VERSION 0.0.1
  DESCRIPTION "Salamander viscous plugin for simulating swimming")

# Find Gazebo
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

# Compilation flags
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

# Viscous swimming plugin
add_library(
  biorob_salamander_viscous_swimming_plugin
  SHARED
  src/viscous_swimming_plugin.cc)
target_link_libraries(
  biorob_salamander_viscous_swimming_plugin
  ${GAZEBO_libraries})
set_target_properties(
  biorob_salamander_viscous_swimming_plugin
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

# Library version
set_target_properties(
  biorob_salamander_viscous_swimming_plugin
  PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(
  biorob_salamander_viscous_swimming_plugin
  PROPERTIES SOVERSION 0)

# Installation
include(GNUInstallDirs)
install(
  TARGETS biorob_salamander_viscous_swimming_plugin
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
