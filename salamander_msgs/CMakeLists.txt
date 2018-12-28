cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

cmake_policy(SET CMP0048 NEW)
project(
  biorob_salamander_msgs
  VERSION 0.0.1
  DESCRIPTION "Salamander messages")

# Find Gazebo
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

# Compilation flags
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

ADD_SUBDIRECTORY(proto)

# Binary
SET(MSG_SRC "salamander_msgs_cpp/log_kinematics.pb.cc")
SET(MSG_HEADER "salamander_msgs_cpp/log_kinematics.pb.h")
include_directories(salamander_msgs_cpp)
ADD_LIBRARY(
  salamander_msg_log_kinematics
  SHARED
  ${MSG_SRC}
  ${MSG_HEADER})
set_target_properties(
  salamander_msg_log_kinematics
  PROPERTIES
  PUBLIC_HEADER ${MSG_HEADER})
set_target_properties(
  salamander_msg_log_kinematics
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

# Install
include(GNUInstallDirs)
INSTALL(
  TARGETS salamander_msg_log_kinematics
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/salamander-msgs)

# # Protobuf
# add_library(
#   log_kinematics_pb
#   SHARED
#   plugins/proto/log_kinematics.pb.cc)
# target_link_libraries(
#   log_kinematics_pb
#   ${PROTOBUF_LIBRARY})
# set_target_properties(
#   log_kinematics_pb
#   PROPERTIES
#   LIBRARY_OUTPUT_DIRECTORY ../bin/)

# # Library version
# set_target_properties(
#   biorob_salamander_control_plugin
#   PROPERTIES VERSION ${PROJECT_VERSION})
# set_target_properties(
#   biorob_salamander_control_plugin
#   PROPERTIES SOVERSION 0)

# # Installation
# include(GNUInstallDirs)
# install(
#   TARGETS biorob_salamander_control_plugin
#   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#   PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
