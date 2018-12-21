cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# Find Gazebo
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

# Compilation flags
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

# Plugins
add_subdirectory(salamander_msgs)
include_directories(${CMAKE_BINARY_DIR}/salamander_msgs/proto)
add_subdirectory(salamander_parameters)
include_directories(salamander_parameters/src)
add_subdirectory(salamander_control)
add_subdirectory(salamander_control2)
add_subdirectory(salamander_viscous)
# add_subdirectory(salamander_evolution)
add_subdirectory(salamander_logging)
add_subdirectory(salamander_benchmark)
