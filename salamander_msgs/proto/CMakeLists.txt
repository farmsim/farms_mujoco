INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf REQUIRED)
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})

# Generate code
# message(${PROTO_SRC})
# message(${PROTO_HEADER})
# PROTOBUF_GENERATE_CPP(
#   PROTO_SRC
#   PROTO_HEADER
#   log_kinematics.proto)
# protobuf_generate_python(
#   PROTO_PY
#   log_kinematics.proto)

# # Binary
# ADD_LIBRARY(
#   salamander_msg_log_kinematics
#   SHARED
#   ${PROTO_HEADER}
#   ${PROTO_SRC})
# set_target_properties(
#   salamander_msg_log_kinematics
#   PROPERTIES
#   PUBLIC_HEADER ${PROTO_HEADER})
# set_target_properties(
#   salamander_msg_log_kinematics
#   PROPERTIES
#   LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

# # Install
# include(GNUInstallDirs)
# INSTALL(
#   TARGETS salamander_msg_log_kinematics
#   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#   PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/salamander-msgs)
