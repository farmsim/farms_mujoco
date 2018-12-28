# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gps.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import time_pb2 as time__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='gps.proto',
  package='gazebo.msgs',
  syntax='proto2',
  serialized_pb=_b('\n\tgps.proto\x12\x0bgazebo.msgs\x1a\ntime.proto\"\xbc\x01\n\x03GPS\x12\x1f\n\x04time\x18\x01 \x02(\x0b\x32\x11.gazebo.msgs.Time\x12\x11\n\tlink_name\x18\x02 \x02(\t\x12\x14\n\x0clatitude_deg\x18\x03 \x02(\x01\x12\x15\n\rlongitude_deg\x18\x04 \x02(\x01\x12\x10\n\x08\x61ltitude\x18\x05 \x02(\x01\x12\x15\n\rvelocity_east\x18\x06 \x01(\x01\x12\x16\n\x0evelocity_north\x18\x07 \x01(\x01\x12\x13\n\x0bvelocity_up\x18\x08 \x01(\x01')
  ,
  dependencies=[time__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_GPS = _descriptor.Descriptor(
  name='GPS',
  full_name='gazebo.msgs.GPS',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='time', full_name='gazebo.msgs.GPS.time', index=0,
      number=1, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='link_name', full_name='gazebo.msgs.GPS.link_name', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='latitude_deg', full_name='gazebo.msgs.GPS.latitude_deg', index=2,
      number=3, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='longitude_deg', full_name='gazebo.msgs.GPS.longitude_deg', index=3,
      number=4, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='altitude', full_name='gazebo.msgs.GPS.altitude', index=4,
      number=5, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='velocity_east', full_name='gazebo.msgs.GPS.velocity_east', index=5,
      number=6, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='velocity_north', full_name='gazebo.msgs.GPS.velocity_north', index=6,
      number=7, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='velocity_up', full_name='gazebo.msgs.GPS.velocity_up', index=7,
      number=8, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=39,
  serialized_end=227,
)

_GPS.fields_by_name['time'].message_type = time__pb2._TIME
DESCRIPTOR.message_types_by_name['GPS'] = _GPS

GPS = _reflection.GeneratedProtocolMessageType('GPS', (_message.Message,), dict(
  DESCRIPTOR = _GPS,
  __module__ = 'gps_pb2'
  # @@protoc_insertion_point(class_scope:gazebo.msgs.GPS)
  ))
_sym_db.RegisterMessage(GPS)


# @@protoc_insertion_point(module_scope)
