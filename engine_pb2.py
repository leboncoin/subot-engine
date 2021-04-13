# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: engine.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='engine.proto',
  package='supportanalyticsengine',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x0c\x65ngine.proto\x12\x16supportanalyticsengine\"\x14\n\x04Text\x12\x0c\n\x04text\x18\x01 \x01(\t\"+\n\x08\x43\x61tegory\x12\x10\n\x08\x63\x61tegory\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x02\"B\n\nCategories\x12\x34\n\ncategories\x18\x01 \x03(\x0b\x32 .supportanalyticsengine.Category2\xbf\x01\n\x06\x45ngine\x12Z\n\x14\x41nalyseMessageLabels\x12\x1c.supportanalyticsengine.Text\x1a\".supportanalyticsengine.Categories\"\x00\x12Y\n\x13\x41nalyseMessageTools\x12\x1c.supportanalyticsengine.Text\x1a\".supportanalyticsengine.Categories\"\x00\x62\x06proto3'
)




_TEXT = _descriptor.Descriptor(
  name='Text',
  full_name='supportanalyticsengine.Text',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='supportanalyticsengine.Text.text', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=40,
  serialized_end=60,
)


_CATEGORY = _descriptor.Descriptor(
  name='Category',
  full_name='supportanalyticsengine.Category',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='category', full_name='supportanalyticsengine.Category.category', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='score', full_name='supportanalyticsengine.Category.score', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=62,
  serialized_end=105,
)


_CATEGORIES = _descriptor.Descriptor(
  name='Categories',
  full_name='supportanalyticsengine.Categories',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='categories', full_name='supportanalyticsengine.Categories.categories', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=107,
  serialized_end=173,
)

_CATEGORIES.fields_by_name['categories'].message_type = _CATEGORY
DESCRIPTOR.message_types_by_name['Text'] = _TEXT
DESCRIPTOR.message_types_by_name['Category'] = _CATEGORY
DESCRIPTOR.message_types_by_name['Categories'] = _CATEGORIES
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Text = _reflection.GeneratedProtocolMessageType('Text', (_message.Message,), {
  'DESCRIPTOR' : _TEXT,
  '__module__' : 'engine_pb2'
  # @@protoc_insertion_point(class_scope:supportanalyticsengine.Text)
  })
_sym_db.RegisterMessage(Text)

Category = _reflection.GeneratedProtocolMessageType('Category', (_message.Message,), {
  'DESCRIPTOR' : _CATEGORY,
  '__module__' : 'engine_pb2'
  # @@protoc_insertion_point(class_scope:supportanalyticsengine.Category)
  })
_sym_db.RegisterMessage(Category)

Categories = _reflection.GeneratedProtocolMessageType('Categories', (_message.Message,), {
  'DESCRIPTOR' : _CATEGORIES,
  '__module__' : 'engine_pb2'
  # @@protoc_insertion_point(class_scope:supportanalyticsengine.Categories)
  })
_sym_db.RegisterMessage(Categories)



_ENGINE = _descriptor.ServiceDescriptor(
  name='Engine',
  full_name='supportanalyticsengine.Engine',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=176,
  serialized_end=367,
  methods=[
  _descriptor.MethodDescriptor(
    name='AnalyseMessageLabels',
    full_name='supportanalyticsengine.Engine.AnalyseMessageLabels',
    index=0,
    containing_service=None,
    input_type=_TEXT,
    output_type=_CATEGORIES,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='AnalyseMessageTools',
    full_name='supportanalyticsengine.Engine.AnalyseMessageTools',
    index=1,
    containing_service=None,
    input_type=_TEXT,
    output_type=_CATEGORIES,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_ENGINE)

DESCRIPTOR.services_by_name['Engine'] = _ENGINE

# @@protoc_insertion_point(module_scope)
