# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import engine_pb2 as engine__pb2


class EngineStub(object):
    """Missing associated documentation comment in .proto file"""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.AnalyseMessageLabels = channel.unary_unary(
                '/supportanalyticsengine.Engine/AnalyseMessageLabels',
                request_serializer=engine__pb2.Text.SerializeToString,
                response_deserializer=engine__pb2.Categories.FromString,
                )
        self.AnalyseMessageTools = channel.unary_unary(
                '/supportanalyticsengine.Engine/AnalyseMessageTools',
                request_serializer=engine__pb2.Text.SerializeToString,
                response_deserializer=engine__pb2.Categories.FromString,
                )


class EngineServicer(object):
    """Missing associated documentation comment in .proto file"""

    def AnalyseMessageLabels(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def AnalyseMessageTools(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_EngineServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'AnalyseMessageLabels': grpc.unary_unary_rpc_method_handler(
                    servicer.AnalyseMessageLabels,
                    request_deserializer=engine__pb2.Text.FromString,
                    response_serializer=engine__pb2.Categories.SerializeToString,
            ),
            'AnalyseMessageTools': grpc.unary_unary_rpc_method_handler(
                    servicer.AnalyseMessageTools,
                    request_deserializer=engine__pb2.Text.FromString,
                    response_serializer=engine__pb2.Categories.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'supportanalyticsengine.Engine', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Engine(object):
    """Missing associated documentation comment in .proto file"""

    @staticmethod
    def AnalyseMessageLabels(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/supportanalyticsengine.Engine/AnalyseMessageLabels',
            engine__pb2.Text.SerializeToString,
            engine__pb2.Categories.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def AnalyseMessageTools(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/supportanalyticsengine.Engine/AnalyseMessageTools',
            engine__pb2.Text.SerializeToString,
            engine__pb2.Categories.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)
