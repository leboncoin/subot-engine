#!/usr/bin/env python
# coding: utf8
"""Train a convolutional neural network text classifier on the
IMDB dataset, using the TextCategorizer component. The dataset will be loaded
automatically via Thinc's built-in dataset loader. The model is added to
spacy.pipeline, and predictions are available via `doc.cats`. For more details,
see the documentation:
* Training: https://spacy.io/usage/training

Compatible with: spaCy v2.0.0+
"""
from concurrent import futures
from utils import init_vault, load_local_model, load_remote_model

import os
import grpc
import boto3

import engine_pb2
import engine_pb2_grpc


class EngineServicer(object):
    """Interface of the Service"""

    def __init__(self, env, bucket, vault_config):
        self.env = env
        self.vault = init_vault(vault_config)
        aws_credentials = self.vault.read(
            "apps/team-engprod/support-analytics/{env}/aws".format(env=env))

        self.s3_client = boto3.client("s3",
                                      aws_access_key_id=aws_credentials.get(
                                          'data').get('access_key'),
                                      aws_secret_access_key=aws_credentials.get(
                                          'data').get('secret_key'))
        self.bucket = bucket

        self.tools_model = self.init_model("tools")
        self.labels_model = self.init_model("labels")
        print("servicer is ready to accept connections")

    def init_model(self, model_name):
        print("loading model {}".format(model_name))
        if self.env == "local":
            return load_local_model(model_name)
        return load_remote_model(self.s3_client, self.bucket, model_name)

    def AnalyseMessageLabels(self, request, context):
        """Message Labels analyser
        Returns the labels that match a message
        """
        result = self.labels_model(request.text)
        categories = []
        for k, v in result.cats.items():
            categories.append(engine_pb2.Category(category=k, score=v))
        return engine_pb2.Categories(categories=categories)

    def AnalyseMessageTools(self, request, context):
        """Message Labels analyser
        Returns the labels that match a message
        """
        result = self.tools_model(request.text)
        categories = []
        for k, v in result.cats.items():
            categories.append(engine_pb2.Category(category=k, score=v))
        return engine_pb2.Categories(categories=categories)


def serve():
    print("Retrieving config before serving engine")
    bucket = os.environ.get("BUCKET")
    env = os.environ.get("ENV")

    vault_config = {
        "url": os.environ.get("VAULT_ADDR"),
        "role_id": os.environ.get("VAULT_ROLE_ID"),
        "secret_id": os.environ.get("VAULT_SECRET_ID"),
    }

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    print("Adding engine servicer")
    engine_pb2_grpc.add_EngineServicer_to_server(
        EngineServicer(env, bucket, vault_config), server)
    server.add_insecure_port('[::]:50051')
    print("Starting...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
