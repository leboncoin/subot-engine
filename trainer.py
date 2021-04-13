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
from __future__ import unicode_literals, print_function

import json

from elasticsearch import Elasticsearch
from spacy.util import minibatch, compounding
from utils import init_vault, load_local_model, load_remote_model
from pathlib import Path
import requests

import os
import spacy
import boto3
import random

messages_index = "messages"
# messages_index = "snapshot_messages"

USER_QUERY = {
    "query": {
        "term": {
            "type": "user"
        }
    },
}

MATCH_ALL = {
    "query": {
        "match_all": {}
    },
}


class ModelTrainer(object):
    def __init__(self, env, n_iter, bucket, es_url, vault_config):
        self.vault = init_vault(vault_config)
        self.env = env
        aws_credentials = self.vault.read(
            "apps/team-engprod/support-analytics/{env}/aws".format(env=env))

        self.slack_credentials = self.vault.read(
            "apps/team-engprod/support-analytics/{env}/slack".format(env=env))

        self.es = Elasticsearch(es_url)
        self.n_iter = n_iter
        self.nlp = spacy.blank("fr")
        self.bucket = bucket
        self.s3_client = boto3.client("s3",
                                      aws_access_key_id=aws_credentials.get(
                                          'data').get('access_key'),
                                      aws_secret_access_key=aws_credentials.get(
                                          'data').get('secret_key'))

    def train(self, model_name):
        output = []
        # add the text classifier to the pipeline
        textcat = self.nlp.create_pipe(
            "textcat",
            config={"exclusive_classes": True, "architecture": "simple_cnn"}
        )
        self.nlp.add_pipe(textcat, last=True)

        # add label to text classifier
        for cat in self.fetch_cats(model_name):
            textcat.add_label(cat)

        # load dataset
        print("Loading support messages...")
        (train_texts, train_cats), (dev_texts, dev_cats) = self.load_data(
            model_name)
        print(
            "Using {} examples ({} training, {} evaluation)".format(
                len(train_texts) + len(dev_texts), len(train_texts),
                len(dev_texts)
            )
        )

        train_data = list(
            zip(train_texts, [{"cats": cats} for cats in train_cats]))

        # get names of other pipes to disable them during training
        pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in self.nlp.pipe_names if
                       pipe not in pipe_exceptions]

        with self.nlp.disable_pipes(*other_pipes):  # only train textcat
            optimizer = self.nlp.begin_training()
            print("Training the model...")
            output.append(
                "#\t{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
            max_batch_size = 64
            if len(train_data) < 1000:
                max_batch_size /= 2
            if len(train_data) < 500:
                max_batch_size /= 2
            batch_sizes = compounding(4.0, max_batch_size, 1.001)
            for i in range(self.n_iter):
                losses = {}
                random.shuffle(train_data)
                batches = minibatch(train_data, size=batch_sizes)
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                                    losses=losses)
                with textcat.model.use_params(optimizer.averages):
                    # evaluate on the dev data split off in load_data()
                    scores = self.evaluate(textcat,
                                           dev_texts,
                                           dev_cats)
                output.append(
                    "{0}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}".format(
                        # print a simple table
                        i + 1,
                        losses["textcat"],
                        scores["textcat_p"],
                        scores["textcat_r"],
                        scores["textcat_f"],
                    )
                )

        print("\n".join(output))

        # test the trained model
        test_text = "Hello proctool, mon job integration démarre pas"
        doc = self.nlp(test_text)
        self.display_results(test_text, doc)
        self.save_model(model_name, optimizer)
        self.test_model(model_name, test_text)
        self.nlp.remove_pipe("textcat")
        self.send_report(output, model_name)

    @staticmethod
    def display_results(test_text, doc):
        print(test_text)
        sort_orders = sorted(doc.cats.items(), key=lambda x: x[1], reverse=True)
        for i in sort_orders:
            if i[1] < 0.5:
                break
            print(i[0], str(round(i[1], 2)))

    def save_model(self, model_name, optimizer):
        if self.env == "local":
            self.save_model_to_file(model_name, optimizer)
            return
        with self.nlp.use_params(optimizer.averages):
            model = self.nlp.to_bytes()
        self.s3_client.put_object(Body=model,
                                  Bucket=self.bucket,
                                  Key=model_name)

    def save_model_to_file(self, model_name, optimizer):
        output_dir = "models/{}".format(model_name)
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)

        with self.nlp.use_params(optimizer.averages):
            self.nlp.to_disk(output_dir)
            print("Saved model to", output_dir)

    def test_model(self, model_name, test_string):
        print("Loading from", model_name)
        if self.env == "local":
            nlp = load_local_model(model_name)
        else:
            nlp = load_remote_model(self.s3_client, self.bucket, model_name)

        # test the saved model
        doc = nlp(test_string)
        self.display_results(test_string, doc)

    def fetch_dataset(self, model_name):
        es_response = self.es.search(index=messages_index, body=USER_QUERY,
                                     _source=["text", model_name,
                                              "ai_{}".format(model_name)],
                                     size=10000)

        return es_response.get("hits").get("hits")

    def fetch_cats(self, model_name):
        cats = []
        es_response = self.es.search(index=model_name, body=MATCH_ALL,
                                     size=1000)

        for item in es_response.get("hits").get("hits"):
            cats.append(item.get("_source").get("name"))

        return cats

    def load_data(self, model_name):
        payload = self.fetch_dataset(model_name)
        cats = self.fetch_cats(model_name)

        train_texts = []
        train_cats = []
        dev_text = []
        dev_cats = []

        for item in payload:
            msg = item.get("_source")
            if not msg.get(model_name):
                continue
            common = set(cats).intersection(msg.get(model_name))
            if len(common) > 0:
                if len(msg.get(model_name)) > 1:
                    train_texts.append(msg.get("text"))
                    train_cats.append({cat: cat in common for cat in cats})
                    continue
                if not msg.get("ai_" + model_name):
                    train_texts.append(msg.get("text"))
                    train_cats.append({cat: cat in common for cat in cats})
                    continue
                if msg.get("ai_" + model_name)[0].get(
                        "category") not in msg.get(model_name):
                    train_texts.append(msg.get("text"))
                    train_cats.append({cat: cat in common for cat in cats})
                    continue
                dev_text.append(msg.get("text"))
                dev_cats.append({cat: cat in common for cat in cats})

        return (train_texts, train_cats), (dev_text, dev_cats)

    def evaluate(self, textcat, texts, cats):
        docs = (self.nlp.tokenizer(text) for text in texts)
        fp = fn = 1e-8
        tn = tp = 0.0
        for i, doc in enumerate(textcat.pipe(docs)):
            gold = cats[i]
            for label, score in doc.cats.items():
                if label not in gold:
                    continue

                if score >= 0.5 and gold[label]:
                    tp += 1.0
                elif score >= 0.5 and not gold[label]:
                    fp += 1.0
                elif score < 0.5 and not gold[label]:
                    tn += 1
                elif score < 0.5 and gold[label]:
                    fn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if (precision + recall) == 0:
            f_score = 0.0
        else:
            f_score = 2 * (precision * recall) / (precision + recall)
        return {"textcat_p": precision, "textcat_r": recall,
                "textcat_f": f_score}

    def send_report(self, report, model):
        if self.env != "prd":
            return
        payload = {
            "text": "Training report for model {} : \n{}".format(model, report),
            "channel": self.slack_credentials.get('data').get(
                'training_notification_channel')
        }
        headers = {
            "Authorization": "Bearer {}".format(
                self.slack_credentials.get('data').get(
                    'bot_user_oauth_access_token'))
        }
        requests.post("https://adevinta.slack.com/api/chat.postMessage",
                      data=json.dumps(payload),
                      headers=headers)


def main():
    n_iter = int(os.environ.get("N_ITER"))
    es_url = os.environ.get("ELASTIC_URL")
    bucket = os.environ.get("BUCKET")
    env = os.environ.get("ENV")

    vault_config = {
        "url": os.environ.get("VAULT_ADDR"),
        "role_id": os.environ.get("VAULT_ROLE_ID"),
        "secret_id": os.environ.get("VAULT_SECRET_ID"),
    }
    trainer = ModelTrainer(env, n_iter, bucket, es_url, vault_config)
    for model in ["labels", "tools"]:
        trainer.train(model)


if __name__ == "__main__":
    main()
