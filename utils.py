import hvac
import spacy


def init_vault(vault_config):
    client = hvac.Client(url=vault_config.get('url'))
    client.auth_approle(vault_config.get('role_id'),
                        vault_config.get('secret_id'))
    return client


def load_local_model(model_name):
    output_dir = "models/{}".format(model_name)
    print("Loading local model from from", output_dir)
    nlp = spacy.load(output_dir)
    return nlp


def load_remote_model(s3_client, bucket, model_name):
    s3_object = s3_client.get_object(Bucket=bucket,
                                     Key=model_name)
    model = s3_object.get("Body").read()

    # test the saved model
    nlp = spacy.blank("fr")

    # add the text classifier to the pipeline
    textcat = nlp.create_pipe(
        "textcat",
        config={"exclusive_classes": True, "architecture": "simple_cnn"}
    )
    tokenizer = nlp.create_pipe("tokenizer")
    nlp.add_pipe(tokenizer)
    nlp.add_pipe(textcat, last=True)
    nlp.from_bytes(model)
    return nlp
