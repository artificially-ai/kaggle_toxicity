from toxicity.toxicity_classifier import ToxicityClassifier
from subprocess import call

import json

if __name__ == '__main__':
    hyper_parameters = json.load(open('/data/hyperparams.json'))
    toxicity = ToxicityClassifier(hyper_parameters)
    toxicity.train_model()

    call("cp /data/config /root/.aws/.".split(sep=' '))
    call("cp /data/credentials /root/.aws/.".split(sep=' '))

    call("aws s3 cp --recursive /ekholabs/toxicity/model_output/multi-conv s3://ekholabs-kaggle-models".split(sep=' '))
