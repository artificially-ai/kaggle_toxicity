import sys
import json

from toxicity.toxicity_cnn_classifier import ToxicityClassifier
from toxicity.toxicity_cnn_classifier import ToxicityCNNClassifier
from toxicity.toxicity_lstm_classifier import ToxicityLSTMClassifier

from subprocess import call

if __name__ == '__main__':

    classifiers = {'cnn' : ToxicityCNNClassifier(), 'lstm' : ToxicityLSTMClassifier() }

    if len(sys.argv) < 2:
        print('Please, pass the model type you want to execute. for example, "cnn" or "lstm".')
        sys.exit(1)

    model_type = sys.argv[1]
    hyperparameter = 'hyperparams_%s.json' % model_type

    hyper_parameters = json.load(open('/data/%s' % hyperparameter))
    toxicity = classifiers[model_type]
    toxicity.init(hyper_parameters)
    toxicity.train_model()

    call("cp /data/config /root/.aws/.".split(sep=' '))
    call("cp /data/credentials /root/.aws/.".split(sep=' '))

    call("aws s3 cp --recursive /ekholabs/toxicity/model_output/" + model_type +" s3://ekholabs-kaggle-models".split(sep=' '))
