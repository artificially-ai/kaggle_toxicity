import sys
import json

from toxicity.toxicity_cnn_classifier import ToxicityCNNClassifier
from toxicity.toxicity_lstm_classifier import ToxicityLSTMClassifier

from utils.s3 import S3Utils

if __name__ == '__main__':

    classifiers = {'cnn': ToxicityCNNClassifier('model_output/cnn'), 'lstm': ToxicityLSTMClassifier('model_output/lstm')}

    if len(sys.argv) < 2:
        print('Please, pass the model type you want to execute. for example, "cnn" or "lstm".')
        sys.exit(1)

    model_type = sys.argv[1]
    params = 'hyperparams_%s.json' % model_type
    print('Parameters file:', params)

    hyper_parameters = json.load(open('/data/%s' % params))
    toxicity = classifiers[model_type]
    toxicity.init(hyper_parameters)
    toxicity.train_model()

    S3Utils.upload(model_type)
