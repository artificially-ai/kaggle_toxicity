import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text

from sklearn.model_selection import train_test_split


class ToxicityClassifier:

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def init(self, hyper_parameters):
        self.n_classes = hyper_parameters['n_classes']

        self.epochs = hyper_parameters['epochs']
        self.batch_size = hyper_parameters['batch_size']
        self.patience = hyper_parameters['patience']
        self.test_split = hyper_parameters['test_split']

        self.input_dimensions = hyper_parameters['input_dimensions']
        self.unique_words = hyper_parameters['unique_words']
        self.max_review_length = hyper_parameters['max_review_length']
        self.pad_type = self.trunc_type = hyper_parameters['padding']

        self.dense_1_dimenssions = hyper_parameters['dense_1_dimensions']
        self.dense_dropout = hyper_parameters['dense_dropout']

        self.train_df = None
        self.test_df = None
        self.classes = None
        self.modelCheckPoint = None
        self.earlyStopping = None

        self.load_data()

    def load_data(self):
        self.train_df = pd.read_csv('data/toxicity/train.csv')
        self.test_df = pd.read_csv('data/toxicity/test.csv')

    def preprocess_data(self):
        train_sentences_series = self.train_df['comment_text'].fillna("_").values

        # Tokeninze the Training data
        tokenizer = text.Tokenizer(num_words=self.unique_words)
        tokenizer.fit_on_texts(list(train_sentences_series))
        train_tokenized_sentences = tokenizer.texts_to_sequences(train_sentences_series)
        X_train = pad_sequences(train_tokenized_sentences, maxlen=self.max_review_length, padding=self.pad_type,
                                truncating=self.trunc_type, value=0)

        self.classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        y_train = self.train_df[self.classes].values

        # Tokeninze the Test data
        test_sentences_series = self.test_df['comment_text'].fillna("_").values
        test_tokenized_sentences = tokenizer.texts_to_sequences(test_sentences_series)
        X_test_sub = pad_sequences(test_tokenized_sentences, maxlen=self.max_review_length, padding=self.pad_type,
                                   truncating=self.trunc_type, value=0)

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=self.test_split)
        return X_train, X_valid, y_train, y_valid, X_test_sub

    def build_model(self):
        pass

    def compile_model(self):
        pass

    def train_model(self):
        pass

    def save_submission(self, y_hat):
        pass