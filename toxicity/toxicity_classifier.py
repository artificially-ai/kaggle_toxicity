from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text
from keras.models import Model
from keras.layers import Input, concatenate, Dense, Dropout
from keras.layers import Embedding, Conv1D, SpatialDropout1D, GlobalMaxPool1D

from keras.callbacks import ModelCheckpoint, EarlyStopping

import os

from sklearn.model_selection import train_test_split

import pandas as pd

from activations.ReLUs import ReLUs


class ToxicityClassifier:

    def __init__(self, hyper_parameters):
        self.output_dir = 'model_output/multi-conv'

        self.classes = hyper_parameters['classes']

        self.epochs = hyper_parameters['epochs']
        self.batch_size = hyper_parameters['batch_size']
        self.patience = hyper_parameters['patience']
        self.activation_fn = hyper_parameters['activation_fn']
        self.test_split = hyper_parameters['test_split']
        self.val_split = hyper_parameters['val_split']

        self.input_dimensions = hyper_parameters['input_dimensions']
        self.unique_words = hyper_parameters['unique_words']
        self.max_review_length = hyper_parameters['max_review_length']
        self.pad_type = self.trunc_type = hyper_parameters['padding']
        self.embedding_dropout = hyper_parameters['embedding_dropout']

        self.n_conv_1 = hyper_parameters['layer_1_dimenssions']
        self.n_conv_2 = hyper_parameters['layer_2_dimenssions']
        self.n_conv_3 = hyper_parameters['layer_3_dimenssions']
        self.k_conv_1 = int(self.n_conv_1 / (self.n_conv_1 / 3))
        self.k_conv_2 = int(self.n_conv_2 / self.n_conv_1)
        self.k_conv_3 = int(self.n_conv_3 / self.n_conv_1)

        self.dense_1_dimenssions = hyper_parameters['dense_1_dimenssions']
        self.dense_2_dimenssions = hyper_parameters['dense_2_dimenssions']
        self.dense_dropout = hyper_parameters['dense_dropout']

        self.train_df = None
        self.test_df = None
        self.classes = None
        self.modelCheckPoint = None
        self.earlyStopping = None

        ReLUs.config()

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
        input_layer = Input(shape=(self.max_review_length,), dtype='int16', name='input')

        embedding_layer = Embedding(self.unique_words, self.input_dimensions, input_length=self.max_review_length, name='embedding_1')(
            input_layer)
        drop_embed_layer = SpatialDropout1D(self.embedding_dropout, name='drop_embed')(embedding_layer)

        conv_1 = Conv1D(self.n_conv_1, self.k_conv_1, activation=self.activation_fn, name='conv_1')(drop_embed_layer)
        maxp_1 = GlobalMaxPool1D(name='maxp_1')(conv_1)

        conv_2 = Conv1D(self.n_conv_2, self.k_conv_2, activation=self.activation_fn, name='conv_2')(drop_embed_layer)
        maxp_2 = GlobalMaxPool1D(name='maxp_2')(conv_2)

        conv_3 = Conv1D(self.n_conv_3, self.k_conv_3, activation=self.activation_fn, name='conv_3')(drop_embed_layer)
        maxp_3 = GlobalMaxPool1D(name='maxp_3')(conv_3)

        concat = concatenate([maxp_1, maxp_2, maxp_3])

        dense_layer_1 = Dense(self.dense_1_dimenssions, activation=self.activation_fn, name='dense_1')(concat)
        drop_dense_layer_1 = Dropout(self.dense_dropout, name='drop_dense_1')(dense_layer_1)
        dense_layer_2 = Dense(self.dense_2_dimenssions, activation=self.activation_fn, name='dense_2')(drop_dense_layer_1)
        drop_dense_layer_2 = Dropout(self.dense_dropout, name='drop_dense_2')(dense_layer_2)

        predictions = Dense(self.classes, activation='sigmoid', name='output')(drop_dense_layer_2)

        return Model(input_layer, predictions)

    def compile_model(self):
        model = self.build_model()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.modelCheckPoint = ModelCheckpoint(filepath=self.output_dir + '/weights-multicnn-toxicity.hdf5', save_best_only=True,
                                          mode='min')
        self.earlyStopping = EarlyStopping(mode='min', patience=self.patience)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        return model

    def save_submission(self, y_hat):
        sample_submission = pd.read_csv("data/toxicity/sample_submission.csv")

        sample_submission[self.classes] = y_hat
        sample_submission.to_csv(self.output_dir + 'submission_multicnn.csv', index=False)

    def train_model(self):
        X_train, X_valid, y_train, y_valid, X_test_sub = self.preprocess_data()

        model = self.compile_model()
        model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=2, validation_data=(X_valid, y_valid),
                  callbacks=[self.modelCheckPoint, self.earlyStopping])

        model.save(filepath=self.output_dir + '/model-multicnn-toxicity.hdf5')
        y_hat = model.predict(X_test_sub)
        self.save_submission(y_hat)
