import os
import pandas as pd

from keras.models import Input, Model
from keras.layers import Dense, Embedding, LSTM, concatenate, Flatten
from keras.layers.wrappers import Bidirectional

from keras.callbacks import ModelCheckpoint, EarlyStopping

from toxicity.toxicity_classifier import ToxicityClassifier


class ToxicityLSTMClassifier(ToxicityClassifier):

    def __init__(self, output_dir):
        super().__init__(output_dir)

    def init(self, hyper_parameters):
        super().init(hyper_parameters)

        self.n_lstm_1 = hyper_parameters['lstm_1_dimensions']
        self.n_lstm_2 = hyper_parameters['lstm_2_dimensions']
        self.drop_lstm = hyper_parameters['lstm_dropout']

    def build_model(self):
        input_layer = Input(shape=(self.max_review_length,), dtype='int16', name='input')
        embedding_layer = Embedding(self.unique_words, self.input_dimensions, input_length=self.max_review_length,
                                    name='embedding_1')(input_layer)

        bi_lstm_1 = Bidirectional(LSTM(self.n_lstm_1, dropout=self.drop_lstm, return_sequences=True,
                                       name='bi_lstm_1'))(embedding_layer)
        bi_lstm_2 = Bidirectional(LSTM(self.n_lstm_2, dropout=self.drop_lstm, return_sequences=True,
                                       name='bi_lstm_2'))(embedding_layer)

        concat = concatenate([bi_lstm_1, bi_lstm_2])

        flat = Flatten()(concat)
        output = Dense(self.n_classes, activation='sigmoid', name='output')(flat)

        return Model(input_layer, output)

    def compile_model(self):
        model = self.build_model()
        print(model.summary())

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        modelCheckPoint = ModelCheckpoint(filepath=self.output_dir + '/weights-lstm-toxicity.hdf5',
                                               save_best_only=True, mode='min')
        earlyStopping = EarlyStopping(mode='min', patience=self.patience)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        return modelCheckPoint, earlyStopping, model

    def train_model(self):
        X_train, X_valid, y_train, y_valid, X_test_sub = self.preprocess_data()

        modelCheckPoint, earlyStopping, model = self.compile_model()
        model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=2,
                  validation_data=(X_valid, y_valid), callbacks=[modelCheckPoint, earlyStopping])

        model.save(filepath=self.output_dir + '/model-lstm-toxicity.hdf5')
        y_hat = model.predict(X_test_sub)
        self.save_submission(y_hat)

    def save_submission(self, y_hat):
        sample_submission = pd.read_csv("data/toxicity/sample_submission.csv")

        sample_submission[self.classes] = y_hat
        sample_submission.to_csv(self.output_dir + '/submission_lstm.csv', index=False)
