import os
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.layers.wrappers import Bidirectional

from keras.callbacks import ModelCheckpoint, EarlyStopping

from toxicity.toxicity_classifier import ToxicityClassifier


class ToxicityLSTMClassifier(ToxicityClassifier):

    def init(self, hyper_parameters):
        super().init('model_output/bi_lstm', hyper_parameters)

        self.n_lstm_1 = hyper_parameters['lstm_1_dimenssions']
        self.n_lstm_2 = hyper_parameters['lstm_2_dimenssions']
        self.drop_lstm = hyper_parameters['lstm_dropout']

        self.dense_1_dimenssions = hyper_parameters['dense_1_dimenssions']
        self.dense_dropout = hyper_parameters['dense_dropout']

    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.unique_words, self.input_dimensions, input_length=self.max_review_length))
        model.add(Bidirectional(LSTM(self.n_lstm_1, dropout=self.drop_lstm, return_sequences=True)))
        model.add(Bidirectional(LSTM(self.n_lstm_2, dropout=self.drop_lstm, return_sequences=True)))

        model.add(Dense(self.dense_1_dimenssions, activation=self.activation_fn))
        model.add(Dropout(self.dense_dropout))
        model.add(Dense(self.n_classes, activation='softmax'))

        return model

    def compile_model(self):
        model = self.build_model()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.modelCheckPoint = ModelCheckpoint(filepath=self.output_dir + '/weights-lstm-toxicity.hdf5',
                                               save_best_only=True, mode='min')
        self.earlyStopping = EarlyStopping(mode='min', patience=self.patience)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        return model

    def train_model(self):
        X_train, X_valid, y_train, y_valid, X_test_sub = self.preprocess_data()

        model = self.compile_model()
        model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=2, validation_data=(X_valid, y_valid),
                  callbacks=[self.modelCheckPoint, self.earlyStopping])

        model.save(filepath=self.output_dir + '/model-lstm-toxicity.hdf5')
        y_hat = model.predict(X_test_sub)
        self.save_submission(y_hat)

    def save_submission(self, y_hat):
        sample_submission = pd.read_csv("data/toxicity/sample_submission.csv")

        sample_submission[self.classes] = y_hat
        sample_submission.to_csv(self.output_dir + '/submission_lstm.csv', index=False)
