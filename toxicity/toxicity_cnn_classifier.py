import os

import keras
import pandas as pd

from keras.models import Model
from keras.layers import Input, concatenate, Dense, Dropout, BatchNormalization
from keras.layers import Embedding, Conv1D, GlobalMaxPool1D

from keras.callbacks import ModelCheckpoint, EarlyStopping

from toxicity.toxicity_classifier import ToxicityClassifier

from activations.ReLUs import ReLUs


class ToxicityCNNClassifier(ToxicityClassifier):

    def __init__(self, output_dir):
        super().__init__(output_dir)

    def init(self, hyper_parameters):
        super().init(hyper_parameters)

        self.n_conv_1 = hyper_parameters['layer_1_dimensions']
        self.n_conv_2 = hyper_parameters['layer_2_dimensions']
        self.n_conv_3 = hyper_parameters['layer_3_dimensions']
        self.k_conv_1 = hyper_parameters['kernel_1']
        self.k_conv_2 = hyper_parameters['kernel_2']
        self.k_conv_3 = hyper_parameters['kernel_3']

        self.conv_dropout = hyper_parameters['conv_dropout']

        self.dense_1_dimenssions = hyper_parameters['dense_1_dimensions']
        self.dense_dropout = hyper_parameters['dense_dropout']

        self.e_param = hyper_parameters['e_param']
        self.activation_fn = hyper_parameters['activation_fn']

        ReLUs.config(self.e_param)

    def build_model(self):
        input_layer = Input(shape=(self.max_review_length,), dtype='int16', name='input')

        embedding_layer = Embedding(self.unique_words, self.input_dimensions, input_length=self.max_review_length,
                                    name='embedding_1')(input_layer)

        conv_1 = Conv1D(self.n_conv_1, self.k_conv_1, activation=self.activation_fn, name='conv_1')(embedding_layer)
        maxp_1 = GlobalMaxPool1D(name='maxp_1')(conv_1)
        drop_1 = Dropout(self.conv_dropout, name='conv_drop_1')(maxp_1)
        norm_1 = BatchNormalization(name='norm_1')(drop_1)

        conv_2 = Conv1D(self.n_conv_2, self.k_conv_2, activation=self.activation_fn, name='conv_2')(embedding_layer)
        maxp_2 = GlobalMaxPool1D(name='maxp_2')(conv_2)
        drop_2 = Dropout(self.conv_dropout, name='conv_drop_2')(maxp_2)
        norm_2 = BatchNormalization(name='norm_2')(drop_2)

        conv_3 = Conv1D(self.n_conv_3, self.k_conv_3, activation=self.activation_fn, name='conv_3')(embedding_layer)
        maxp_3 = GlobalMaxPool1D(name='maxp_3')(conv_3)
        drop_3 = Dropout(self.conv_dropout, name='conv_drop_3')(maxp_3)
        norm_3 = BatchNormalization(name='norm_3')(drop_3)

        concat = concatenate([norm_1, norm_2, norm_3])

        dense_layer_1 = Dense(self.dense_1_dimenssions, activation=self.activation_fn, name='dense_1')(concat)
        drop_dense_layer_1 = Dropout(self.dense_dropout, name='drop_dense_1')(dense_layer_1)

        output = Dense(self.n_classes, activation='sigmoid', name='output')(drop_dense_layer_1)

        model = Model(input_layer, output)
        print(model.summary())

        return model

    def compile_model(self):
        model = self.build_model()

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        modelCheckPoint = ModelCheckpoint(monitor='val_acc', filepath=self.output_dir + '/weights-multicnn-toxicity.hdf5',
                                               save_best_only=True, mode='max')
        earlyStopping = EarlyStopping(monitor='val_acc', mode='max', patience=self.patience)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        return modelCheckPoint, earlyStopping, model

    def train_model(self):
        X_train, X_valid, y_train, y_valid, X_test_sub = self.preprocess_data()

        modelCheckPoint, earlyStopping, model = self.compile_model()
        model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=2,
                  validation_data=(X_valid, y_valid), callbacks=[modelCheckPoint, earlyStopping])

        saved_model = keras.models.load_model(self.output_dir + '/weights-multicnn-toxicity.hdf5')
        y_hat = saved_model.predict(X_test_sub)
        self.save_submission(y_hat)

    def save_submission(self, y_hat):
        sample_submission = pd.read_csv("data/toxicity/sample_submission.csv")

        sample_submission[self.classes] = y_hat
        sample_submission.to_csv(self.output_dir + '/submission_multicnn.csv', index=False)