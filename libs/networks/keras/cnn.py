"""
cnn.py


Created by limsi on 13/02/2019
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf

from data.base_data_loaders import DataTypes
from libs.models.base import KerasWrapperInterface
from libs.losses import LossFunctionHelper, LossTypes


class KerasCNNWrapper(KerasWrapperInterface):

    def __init__(self, params):

        # Data parameters
        self.lags = params['num_lags']
        self.time_steps = params['rnn_window']
        self.input_size = params['input_size']
        self.output_size = params['output_size']

        # Network params
        self.loss_type = params['loss_type']
        self.hidden_layer_size = params['hidden_layer_size']
        self.dropout_rate = params['dropout_rate']
        self.max_gradient_norm = params['max_gradient_norm']
        self.learning_rate = params['learning_rate']
        self.minibatch_size = params['minibatch_size']
        self.num_epochs = params['num_epochs']
        self.early_stopping_patience = params['early_stopping_patience']

        # Serialisation Options
        self.name = self.__class__.__name__
        self.model_folder = params['model_folder']

        self.model = self.build_model()

    def build_model(self, time_steps=None):

        if time_steps is None:
            time_steps = self.time_steps

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dropout(self.dropout_rate, input_shape=(time_steps, self.input_size)))
        model.add(tf.keras.layers.Conv1D(filters=self.hidden_layer_size,
                                         kernel_size=self.lags,
                                         strides=1,
                                         padding='causal',
                                         dilation_rate=1,
                                         activation='tanh'  # per mlp case
                                         ))
        model.add(tf.keras.layers.Conv1D(filters=self.hidden_layer_size,
                                         kernel_size=self.lags,
                                         strides=1,
                                         padding='causal',
                                         dilation_rate=1,
                                         activation='tanh'  # per mlp case
                                         ))
        model.add(tf.keras.layers.AveragePooling1D(pool_size=self.lags))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(self.hidden_layer_size, activation='tanh'))
        model.add(tf.keras.layers.Dense(self.output_size,
                                        activation=LossFunctionHelper.get_output_layer_by_loss(self.loss_type)))

        adam = tf.keras.optimizers.Adam(lr=self.learning_rate, clipnorm=self.max_gradient_norm)

        model.compile(loss=LossFunctionHelper.get_loss_function(self.loss_type),
                      optimizer=adam,
                      metrics=LossFunctionHelper.get_default_metrics(self.loss_type))

        return model

    def fit(self, data_loader):
        now = dt.datetime.now()
        tmp_identifier = now.strftime("%Y%m%d_%H%M%S_%f")
        tmp_path_name = os.path.join(self.model_folder, '{}_{}_tmp.check'.format(self.name, tmp_identifier))

        # Add relevant callbacks
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience),
                     tf.keras.callbacks.ModelCheckpoint(filepath=tmp_path_name,
                                                        monitor='val_loss',
                                                        save_best_only=True,
                                                        save_weights_only=True)]

        raw_data = data_loader.make_stacked_keras_dataset(self.time_steps)
        data = raw_data[DataTypes.TRAIN]['inputs']
        labels = LossFunctionHelper.process_outputs_by_loss(raw_data[DataTypes.TRAIN]['outputs'], self.loss_type)
        val_data = raw_data[DataTypes.VALID]['inputs']
        val_labels = LossFunctionHelper.process_outputs_by_loss(raw_data[DataTypes.VALID]['outputs'],
                                                                self.loss_type)

        self.model.fit(data, labels,
                       epochs=self.num_epochs,
                       batch_size=self.minibatch_size,
                       validation_data=(val_data, val_labels),
                       callbacks=callbacks,
                       shuffle=True)
        try:
            self.load(tmp_path_name)  # Used to get the best checkpoint in again

        except:
            print("Cannot load {}, skipping ...".format(tmp_path_name))

    def predict(self, data_loader, data_type):

        raw_data = data_loader.make_stacked_keras_dataset(self.time_steps)[data_type]
        prediction = self.model.predict(raw_data['inputs'])

        return prediction

    def evaluate(self, data_loader, data_type):

        # aggregate evaluation
        raw_data = data_loader.make_stacked_keras_dataset(self.time_steps)[data_type]

        positions = self.model.predict(raw_data['inputs'])
        target = LossFunctionHelper.process_outputs_by_loss(raw_data['outputs'], self.loss_type)

        performance = LossFunctionHelper.calc_performance_by_type(target, positions, self.loss_type)

        return performance

    def get_captured_returns(self, data_loader, data_type):

        raw_data = data_loader.make_stacked_keras_dataset(self.time_steps)[data_type]

        positions = self.model.predict(raw_data['inputs']).flatten()

        if self.loss_type == LossTypes.BINARY:
            positions = 2*positions-1.0  # size it between -1 & 1

        elif self.loss_type == LossTypes.MSE:

            # capped & floored @ 2 stds
            original_shape = positions.shape
            positions = np.reshape(np.array([min(max(x, -2), 2) / 2 for x in positions.flatten()]),
                                   original_shape)

        daily_returns = raw_data['outputs'].flatten()
        time_stamps = raw_data['timestamps'].flatten()

        # Upgrade to series
        output_df = pd.DataFrame({'positions': pd.Series(positions, index=time_stamps),
                                  'returns': pd.Series(daily_returns, index=time_stamps)})

        output_df['captured_returns'] = output_df['positions'] * output_df['returns']

        return output_df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class KerasWavenetWrapper(KerasCNNWrapper):

    # Dropout on final layer - https://arxiv.org/pdf/1702.07825.pdf

    def build_model(self, time_steps=None):
        def dilated_convolution(dilation_rate, x, lags):
            tanh = tf.keras.layers.Conv1D(filters=self.hidden_layer_size,
                                          kernel_size=lags,
                                          strides=1,
                                          padding='causal',
                                          dilation_rate=dilation_rate,
                                          activation='tanh'  # per mlp case
                                           )(x)
            sigmoid = tf.keras.layers.Conv1D(filters=self.hidden_layer_size,
                                          kernel_size=lags,
                                          strides=1,
                                          padding='causal',
                                          dilation_rate=dilation_rate,
                                          activation='sigmoid' # per mlp case
                                          )(x)
            return tf.keras.layers.multiply([tanh, sigmoid])

        def add(a, b):
            return tf.keras.layers.add([a, b])

        # Start Layer building
        if time_steps is None:
            time_steps = self.time_steps

        input = tf.keras.layers.Input(shape=(time_steps, self.input_size))

        dropout_input = tf.keras.layers.Dropout(self.dropout_rate)(input)

        weekly = dilated_convolution(5, dropout_input, 5)

        skipped_weekly = add(weekly,  tf.keras.layers.Dense(self.hidden_layer_size,
                                                  activation='linear')(dropout_input))

        monthly = dilated_convolution(21, skipped_weekly, 4)

        skipped_monthly = add(monthly, skipped_weekly)

        quarterly = dilated_convolution(63, skipped_monthly, 3)

        merged = tf.keras.layers.add([weekly, monthly, quarterly])
        last_merged = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(merged)  # take the last value only
        flattened = tf.keras.layers.Dropout(self.dropout_rate)(last_merged)

        mlp1 = tf.keras.layers.Dense(self.hidden_layer_size, activation='tanh')(flattened)
        mlp2 = tf.keras.layers.Dense(self.output_size,
                                     activation=LossFunctionHelper.get_output_layer_by_loss(self.loss_type))(mlp1)

        model = tf.keras.Model(inputs=input, outputs=mlp2)
        adam = tf.keras.optimizers.Adam(lr=self.learning_rate, clipnorm=self.max_gradient_norm)

        model.compile(loss=LossFunctionHelper.get_loss_function(self.loss_type),
                      optimizer=adam,
                      metrics=LossFunctionHelper.get_default_metrics(self.loss_type))

        return model