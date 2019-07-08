"""
rnn.py


Created by limsi on 11/02/2019
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf

from data.base_data_loaders import DataTypes

from libs.models.base import KerasWrapperInterface
from libs.losses import LossFunctionHelper, LossTypes, TRANSACTION_COSTS


class KerasLSTMWrapper(KerasWrapperInterface):

    def __init__(self, params):

        # Data parameters
        self.time_steps =int(params['rnn_window'])
        self.input_size = int(params['input_size'])
        self.output_size = int(params['output_size'])

        # Network params
        self.loss_type = LossTypes[params['loss_type'].replace("LossTypes.", "")] \
                            if isinstance(params['loss_type'], str) else params['loss_type']
        self.hidden_layer_size = int(params['hidden_layer_size'])
        self.dropout_rate = float(params['dropout_rate'])
        self.max_gradient_norm = float(params['max_gradient_norm'])
        self.learning_rate = float(params['learning_rate'])
        self.minibatch_size = int(params['minibatch_size'])
        self.num_epochs = int(params['num_epochs'])
        self.early_stopping_patience = int(params['early_stopping_patience'])

        # Serialisation Options
        self.name = self.__class__.__name__
        self.model_folder = params['model_folder']

        self.model = self.build_model()

    def build_model(self,  time_steps=None):

        if time_steps is None:
            time_steps=self.time_steps

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(self.hidden_layer_size,
                                       return_sequences=True,
                                       input_shape=(time_steps,
                                                    self.input_size),
                                       dropout=self.dropout_rate,
                                       recurrent_dropout=self.dropout_rate,
                                       stateful=False  # never need this
                                       ))

        model.add(tf.keras.layers.Dropout(self.dropout_rate))

        model.add(tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(self.output_size,
                                          activation=LossFunctionHelper.get_output_layer_by_loss(self.loss_type),
                                          kernel_constraint=tf.keras.constraints.max_norm(3))))

        adam = tf.keras.optimizers.Adam(lr=self.learning_rate, clipnorm=self.max_gradient_norm)

        model.compile(loss=LossFunctionHelper.get_loss_function(self.loss_type),
                      optimizer=adam,
                      metrics=LossFunctionHelper.get_default_metrics(self.loss_type),
                      sample_weight_mode='temporal')

        return model

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fit(self, data_loader):

        raw_data = data_loader.make_rnn_tf_dataset(self.time_steps,
                                                   include_vol_normaliser=self.loss_type==LossTypes.MIN_TURNOVER)

        self._fit_from_data_map(raw_data)

    def _fit_from_data_map(self, raw_data):

        now = dt.datetime.now()
        tmp_identifier = now.strftime("%Y%m%d_%H%M%S_%f")
        tmp_path_name = os.path.join(self.model_folder, '{}_{}_tmp.check'.format(self.name, tmp_identifier))

        # Add relevant callbacks
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience),
                     tf.keras.callbacks.ModelCheckpoint(filepath=tmp_path_name,
                                                        monitor='val_loss',
                                                        save_best_only=True,
                                                        save_weights_only=True)]

        # Additional modifications for turnover regularizer
        if self.loss_type == LossTypes.MIN_TURNOVER:

            # Process train data
            data_chunk = raw_data[DataTypes.TRAIN]
            train_vol_normaliser = data_chunk['vol_normaliser']
            data = data_chunk['inputs']

            labels = np.nan_to_num((data_chunk['outputs'] / train_vol_normaliser), 0.0)
            active_flags = (np.sum(data_chunk['active_entries'], axis=-1) > 0.0) \
                           * train_vol_normaliser[:, :, 0]

            # Process valid data
            data_chunk = raw_data[DataTypes.VALID]
            val_vol_normaliser = data_chunk['vol_normaliser']
            val_data = data_chunk['inputs']

            val_labels = np.nan_to_num(data_chunk['outputs'] / val_vol_normaliser, 0.0)
            val_flags = (np.sum(data_chunk['active_entries'], axis=-1) > 0.0) \
                        * val_vol_normaliser[:, :, 0]

        else:
            # Old methods unchanged)
            data = raw_data[DataTypes.TRAIN]['inputs']
            labels = LossFunctionHelper.process_outputs_by_loss(raw_data[DataTypes.TRAIN]['outputs'], self.loss_type)
            active_flags = (np.sum(raw_data[DataTypes.TRAIN]['active_entries'], axis=-1) > 0.0) * 1.0
            val_data = raw_data[DataTypes.VALID]['inputs']
            val_labels = LossFunctionHelper.process_outputs_by_loss(raw_data[DataTypes.VALID]['outputs'],
                                                                    self.loss_type)
            val_flags = (np.sum(raw_data[DataTypes.VALID]['active_entries'], axis=-1) > 0.0) * 1.0

        self.model.fit(data, labels,
                       sample_weight=active_flags,
                       epochs=self.num_epochs,
                       batch_size=self.minibatch_size,
                       validation_data=(val_data, val_labels, val_flags),
                       callbacks=callbacks,
                       shuffle=True)
        try:
            self.load(tmp_path_name)  # Used to get the best checkpoint in again

        except:
            print("Cannot load {}, skipping ...".format(tmp_path_name))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def predict(self, data_loader, data_type):

        raw_data = data_loader.make_rnn_tf_dataset(self.time_steps)[data_type]
        prediction = self.model.predict(raw_data['inputs'])

        return prediction

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_captured_returns(self, data_loader, data_type, feature_impt_col=None):

        all_data_types = [DataTypes.TRAIN, DataTypes.VALID, DataTypes.TEST]
        raw_data = data_loader.make_single_run_dataset(is_rnf=False,
                                                       all_data_types=all_data_types,
                                                       feature_impt_col=feature_impt_col)[data_type]

        return self._get_captured_returns_from_data(raw_data)

    def _get_captured_returns_from_data(self, raw_data):
        time_steps = raw_data['inputs'].shape[1]

        print("Saving current model to temp file")
        tmp_path_name = os.path.join(self.model_folder, '{}_tmp.check'.format(self.name))
        self.model.save_weights(tmp_path_name, overwrite=True)

        tmp_model = self.build_model(time_steps=time_steps)
        tmp_model.load_weights(tmp_path_name)

        print("Predicting with temp model")
        positions = tmp_model.predict(raw_data['inputs']).flatten()

        if self.loss_type == LossTypes.BINARY:
            positions = 2 * positions - 1.0  # size it between -1 & 1

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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def evaluate(self, data_loader, data_type):

        raw_data = data_loader.make_rnn_tf_dataset(self.time_steps,
                               include_vol_normaliser=self.loss_type in LossFunctionHelper.TURNOVER_TYPES)[data_type]

        return self._evaluate_from_data(raw_data)

    def _evaluate_from_data(self, raw_data):

        # Compute transaction cost adjustment if necessary
        if self.loss_type in LossFunctionHelper.TURNOVER_TYPES:

            vol_normaliser = raw_data['vol_normaliser']

            # Positions & target
            positions = self.model.predict(raw_data['inputs'])
            target = LossFunctionHelper.process_outputs_by_loss(raw_data['outputs'], self.loss_type)
            # Turnover
            turnover = positions.copy()
            # turnover[:, 0, :] = np.zeros(turnover[:, 0, :].shape)  # ignore cost of first position
            turnover[:, 1:, :] -= turnover[:, :-1, :]
            turnover = np.abs(turnover)

            slippage = turnover * TRANSACTION_COSTS * vol_normaliser


        else:
            target = LossFunctionHelper.process_outputs_by_loss(raw_data['outputs'], self.loss_type)
            positions = self.model.predict(raw_data['inputs'])
            slippage = 0.0

        performance = LossFunctionHelper.calc_performance_by_type(target, positions, self.loss_type,
                                                                  transaction_costs=slippage)

        return performance
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
