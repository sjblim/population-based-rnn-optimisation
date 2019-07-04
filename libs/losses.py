"""
losses.py


Created by limsi on 03/04/2019
"""

from enum import IntEnum
import tensorflow as tf


class LossTypes(IntEnum):

    MSE = 1,
    BINARY = 2


class LossFunctionHelper:

    _loss_name_map = {LossTypes.BINARY: 'binary',
                      LossTypes.MSE: "mse"}

    @classmethod
    def get_valid_losses(cls):
        return set(cls._loss_name_map.keys())

    @classmethod
    def get_name(cls, loss_type):
        return cls._loss_name_map[loss_type]

    @staticmethod
    def get_output_layer_by_loss(loss_type):

        if loss_type == LossTypes.BINARY:

            return tf.nn.sigmoid

        elif loss_type == LossTypes.MSE:
            return lambda x: x  # linear

        else:
            raise ValueError("Unrecognised loss type: {}".format(loss_type))

    @staticmethod
    def process_outputs_by_loss(outputs, loss_type):

        return outputs

    @staticmethod
    def get_loss_function(loss_type):

        return {
                LossTypes.BINARY: 'binary_crossentropy',
                LossTypes.MSE: 'mse'
                }[loss_type]









