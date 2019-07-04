"""
fru.py


Created by limsi on 25/04/2019
"""

import tensorflow as tf

from libs.networks.lstm import BaseLSTMModel
from libs.networks.cells.fru_cell import FRUCell, FRUStateTuple


class FRUModel(BaseLSTMModel):

    def _make_rnn_cell(self):
        cell = FRUCell(self.hidden_layer_size,

                       self.unroll_length,
                       activation=self.memory_activation)

        return cell

    def __init__(self, params, sess, StateInitClass, variable_scope_suffix=""):

        self.unroll_length = float(params['rnn_window'])

        super().__init__(params, sess, StateInitClass, variable_scope_suffix)

    def _format_state_placeholder(self, state_holder):

        t, c = state_holder[:, :1], state_holder[:, 1:]

        return FRUStateTuple(t*0, c)  # always set first time step to 0

    def _format_final_state(self, final_state):

        t, c = final_state

        return tf.concat([t, c], axis=-1)

    def _add_dropout(self, with_dropout, cell):

        # Setup graph now
        if with_dropout:
            keep_probs = (1 - self.dropout_rate)
        else:
            keep_probs = 1.0

        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                             input_keep_prob=keep_probs,
                                             output_keep_prob=keep_probs,
                                             state_keep_prob=1.0,
                                             variational_recurrent=True,
                                             input_size=self.input_size,
                                             dtype=tf.float32)
        return cell