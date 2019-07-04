"""
phased_lstm.py


Created by limsi on 26/04/2019
"""

import tensorflow as tf
from libs.networks.lstm import BaseLSTMModel


class PhasedLSTMCellWrapper(tf.contrib.rnn.PhasedLSTMCell):

    def __init__(self,
                 num_units,
                 use_peepholes=False,
                 leak=0.001,
                 ratio_on=0.1,
                 trainable_ratio_on=True,
                 period_init_min=1.0,
                 period_init_max=1000.0,
                 reuse=None,
                 dt=int(10e-6)):

        super().__init__(num_units,
                         use_peepholes,
                         leak,
                         ratio_on,
                         trainable_ratio_on,
                         period_init_min,
                         period_init_max,
                         reuse)

        # make time become a state property like the FRU
        self.dt = dt

    @property
    def state_size(self):

        c_size, h_size = super().state_size

        return tf.nn.rnn_cell.LSTMStateTuple(c_size+1, h_size)

    def call(self, inputs, state):

        (c_combined, h) = state

        c, t = c_combined[:, :-1], c_combined[:, -1:]

        combined_inputs = (t, inputs)

        new_h, new_state = super().call(combined_inputs, (c, h))

        # Update time param
        c, h = new_state
        output_state = tf.nn.rnn_cell.LSTMStateTuple(tf.concat([c, t+self.dt], axis=-1), h)

        return new_h, output_state


class PhasedLSTM(BaseLSTMModel):

    def __init__(self, params, sess, StateInitClass, variable_scope_suffix=""):

        self.dt = float(params['time_increment'])

        super().__init__(params, sess, StateInitClass, variable_scope_suffix)

    def _make_rnn_cell(self):

        with tf.variable_scope(self.variable_scope_name):
            cell = PhasedLSTMCellWrapper(self.hidden_layer_size,
                                         dt=self.dt)

        return cell

    def _format_state_placeholder(self, state_holder):

        c, h = self.rnn_cell.zero_state(1, dtype=tf.float32)

        c_dims = c.get_shape().as_list()[-1]

        c, t, h = state_holder[:, :c_dims-1], state_holder[:, c_dims-1:c_dims], state_holder[:, c_dims:]

        return tf.nn.rnn_cell.LSTMStateTuple(tf.concat([c, t*0], axis=-1), h)  # always set first time step to 0

    def _format_final_state(self, final_state):

        c, h = final_state

        return tf.concat([c, h], axis=-1)