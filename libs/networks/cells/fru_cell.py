"""
fru_cell.py


Created by limsi on 25/04/2019
"""


import math
import tensorflow as tf
from tensorflow.python.util import nest
import collections
import numpy as np
import pdb

_FRUStateTuple = collections.namedtuple("FRUStateTuple", ("state", "t"))


class FRUStateTuple(_FRUStateTuple):
    """Tuple used by FRU Cells for `state_size`, `zero_state`, and output state.
    Stores two elements: `(state, t)`, in that order. Where `state` is the hidden state
    and `t` is the time step.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (state, t) = self
        if state.dtype != t.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                    (str(state.dtype), str(t.dtype)))
        return state.dtype


class FRUCell(tf.nn.rnn_cell.RNNCell):
    """Implements a simple distribution based recurrent unit that keeps moving
    averages of the mean map embeddings of features of inputs.
    """
    """
    num_stats: phi size 
    freqs: array of w 
    freqs_mask: mask value when frequency is not equal to zero
    output_dims: output size 
    recur_dims: r size 
    seq_len: length of sequence 
    """

    def __init__(self, state_size,
                 seq_len,
                 freqs=None,
                 freqs_mask=1.0,
                 summarize=True, linear_out=False,
                 include_input=False, activation=tf.nn.relu):

        self._num_stats = state_size
        self._recur_dims = state_size
        self._freqs_array = freqs if freqs is not None else np.array([seq_len*i/60 for i in range(60)], dtype=np.float32)[np.newaxis, :, np.newaxis]  # per github
        self._output_dims = state_size
        self._nfreqs = len(self._freqs_array.flatten())
        self._freqs_mask_array = np.array([0.0 if w == 0 and len(self._freqs_array.flatten()) > 1 else freqs_mask for w in self._freqs_array.flatten()])[np.newaxis, :, np.newaxis]
        #print("frequency_mask = ", self._freqs_mask_array)

        with tf.variable_scope(type(self).__name__):
            self._phases = tf.get_variable("phase", [self._nfreqs],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                                 trainable=True, dtype=tf.float32)

            # input_sizes
            self._coeffs = {}


        # Other sutff
        self._summarize = summarize
        self._linear_out = linear_out
        self._activation = activation
        self._include_input = include_input

        # as tensorflow does not feed current time step to __call__
        # I have to manually record it
        self._seq_len = seq_len

        super().__init__()

    """
    nfreqs*num_stats
    """
    @property
    def state_size(self):
        return FRUStateTuple(1, int(self._nfreqs * self._num_stats))  # flipped to prevent dropout on time

    @property
    def output_size(self):
        return self._output_dims

    @property
    def trainable_weights(self):

        weights = [self._phases]
        for k in self._coeffs:

            weights = weights + list(self._coeffs[k])

        return weights

    def __call__(self, inputs, state_tuple, scope=None):
        """
        recur*: r
        state*: mu, state_tuple includes (state, t)
        stats*: phi
        freq*: frequency vector
        """
        cur_time_step, state = state_tuple

        with tf.variable_scope(type(self).__name__):

            def _linear(input_tensor, output_dims, scope, dtype=tf.float32):

                if scope not in self._coeffs:

                    with tf.variable_scope(scope):

                        _, input_dims = input_tensor.get_shape().as_list()

                        matrix = tf.get_variable("Matrix", [input_dims, output_dims],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                                             dtype=dtype),
                                                 dtype=dtype)
                        bias = tf.get_variable(
                            "Bias", [output_dims],
                            dtype=dtype,
                            initializer=tf.constant_initializer(0.0, dtype=dtype))
                        self._coeffs[scope] = (matrix, bias)

                matrix, bias = self._coeffs[scope]

                return tf.matmul(input_tensor, matrix) + bias

                return matrix, bias

            phases = tf.reshape(self._phases, [1, -1, 1])

            # Make statistics on input.
            if self._recur_dims > 0:

                """
                r_t = f(W^r mu_{t-1} + b^r)
                """
                recur_output = self._activation(
                    _linear(state, self._recur_dims, scope='recur_feats'
                ), name='recur_feats_act')
                """
                phi_t = W^phi r_t + W^x x_t + b^phi 
                """
                stats = self._activation(_linear(
                    tf.concat([inputs, recur_output], axis=-1), self._num_stats, scope='stats',
                ), name='stats_act')
            else:
                stats = self._activation(_linear(
                    inputs, self._num_stats, scope='stats'
                ), name='stats_act')
            # Compute moving averages of statistics for the state.
            with tf.variable_scope('out_state'):
                state_tensor = tf.reshape(
                    state, [-1, self._nfreqs, self._num_stats], 'state_tensor'
                )
                stats_tensor = tf.reshape(
                    stats, [-1, 1, self._num_stats], 'stats_tensor'
                )
                #cur_time_step = tf.Print(cur_time_step, [cur_time_step], message="cur_time_step = ")
                """
                mu_t = mask*mu_{t-1} + cos(2*pi*w*t/T + 2*pi*phase)*phi_t
                """
                out_state = tf.reshape(self._freqs_mask_array*state_tensor +
                                       1.0/self._seq_len*tf.cos(2.0*math.pi/self._seq_len*tf.reshape(cur_time_step, shape=[-1, 1, 1])*self._freqs_array + 2.0*math.pi*phases)*stats_tensor,
                                       [-1, self.state_size[-1]], 'out_state')
            # Compute the output.
            if self._include_input:
                output_vars = [out_state, inputs]
            else:
                output_vars = out_state
            """
            o_t = W^o mu_t + b^o
            """
            output = _linear(
                output_vars, self._output_dims, scope='output'
            )
            if not self._linear_out:
                output = self._activation(output, name='output_act')
            # update time step
            out_state_tuple = FRUStateTuple(cur_time_step+1, out_state)


        """
        o_t and mu_t 
        """
        return output, out_state_tuple
