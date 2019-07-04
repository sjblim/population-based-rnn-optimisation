"""
lstm.py


Created by limsi on 03/04/2019
"""


import tensorflow as tf
import numpy as np
from enum import IntEnum

from tensorflow.python.ops import array_ops

import libs.net_helpers as helpers

from libs.losses import LossTypes, LossFunctionHelper


_ACTIVATION_MAP = {'sigmoid': tf.nn.sigmoid,
                   'elu': tf.nn.elu,
                   'tanh': tf.nn.tanh,
                   'linear': lambda x: x}


class BaseLSTMModel:

    def __init__(self, params, sess, StateInitClass, variable_scope_suffix=""):

        # Other required tensorflow components
        self.variable_scope_name = "{}{}".format(self.__class__.__name__, variable_scope_suffix)
        self.sess = sess

        # Data params
        self.input_size = np.int(params['input_size'])
        self.output_size = np.int(params['output_size'])

        # Training prams
        self.loss_type = LossTypes[params['loss_type'].replace("LossTypes.", "")] \
            if isinstance(params['loss_type'], str) else params['loss_type']
        self.epochs = int(params['num_epochs'])
        self.minibatch_size = int(params['minibatch_size'])
        self.learning_rate = float(params['learning_rate'])
        self.max_global_norm = float(params['max_gradient_norm'])
        self.early_stopping_patience = int(params['early_stopping_patience'])

        with tf.variable_scope(self.variable_scope_name):
            self.global_step = tf.get_variable('global_step_tfrnn',
                                               initializer=0,
                                               dtype=np.int32,
                                               trainable=False)

        # Network params
        self.dropout_rate = np.float(params['dropout_rate'])
        self.hidden_layer_size = int(params['hidden_layer_size'])

        self.rnn_cell = None
        self.output_w = None
        self.output_b = None
        self.memory_activation = _ACTIVATION_MAP[params['hidden_activation']]
        self.output_activation = LossFunctionHelper.get_output_layer_by_loss(self.loss_type)

        self.init_network_components()

        # State initialiser
        with tf.variable_scope(self.variable_scope_name):  # so it loads with the model
            self.state_initialiser = StateInitClass(self.get_initial_state_dims(), params, sess)

        # Create placeholder graph
        self. create_warmup_network()

        print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Creating {} with:".format(self.variable_scope_name))
        for k in params:
            print("#", k, str(params[k]))
        print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# In[*] Graph Construction Elements

    def _make_rnn_cell(self):
        cell = tf.nn.rnn_cell.LSTMCell(self.hidden_layer_size,
                                       activation=self.memory_activation,
                                       state_is_tuple=True,
                                       name=self.variable_scope_name)

        return cell

    def init_network_components(self):

        # Network bits
        with tf.variable_scope(self.variable_scope_name):
            self.rnn_cell = self._make_rnn_cell()
            self.output_w = tf.get_variable("Output_W",
                                            [self.hidden_layer_size, self.output_size],
                                            dtype=tf.float32)
            self.output_b = tf.get_variable("Output_b",
                                            [self.output_size],
                                            dtype=tf.float32)

    def create_warmup_network(self):

        # Helps to load graph later - not explicitly required...

        # Load small prediction graph to get variables onto sess
        test_chunk = {'inputs': np.zeros((2, 5, self.input_size)),
                      'outputs': np.zeros((2, 5, self.output_size)),
                      'sequence_lengths': np.zeros(2),
                      'active_entries': np.zeros((2, 5, 1))}

        _ = self.get_prediction_graph(test_chunk)  # so variables are always put on the graph

    def _add_dropout(self, with_dropout, cell):

        # Setup graph now
        if with_dropout:
            keep_probs = (1 - self.dropout_rate)
        else:
            keep_probs = 1.0

        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                             input_keep_prob=keep_probs,
                                             output_keep_prob=keep_probs,
                                             state_keep_prob=keep_probs,
                                             variational_recurrent=True,
                                             input_size=self.input_size,
                                             dtype=tf.float32)

        return cell

    def _build_state_graph(self,
                           data_chunk,
                           initial_states,
                           with_dropout=True,
                           cell_to_use=None):

        input_minibatch = tf.cast(data_chunk['inputs'], tf.float32)
        sequence_lengths = tf.cast(data_chunk['sequence_lengths'], tf.int32)

        input_shapes = input_minibatch.get_shape()
        time_steps = input_shapes.as_list()[1]

        # setup cell
        cell = cell_to_use if cell_to_use is not None else self.rnn_cell
        cell = self._add_dropout(with_dropout, cell)

        # Format initial states
        vals, states = tf.nn.dynamic_rnn(cell,
                                         input_minibatch,
                                         initial_state=initial_states,
                                         dtype=tf.float32,
                                         sequence_length=sequence_lengths,
                                         scope=self.variable_scope_name)
        return vals, states

    def _build_prediction_graph(self,
                                data_chunk,
                                initial_states,
                                with_dropout=True,
                                cell_to_use=None):

        vals, states = self._build_state_graph(data_chunk,
                                               initial_states,
                                               with_dropout,
                                               cell_to_use)

        # Add linear output layer
        val_shape = vals.get_shape().as_list()
        flattened = tf.reshape(vals, (-1, val_shape[-1]))
        flat_outputs = self.output_activation(tf.matmul(flattened, self.output_w) + self.output_b)
        outputs = tf.reshape(flat_outputs, [-1, val_shape[1], self.output_size])

        return outputs, states

    def _calc_loss_function(self,
                            predictions,
                            output_minibatch,
                            active_entries,
                            specific_axis=None):

        performance_metric = LossFunctionHelper.get_name(self.loss_type)

        loss = helpers.calc_loss_function(performance_metric,
                                          predictions,
                                          output_minibatch,
                                          active_entries,
                                          specific_axis)

        return loss

    def _placeholders_from_data(self, raw_dataset):
        data_chunk = {}
        for k in raw_dataset:
            shape = list(raw_dataset[k].shape)
            shape[0] = None  # for flexibility
            data_chunk[k] = tf.placeholder(tf.float32,
                                           shape=shape,
                                           name=k)
        return data_chunk

# In[*] Graph
    def get_training_graph(self,
                           raw_dataset):

        print("Getting training graph")

        data_chunk = self._placeholders_from_data(raw_dataset)

        output_minibatch = tf.cast(data_chunk['outputs'], tf.float32)
        active_entries = tf.cast(data_chunk['active_entries'], tf.float32)

        # Setup initial states
        state_holder = self.state_initialiser.get_tensorflow_variable()
        initial_states = self._format_state_placeholder(state_holder)

        # Get unpack predictions
        predictions, final_states = self._build_prediction_graph(data_chunk,
                                                                 initial_states=initial_states,
                                                                 with_dropout=True)

        # Setup loss functions and optimisers
        print("Getting loss functions")
        loss = self._calc_loss_function(predictions,
                                        output_minibatch,
                                        active_entries)

        print("Setting up optimiser")
        optimiser = helpers.get_optimization_graph(loss,
                                                   learning_rate=self.learning_rate,
                                                   max_global_norm=self.max_global_norm,
                                                   global_step=self.global_step)
        # Parcel outputs
        handles = {'loss': loss,
                   'optimiser': optimiser,
                   'initial_states': self.state_initialiser.placeholder,
                   'final_states': self._format_final_state(final_states),
                   'data_placeholders': data_chunk}

        return handles

    def get_validation_graph(self, raw_dataset):

        return self.get_evaluation_graph(raw_dataset)

    def get_evaluation_graph(self, raw_dataset):

        print("Getting evaluation graph")

        data_chunk = self._placeholders_from_data(raw_dataset)

        # Setup initial states
        state_holder = self.state_initialiser.get_tensorflow_variable()
        initial_states = self._format_state_placeholder(state_holder)

        predictions, final_states \
            = self._build_prediction_graph(data_chunk,
                                           initial_states=initial_states,
                                           with_dropout=False)

        counts = tf.cast(tf.reduce_sum(data_chunk['active_entries']), tf.float32)  # for cases with multiple outputs

        active_entries = tf.cast(data_chunk['active_entries'], tf.float32)
        outputs = tf.cast(data_chunk['outputs'], tf.float32)

        loss = self._calc_loss_function(predictions, outputs, active_entries) * counts

        output_map = {'total_loss': loss,
                      'counts': counts,
                      'initial_states': self.state_initialiser.placeholder,
                      'final_states': self._format_final_state(final_states),
                      'data_placeholders': data_chunk}

        return output_map

    def get_prediction_graph(self, raw_dataset):

        print("Getting prediction graph")

        data_chunk = self._placeholders_from_data(raw_dataset)

        # Setup initial states
        state_holder = self.state_initialiser.get_tensorflow_variable()
        initial_states = self._format_state_placeholder(state_holder)

        predictions, final_states \
            = self._build_prediction_graph(data_chunk,
                                           initial_states=initial_states,
                                           with_dropout=False)

        output_map = {'predictions': predictions,
                      'initial_states': self.state_initialiser.placeholder,
                      'final_states': self._format_final_state(final_states),
                      'data_placeholders': data_chunk}

        return output_map

    def get_neuroevolution_graph(self, raw_dataset, use_evaluation=True):

        with tf.variable_scope(self.variable_scope_name):

            # Pull out RNN weights
            trainable_weights = list(self.rnn_cell.trainable_weights)
            # kernel_weight, kernel_bias = self.rnn_cell.trainable_weights
            weight_variables = trainable_weights + [self.output_w, self.output_b]
            num_elems = [np.prod(w.get_shape().as_list()) for w in weight_variables]

            # Wire placeholders into variables
            total_elems = np.sum(num_elems)
            weight_placeholders = tf.placeholder(tf.float32,
                                          shape=total_elems,
                                          name="rnn_weights")

            cum_num_elems = np.cumsum(num_elems)
            assignment_ops = []
            prev_elem_idx = 0
            for i, elem_idx in enumerate(cum_num_elems):
                cur_variable = weight_variables[i]
                cur_placeholder = tf.reshape(weight_placeholders[prev_elem_idx: elem_idx], cur_variable.shape)
                assign = tf.assign(cur_variable, cur_placeholder)

                assignment_ops.append(assign)
                prev_elem_idx = elem_idx

            # Get the right output graph
            output_map = self.get_evaluation_graph(raw_dataset) if use_evaluation \
                            else self.get_prediction_graph(raw_dataset)

        # Tag on weight update elements
        output_map['weight_placeholder'] = weight_placeholders
        output_map['weight_assignment_ops'] = assignment_ops

        return output_map


    # In[*]: State initialisation functions
    def get_initial_state_dims(self):
        c, h = self.rnn_cell.zero_state(1, dtype=tf.float32)

        dims = c.get_shape().as_list()[-1] + h.get_shape().as_list()[-1]

        return dims
    """
    def _get_state_init_placeholder(self):

        dims = self.get_initial_state_dims()

        with tf.variable_scope(self.variable_scope_name):
            state_holder = tf.placeholder(tf.float32, shape=[None, dims], name="initial_state")

        return state_holder
    """

    def _format_state_placeholder(self, state_holder):

        c, h = self.rnn_cell.zero_state(1, dtype=tf.float32)

        c_dims = c.get_shape().as_list()[-1]

        c, h = state_holder[:, :c_dims], state_holder[:, c_dims:]

        return tf.nn.rnn_cell.LSTMStateTuple(c, h)

    def _format_final_state(self, final_state):

        c, h = final_state

        return tf.concat([c, h], axis=-1)

# In[*]: Saving and loading
    def _get_checkpoint_name(self, is_best_model):

        variable_scope_name = self.__class__.__name__

        if is_best_model:
            cp_name = variable_scope_name + "_best"
        else:
            cp_name = variable_scope_name + "_tmp"
        return cp_name

    def save(self, model_folder, is_best_model=True):

        variable_scope_name = self.__class__.__name__

        cp_name = self._get_checkpoint_name(is_best_model)
        helpers.save(self.sess, model_folder, cp_name, scope=variable_scope_name)

    def load(self, model_folder, use_best_model=True):
        variable_scope_name = self.__class__.__name__
        cp_name = self._get_checkpoint_name(use_best_model)
        helpers.load(self.sess, model_folder, cp_name, scope=variable_scope_name)

# In[*]: Hyperparams
    @staticmethod
    def get_hyperparm_choices():

        return {'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
                'hidden_layer_size': [5, 10, 20, 40, 80, 160],
                'minibatch_size': [256, 512, 1024, 2048],
                'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                'max_gradient_norm': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                'state_init_mlp_multiplier': [0.25, 0.5, 1.0, 2.0, 4.0]
                }
