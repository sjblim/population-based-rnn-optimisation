# -*- coding: utf-8 -*-
"""
20180107 Joint Modelling with RNNs: 


Created on 11/1/2018 1:27 PM
@author: Bryan
"""

import os
import pathlib
import logging
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


""" 
General
"""


def get_default_tensorflow_config(tf_device='gpu'):

    if tf_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # for training on cpu
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})

    else:
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
        tf_config.gpu_options.allow_growth = True

    return tf_config


def linear(input_, output_size, scope=None, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear", reuse=tf.AUTO_REUSE) as cur_scope:
        matrix = tf.get_variable("Matrix", [shape[-1], output_size], tf.float32,
                                 tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


# Loss functions
def calc_binary_cross_entropy(probs, outputs, weights=1):

    return -tf.reduce_mean((outputs * tf.log(probs +1e-8)
             + (1-outputs)* tf.log(1-probs +1e-8)) * weights)


def calc_loss_function(performance_metric, predictions, output_minibatch, active_entries, specific_axis=None):
    # Compute loss function
    if performance_metric == "mse":
        loss = tf.square(predictions - output_minibatch) * active_entries  # cos some zero entires

    elif performance_metric == "binary":

        loss = (output_minibatch * -tf.log(predictions + 1e-8) \
                + (1 - output_minibatch) * -tf.log(1 - predictions + 1e-8)) \
               * active_entries
    else:
        raise ValueError("Unknown performance metric {}".format(performance_metric))

    if specific_axis is None:
        loss = tf.reduce_sum(loss) / tf.reduce_sum(active_entries)
    else:
        loss = tf.reduce_sum(loss, axis=specific_axis) / tf.reduce_sum(active_entries, axis=specific_axis)

    return loss


def tf_kl_gaussgauss(enc_mu, enc_sigma, prior_mu, prior_sigma, active_entries):
    active_time_steps = tf.expand_dims(tf.reduce_max(active_entries, axis=-1), -1)

    with tf.variable_scope("kl_gaussgauss"):
        kl_loss = tf.reduce_sum(
            0.5 * (2 * tf.log(tf.maximum(1e-9, prior_sigma), name='log_sigma_2')
                   - 2 * tf.log(tf.maximum(1e-9, enc_sigma), name='log_sigma_1')
                   + (tf.square(enc_sigma) + tf.square(enc_mu - prior_mu))
                   / tf.maximum(1e-9, (tf.square(prior_sigma)))
                   - 1)
            * active_time_steps) \
                  / tf.reduce_sum(active_time_steps)

    return kl_loss  # both terms scaled up by number of samples

# Time slicing
def last_relevant_time_slice(output, sequence_length):

    shape = output.get_shape()

    if len(shape) == 3:
        # Get last valid time per batch for batch x time step x feature tensor

        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])

        # Make sure that we get the final index, which is 1 less than sequence length
        index = tf.range(0, batch_size) * max_length + tf.subtract(sequence_length, 1)  # length should be batchsize as well
        flat = tf.reshape(output, [-1, out_size])  # flattens the index into batch * batchsize + length
        relevant = tf.gather(flat, index)

    elif len(shape) == 2:

        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]

        # Make sure that we get the final index, which is 1 less than sequence length
        index = tf.range(0, batch_size) * max_length + tf.subtract(sequence_length,
                                                                   1)  # length should be batchsize as well
        flat = tf.reshape(output, [-1])  # flattens the index into batch * batchsize + length
        relevant = tf.gather(flat, index)

    else:
        raise ValueError("Illegal shape type {0}".format(shape))
    return relevant


def randomise_minibatch_index(Y, minibatch_size):
    batch_num, target_num = Y.shape

    # Randomise rows
    rdm_idx = [i for i in range(batch_num)]
    np.random.shuffle(rdm_idx)

    max_idx = len(rdm_idx)
    num_minibatches = int(max_idx / minibatch_size)
    minibatch_numbers = [j for j in range(num_minibatches)]

    tmp = []
    for count in range(len(minibatch_numbers)):
        j = minibatch_numbers[count]

        ptr = j * minibatch_size
        end_idx = min(minibatch_size + ptr, max_idx)
        minibatch_idx = rdm_idx[ptr:end_idx]

        tmp.append(minibatch_idx)
    return tmp

def get_optimization_graph(loss, learning_rate, max_global_norm, global_step):
    # Optimisation step
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Clip gradients to prevent them from blowing up
    trainables = tf.trainable_variables()
    grads = tf.gradients(loss, trainables)
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
    grad_var_pairs = list(zip(grads, trainables))
    global_step = global_step

    minimize = optimizer.apply_gradients(grad_var_pairs,
                                         global_step=global_step)
    
    return minimize

""" 
Serialisation
"""
def create_folder_if_not_exist(directory):
    #if not os.path.exists(directory):
        #os.makedirs(directory)

    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)  # Also creates directories recursively


def save(tf_session, model_folder, cp_name, scope=None):

    # Save model
    if scope is None:
        saver = tf.train.Saver()
    else:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)

    save_path = saver.save(tf_session, os.path.join(model_folder, "{0}.ckpt".format(cp_name)))
    logging.info("Model saved to: {0}".format(save_path))


def load(tf_session, model_folder, cp_name, scope=None):

    # Load model proper
    load_path = os.path.join(model_folder, "{0}.ckpt".format(cp_name))

    logging.info('Loading model from {0}'.format(load_path))

    print_weights_in_checkpoint(model_folder, cp_name)

    initial_vars = set([v.name for v in tf.get_default_graph().as_graph_def().node])

    # Saver
    if scope is None:
        saver = tf.train.Saver()
    else:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)
    # Load
    saver.restore(tf_session, load_path)
    all_vars = set([v.name for v in tf.get_default_graph().as_graph_def().node])

    logging.info('Restored {0}'.format(','.join(initial_vars.difference(all_vars))))
    logging.info('Existing {0}'.format(','.join(all_vars.difference(initial_vars))))
    logging.info('All {0}'.format(','.join(all_vars)))

    logging.info("Done.")

def print_weights_in_checkpoint(model_folder, cp_name):

    load_path = os.path.join(model_folder, "{0}.ckpt".format(cp_name))

    print_tensors_in_checkpoint_file(file_name=load_path, tensor_name='',
                                     all_tensors=True, all_tensor_names=True)