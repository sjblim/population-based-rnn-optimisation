"""
core.py


Created by limsi on 10/04/2019
"""

import datetime as dte
import tensorflow as tf
import numpy as np

from libs.hyperparam_opt import HyperparamOptManager
from data.training_formatter import TrainingDataFormatter
from libs.neuroevolution.particle import NeuroevolutionParticle

from joblib import Parallel, delayed


def train_neuroevolution(train_data,
                         valid_data,
                         NetworkClass,
                         StateInitClass,
                         EvolutionStrategy,
                         hyperparameter_manager,
                         original_info,
                         tf_config):

    # Main routine
    print("Running Deep Neuroevolution...")
    tf.reset_default_graph()

    # Checks
    if not isinstance(hyperparameter_manager, HyperparamOptManager):
        raise ValueError("hyperparameter_manager is a {}".format(type(hyperparameter_manager).__name__))

    params = hyperparameter_manager.get_next_parameters()

    # Unpack model
    num_models = params['population_size']
    epochs = params['num_epochs']
    early_stopping = params['early_stopping_patience']

    # Variables to track progress
    best_valid = np.Inf  # implemented for early stopping
    iterations_without_update = 0
    # Do an initial log - in case something errors out in between
    hyperparameter_manager.update_score(params, best_valid, None, info="Initial logging")

    # Setup datasets
    batched_train_data = TrainingDataFormatter.create_batched_training_data(train_data, 1)  # single trajectory
    batched_valid_data = TrainingDataFormatter.create_batched_training_data(valid_data, 1)

    # To get weights and name
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
        valid_model = NetworkClass(params, sess, StateInitClass)
        valid_handles = valid_model.get_neuroevolution_graph(raw_dataset=train_data, use_evaluation=True)
        num_weights = valid_handles['weight_placeholder'].get_shape().as_list()[0]
        net_name = type(valid_model).__name__

    #######################################################################################

    # Setup ES population details
    w_0 = np.zeros(num_weights)
    evolution_strategy = EvolutionStrategy(params, w_0)
    try:
        for epoch in range(epochs):

            particle_weights = evolution_strategy.mutate()
            additional_info = original_info + " | Epoch {} of {}".format(epoch, epochs)

            # Build graph for all members of population -- potentially computationally expensive!!
            def build_graph_and_compute_rewards(i):

                with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

                    model = NetworkClass(params, sess, StateInitClass,
                                         variable_scope_suffix="_{}".format(i))

                    handles = model.get_neuroevolution_graph(raw_dataset=train_data, use_evaluation=True)

                    particle = NeuroevolutionParticle(model, handles, batched_train_data)

                    particle.update_weights(particle_weights[i])
                    reward = -particle.compute_loss(verbose=True, additional_info=additional_info)
                return reward

            tf.reset_default_graph()
            rewards = Parallel(n_jobs=-2)(delayed(build_graph_and_compute_rewards)(i) for i in range(num_models))
            rewards = np.array(rewards)

            # Compute weight update
            evolution_strategy.update(rewards)
            w = evolution_strategy.population_weight

            # VALIDATION
            tf.reset_default_graph()
            with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

                valid_model = NetworkClass(params, sess, StateInitClass)
                valid_handles = valid_model.get_neuroevolution_graph(raw_dataset=valid_data, use_evaluation=True)
                valid_particle = NeuroevolutionParticle(valid_model, valid_handles, batched_valid_data)

                sess.run(tf.global_variables_initializer())  # initialse anything unused so we can save it later

                valid_particle.update_weights(w)
                valid_loss = valid_particle.compute_loss(batched_valid_data)

                print("{} | Epoch {} of {} | * validation loss = {} * | net = {} | info = {}".format(
                    dte.datetime.now().strftime("%Y%m%d %H:%M:%S"),
                    epoch,
                    epochs,
                    valid_loss,
                    net_name,
                    original_info))

                # Check nans, terminate early if so
                if np.isnan(valid_loss) or valid_loss == 0.0:
                    print("Nan validation loss - terminating training")
                    raise ValueError("Nan loss found!")

                # EARLY STOPPING & SERIALISATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if valid_loss < best_valid:
                    best_valid = valid_loss

                    valid_model.save(hyperparameter_manager.hyperparam_folder, is_best_model=False)

                    # Update score for this set of params, which automatically fixes the optimal score
                    hyperparameter_manager.update_score(params, valid_loss, valid_model, info="epoch:{}".format(epoch))

                    iterations_without_update = 0

                else:
                    iterations_without_update += 1

                    # Check early stopping criteria, per Keras
                    thresh = early_stopping
                    if iterations_without_update > thresh:
                        print("Early stopping criterion hit at epoch {}! thresh={}, count={}".format(epoch,
                                                                                                     thresh,
                                                                                                     iterations_without_update))
                        raise ValueError("Terminating due to early stopping rules!")

    except Exception as e:
        print("ERROR FOUND, terminating routine! Error msg: {}".format(e))
        raise e

    print("Completed @", dte.datetime.now().strftime("%Y%m%d %H:%M:%S"), "for", epoch,
          "epochs - saving validation scores")













