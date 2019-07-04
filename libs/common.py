"""
common_configs.py


Created by limsi on 04/10/2018
"""
import datetime as dte
import tensorflow as tf
import numpy as np

from libs.hyperparam_opt import HyperparamOptManager
from data.training_formatter import TrainingDataFormatter
from libs.state.basic_init import ZeroInitialiser


def train(train_data,
          valid_data,
          NetworkClass,
          StateInitClass,
          hyperparameter_manager,
          original_info,
          tf_config):

    # Format trajectory index
    T = valid_data['trajectory_index'].max()
    dt = 1/T

    print("Normalising trajectory index")
    train_data = TrainingDataFormatter.normalise_trajectory_index(train_data, T)
    valid_data = TrainingDataFormatter.normalise_trajectory_index(valid_data, T)


    # Main routine
    print("Starting training...")
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

        # Checks
        if not isinstance(hyperparameter_manager, HyperparamOptManager):
            raise ValueError("hyperparameter_manager is a {}".format(type(hyperparameter_manager).__name__))

        params = hyperparameter_manager.get_next_parameters()

        # Unpack model
        model = NetworkClass(params, sess, StateInitClass)
        state_initialiser = model.state_initialiser
        net_name = type(model).__name__
        epochs = model.epochs
        minibatch_size = model.minibatch_size
        full_state_size = model.get_initial_state_dims()

        # Variables to track progress
        best_valid = np.Inf  # implemented for early stopping
        iterations_without_update = 0
        # Do an initial log - in case something errors out in between
        hyperparameter_manager.update_score(params, best_valid, None, info="Initial logging")

        # Setup datasets
        batched_valid_data = TrainingDataFormatter.create_batched_training_data(valid_data, 1000000)

        # Handles
        train_handles = model.get_training_graph(raw_dataset=train_data)
        valid_handles = model.get_validation_graph(raw_dataset=valid_data)
        train_placeholders = train_handles['data_placeholders']
        valid_placeholders = valid_handles['data_placeholders']

        # Start
        sess.run(tf.global_variables_initializer())

        try:
            for epoch in range(epochs):

                # Shuffle trajectory ordering per epoch
                batched_training_data = TrainingDataFormatter.create_batched_training_data(train_data,
                                                                                           minibatch_size,
                                                                                           randomise=True)

                # TRAINING STEP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                num_minibatches = len(batched_training_data)
                for i, data_map in enumerate(batched_training_data):

                    # Setup placeholders
                    placeholder_mapping = {train_placeholders[k]: data_map[k]
                                           for k in train_placeholders}

                    # Add initial states
                    if train_handles['initial_states'] is not None:
                        placeholder_mapping[train_handles['initial_states']] \
                            = state_initialiser.get_states(state_indices=data_map['trajectory_index'])

                    # Actual loss function
                    loss, _, final_states \
                        = sess.run([train_handles['loss'],
                                    train_handles['optimiser'],
                                    train_handles['final_states']],
                                    placeholder_mapping)

                    # Update state regressor
                    if state_initialiser.is_state_regressor:
                        state_regression_loss = state_initialiser.update_regressor(data_map['trajectory_index'] + dt,
                                                                                   final_states)
                        additional_info = original_info + " | s_loss={}".format(state_regression_loss)
                    else:
                        additional_info = original_info

                    print("{} | Epoch {} of {} | iteration = {} of {}, loss = {} | net = {} | info = {}".format(
                        dte.datetime.now().strftime("%Y%m%d %H:%M:%S"),
                        epoch,
                        epochs,
                        i+1,
                        num_minibatches,
                        loss,
                        net_name,
                        additional_info))

                # VALIDATION STEP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                total_loss = 0.0
                total_counts = 0
                for i, data_map in enumerate(batched_valid_data):

                    # Setup placeholders
                    placeholder_mapping = {valid_placeholders[k]: data_map[k]
                                           for k in valid_placeholders}

                    # Add initial states
                    if valid_handles['initial_states'] is not None:
                        placeholder_mapping[valid_handles['initial_states']] \
                            = state_initialiser.get_states(state_indices=data_map['trajectory_index'])

                    # Actual loss function
                    loss, counts, final_states\
                        = sess.run([valid_handles['total_loss'],
                                    valid_handles['counts'],
                                    valid_handles['final_states']],
                                    placeholder_mapping)

                    total_loss += loss
                    total_counts += counts

                    if state_initialiser.is_state_regressor:
                        state_regression_loss = state_initialiser.update_regressor(data_map['trajectory_index'] + dt,
                                                           final_states)
                    else:
                        state_regression_loss = 0.0

                valid_loss = total_loss / total_counts + state_regression_loss

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
                    model.save(hyperparameter_manager.hyperparam_folder,
                               is_best_model=False)
                    # Update score for this set of params, which automatically fixes the optimal score
                    hyperparameter_manager.update_score(params, valid_loss, model, info="epoch:{}".format(epoch))

                    iterations_without_update = 0

                else:
                    iterations_without_update += 1

                    # Check early stopping criteria, per Keras
                    thresh = model.early_stopping_patience
                    if iterations_without_update > thresh:
                        print("Early stopping criterion hit at epoch {}! thresh={}, count={}".format(epoch,
                                                                                                     thresh,
                                                                                                     iterations_without_update))
                        raise ValueError("Terminating due to early stopping rules!")

        except Exception as e:
            print("ERROR FOUND, terminating routine! Error msg: {}".format(e))
            # raise e

        print("Completed @", dte.datetime.now().strftime("%Y%m%d %H:%M:%S"), "for", epoch,
              "epochs - saving validation scores")

    return best_valid


def evaluate_single_trajectory(batched_data,
                               NetworkClass,
                               hyperparameter_manager,
                               tf_config):

    tf.reset_default_graph()

    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

        # Checks
        if not isinstance(hyperparameter_manager, HyperparamOptManager):
            raise ValueError("hyperparameter_manager is a {}".format(type(hyperparameter_manager).__name__))

        if hyperparameter_manager.load_results():

            # Get optimal params
            params = hyperparameter_manager.get_best_params()

            # Load model
            model = NetworkClass(params, sess, ZeroInitialiser)  # always start off witha zero initialiser
            state_initialiser = model.state_initialiser

            # Init all variables
            sess.run(tf.global_variables_initializer())

            # Load model
            model.load(hyperparameter_manager.hyperparam_folder, use_best_model=True)

            # Handles
            handles = model.get_evaluation_graph(batched_data[0])
            placeholders = handles['data_placeholders']

            states = None
            total_loss = 0.0
            total_counts = 0
            for i, data_map in enumerate(batched_data):

                # Setup placeholders
                placeholder_mapping = {placeholders[k]: data_map[k]
                                       for k in placeholders}

                # Add initial states
                if states is None:
                    placeholder_mapping[handles['initial_states']] = state_initialiser.get_states(
                        state_indices=data_map['trajectory_index'])
                else:
                    placeholder_mapping[handles['initial_states']] = states

                # Actual loss function
                loss, counts, states \
                    = sess.run([handles['total_loss'],
                                handles['counts'],
                                handles['final_states']],
                               placeholder_mapping)

                total_loss += loss
                total_counts += counts

                print("Evaluation {} of {}: Loss={}".format(i, len(batched_data), total_loss/total_counts))

            valid_loss = total_loss / total_counts

            return valid_loss

        else:
            raise ValueError("Calibration has not yet been run!!!")


def predict_single_trajectory(batched_data,
                               NetworkClass,
                               StateInitClass,
                               hyperparameter_manager,
                               tf_config):

    tf.reset_default_graph()

    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

        # Checks
        if not isinstance(hyperparameter_manager, HyperparamOptManager):
            raise ValueError("hyperparameter_manager is a {}".format(type(hyperparameter_manager).__name__))

        if hyperparameter_manager.load_results():

            # Get optimal params
            params = hyperparameter_manager.get_best_params()

            # Load model
            model = NetworkClass(params, sess, StateInitClass)  # always start off witha zero initialiser
            state_initialiser = model.state_initialiser

            # Init all variables
            sess.run(tf.global_variables_initializer())

            # Load model
            model.load(hyperparameter_manager.hyperparam_folder, use_best_model=True)

            # Handles
            handles = model.get_prediction_graph(batched_data[0])
            placeholders = handles['data_placeholders']

            states = None
            prediction_holder = []
            for i, data_map in enumerate(batched_data):

                # Setup placeholders
                placeholder_mapping = {placeholders[k]: data_map[k]
                                       for k in placeholders}

                # Add initial states
                if states is None:
                    placeholder_mapping[handles['initial_states']] = state_initialiser.get_states(
                        state_indices=data_map['trajectory_index'])
                else:
                    placeholder_mapping[handles['initial_states']] = states

                # Actual loss function
                local_predictions, states \
                    = sess.run([handles['predictions'],
                                handles['final_states']],
                               placeholder_mapping)

                prediction_holder.append(local_predictions[0, :, :])

                print("Predicting {} of {}".format(i, len(batched_data)))

            predictions = np.concatenate(prediction_holder, axis=0)

            return predictions

        else:
            raise ValueError("Calibration has not yet been run!!!")