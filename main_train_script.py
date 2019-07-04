"""
main_train_script.py


Created by limsi on 04/04/2019
"""

# General
import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime as dte

# Misc
from libs import net_helpers as helpers
from libs.losses import LossTypes, LossFunctionHelper
from libs.hyperparam_opt import HyperparamOptManager

# Data
from data.data_loader_factory import DataLoaderFactory
from data.base_data_loaders import DataTypes
from data.mg_loader import MackeyGlassDataLoader
from data.training_formatter import TrainingDataFormatter

# State intialisers
from libs.state.basic_init import ZeroInitialiser, TrainedInitialiser, UnitGaussianInitialiser, \
    TrainedGaussianInitialiser
from libs.state.regressors import MLPInitialiser, StateInputMLPInitialiser
from libs.state.ensembles import EnsembleMLPInitialiser, EnsembleStateInputMLPInitialiser

# Models
import libs.common as core
from libs.networks.lstm import BaseLSTMModel
from libs.networks.phased_lstm import PhasedLSTM
from libs.networks.fru import FRUModel


# In[*]: Preliminaries

# Constants
default_tf_device = "gpu"

# Maps
valid_models = {'lstm': BaseLSTMModel, 'phased_lstm': PhasedLSTM, 'fru': FRUModel}

valid_state_inits = {'zero': ZeroInitialiser,
                     'trained': TrainedInitialiser,
                     'normal': UnitGaussianInitialiser,
                     'train_norm': TrainedGaussianInitialiser,
                     'mlp_reg': MLPInitialiser,
                     'ensemble': EnsembleMLPInitialiser,
                     'state_input': StateInputMLPInitialiser,
                     'si_ensemble': EnsembleStateInputMLPInitialiser}

valid_unroll_lengths = {'max': 100,
                        'min': 20,
                        'upper': 50,
                        'lower': 40}


def get_args():

    parser = argparse.ArgumentParser(description='Define ML method')
    parser.add_argument('method', metavar='M', type=str, nargs='?',  # 0 or 1 argument
                        choices=list(valid_models.keys()),
                        default="lstm",
                        help='Define ML method')
    parser.add_argument('data_type', metavar='t', type=str, nargs='?',  # t0 or 1 argument
                        choices=DataLoaderFactory.get_valid_loaders(),
                        default="noisy",
                        help='Dataset to load')
    parser.add_argument('state_init', metavar='s', type=str, nargs='?',  # 0 or 1 argument
                        choices=list(valid_state_inits.keys()),
                        default="zero",
                        help='Define state initialisation methods')
    parser.add_argument('unroll', metavar='s', type=str, nargs='?',  # 0 or 1 argument
                        choices=list(valid_unroll_lengths.keys()),
                        default="min",
                        help='How far to rollback the RNN graph')
    parser.add_argument('new_hyperparam_opt', metavar='n', type=str, nargs='?',  # 0 or 1 argument
                        choices=["yes", "no"],
                        default="yes",
                        help='Define whether to start a new hyperparameter opt')

    args = parser.parse_args()

    return args.method, args.data_type, args.state_init, args.unroll, args.new_hyperparam_opt == "yes"


# In[*]: Routine
if __name__ == "__main__":

    # Get experiment params
    model_type, dataset_name, state_init_type, unroll_to, restart_opt = get_args()

    # Setup
    tf_config = helpers.get_default_tensorflow_config(tf_device=default_tf_device)
    expt_name = "{}_{}_{}_{}".format(dataset_name, model_type, state_init_type, unroll_to)
    data_loader = DataLoaderFactory.make_data_loader(dataset_name)
    configs = DataLoaderFactory.get_default_config(dataset_name)
    ModelClass = valid_models[model_type]
    StateInitClass = valid_state_inits[state_init_type]
    unroll_length = valid_unroll_lengths[unroll_to]

    random_search_iterations = configs.RANDOM_SEARCH_ITERATIONS

    fixed_params = {'rnn_window': unroll_length,
                    'input_size': data_loader.get_input_size(),
                    'output_size': data_loader.get_output_size(),
                    'model_folder': configs.MODEL_FOLDER,  # used for intermediate training
                    'loss_type': LossTypes.MSE,
                    'hidden_activation': 'elu',
                    'num_epochs': 300,
                    'early_stopping_patience': 25
                    }

    if StateInitClass.is_ensemble_model:
        fixed_params['num_models'] = 20

    if model_type in {'phased_lstm'}:
        fixed_params['time_increment'] = 1/configs.MAX_TRAJECTORY_LENGTH

    # Hyperparam manager
    opt_manager = HyperparamOptManager(ModelClass.get_hyperparm_choices(),
                                       fixed_params,
                                       expt_name,
                                       configs.MODEL_FOLDER)

    success = opt_manager.load_results()
    if success and not restart_opt:
        print("Loaded results from previous training")
    else:
        print("Creating new hyperparameter optimisation")
        opt_manager.clear()

    print("*** Making data map ***")
    data_map = data_loader.make_rnn_tf_dataset(unroll_length)
    train_data = data_map[DataTypes.TRAIN]
    valid_data = data_map[DataTypes.VALID]

    print("*** Running calibration ***")
    opt_iteration = len(opt_manager.results.columns)
    while opt_iteration < random_search_iterations:

        print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("# Running hyperparam optimisation {} of {} for {}".format(len(opt_manager.results.columns) + 1,
                                                                         random_search_iterations,
                                                                         expt_name))
        print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        print("Commencing model setup...")

        additional_info = "hyperparam opt = {} of {}".format(opt_iteration, random_search_iterations)

        core.train(train_data, valid_data, ModelClass, StateInitClass, opt_manager, additional_info, tf_config)

        opt_iteration = len(opt_manager.results.columns)

    print("*** Running evaluation ***")
    data_map = data_loader.make_rnn_tf_dataset(int(1e4))
    test_data = data_map[DataTypes.TEST]
    batched_test_data = TrainingDataFormatter.create_batched_training_data(test_data, 1, False)

    test_loss = core.evaluate_single_trajectory(batched_test_data,
                                                ModelClass,
                                                opt_manager,
                                                tf_config)

    print("")
    print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Loss = {} for: {}".format(test_loss, expt_name))
    print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Save results
    loss_file = os.path.join(configs.RESULTS_FOLDER, "loss_{}.csv".format(expt_name))
    pd.Series({'expt_name': expt_name,
               'loss': test_loss,
               'time': dte.datetime.now().strftime("%Y%m%d %H:%M:%S")}).to_csv(loss_file, header=True)
