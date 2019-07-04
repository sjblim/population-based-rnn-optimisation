"""
neuroevolution_script.py


Created by limsi on 08/04/2019
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
from data.training_formatter import TrainingDataFormatter

# Models
import libs.common as core
from libs.networks.lstm import BaseLSTMModel
from libs.networks.phased_lstm import PhasedLSTM
from libs.networks.fru import FRUModel
from libs.state.basic_init import ZeroInitialiser
from libs.neuroevolution.core import train_neuroevolution
from libs.neuroevolution.strategies import BasicEvolutionStrategy, ParticleSwarmOptimisationStrategy


# In[*]: Preliminaries

# Constants
default_tf_device = "cpu"  # No backprop calcs required

# Maps
valid_models = {'lstm': BaseLSTMModel, 'phased_lstm': PhasedLSTM, 'fru': FRUModel}
valid_evolution_strategies = {'basic': BasicEvolutionStrategy,
                              'chope': BasicEvolutionStrategy,
                              'pso': ParticleSwarmOptimisationStrategy}

replacements = {'basic': (100, 300, 100, 48),
                # 'chope': (1, 1, 100, 2),
                # 'pso': (1, 5, 100, 2),
                'default': (20, 50, 50, 30)}  # (random search iter, epochs, early stopping, pop size)


def get_args():

    parser = argparse.ArgumentParser(description='Define ML method')
    parser.add_argument('method', metavar='M', type=str, nargs='?',  # 0 or 1 argument
                        choices=list(valid_models.keys()),
                        default="fru",
                        help='Define ML method')
    parser.add_argument('data_type', metavar='t', type=str, nargs='?',  # t0 or 1 argument
                        choices=DataLoaderFactory.get_valid_loaders(),
                        default="short_simple",
                        help='Dataset to load')
    parser.add_argument('evolution_strategy', metavar='s', type=str, nargs='?',  # t0 or 1 argument
                        choices=valid_evolution_strategies.keys(),
                        default="pso",
                        help='How populations evolve')
    parser.add_argument('new_hyperparam_opt', metavar='n', type=str, nargs='?',  # 0 or 1 argument
                        choices=["yes", "no"],
                        default="yes",
                        help='Define whether to start a new hyperparameter opt')

    args = parser.parse_args()

    return args.method, args.data_type, args.evolution_strategy, args.new_hyperparam_opt == "yes"


# In[*]: Routine
if __name__ == "__main__":

    # Get experiment params
    model_type, dataset_name, evolution_strategy_name, restart_opt = get_args()

    # Setup
    tf_config = helpers.get_default_tensorflow_config(tf_device=default_tf_device)
    expt_name = "{}_{}_neuroevolution_{}".format(dataset_name, model_type, evolution_strategy_name)

    data_loader = DataLoaderFactory.make_data_loader(dataset_name)
    configs = DataLoaderFactory.get_default_config(dataset_name)
    ModelClass = valid_models[model_type]
    StateInitClass = ZeroInitialiser
    EvolutionStrategyClass = valid_evolution_strategies[evolution_strategy_name]
    unroll_length = 100000  # only need to do forward propagation

    replacement_name = 'default' if evolution_strategy_name not in replacements else evolution_strategy_name

    random_search_iterations = replacements[replacement_name][0]

    # Parameters
    fixed_params = {'rnn_window': unroll_length,
                    'input_size': data_loader.get_input_size(),
                    'output_size': data_loader.get_output_size(),
                    'model_folder': configs.MODEL_FOLDER,  # used for intermediate training
                    'loss_type': LossTypes.MSE,
                    'hidden_activation': 'elu',
                    'num_epochs': replacements[replacement_name][1],
                    'early_stopping_patience': replacements[replacement_name][2],
                    'population_size': replacements[replacement_name][3]
                    }

    if model_type in {'phased_lstm'}:
        fixed_params['time_increment'] = 1/configs.MAX_TRAJECTORY_LENGTH

    search_params = ModelClass.get_hyperparm_choices()
    search_params['sigma'] = [1e-3, 1e-2, 1e-1, 1.0]
    search_params['w_sigma'] = search_params['dropout_rate']

    # Hyperparam manager
    opt_manager = HyperparamOptManager(search_params,
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

        train_neuroevolution(train_data,
                              valid_data,
                              ModelClass,
                              StateInitClass,
                              EvolutionStrategyClass,
                              opt_manager,
                              additional_info,
                              tf_config)

        opt_iteration = len(opt_manager.results.columns)

    print("*** Running evaluation ***")
    data_map = data_loader.make_rnn_tf_dataset(unroll_length)
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
    loss_file = os.path.join(configs.RESULTS_FOLDER, "loss_{}_{}.csv".format(dataset_name, expt_name))
    pd.Series({'expt_name': expt_name,
               'loss': test_loss,
               'time': dte.datetime.now().strftime("%Y%m%d %H:%M:%S")}).to_csv(loss_file, header=True)





