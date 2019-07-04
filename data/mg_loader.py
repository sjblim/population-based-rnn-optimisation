"""
mg_loader.py


Created by limsi on 02/04/2019
"""


from configs import mg_configs
from data.base_data_loaders import DataTypes, BaseCsvLoader, BaseCombinationLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _sort_unique(l):

    z = list(set(l))
    z.sort()

    return z


class MackeyGlassDataLoader(BaseCsvLoader):

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Basic definitions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    INPUT_COLS = ['y_cur']  # output last

    OUTPUT_COLS = ['y_next']

    RNF_COLS = []  # Artefact from previous project

    COLS_OF_INTEREST = _sort_unique(INPUT_COLS + OUTPUT_COLS + RNF_COLS)
    use_standard_scaler = True
    output_first = False

    def __init__(self):
        self._name = 'mackey'
        trajectory = mg_configs.get_core_simulation()

        self.raw_data = pd.DataFrame({'y_cur': trajectory, 'y_next': trajectory.shift(-1)}).dropna()

    def load_data_from_file(self, data_types=[DataTypes.TRAIN,
                                              DataTypes.VALID,
                                              DataTypes.TEST]):

        data = self.raw_data.copy()

        T = len(data)

        train_valid_boundary = int(T*0.6)
        valid_test_boundary = int(T*0.8)

        data_map = {DataTypes.TRAIN: data.iloc[:train_valid_boundary, :],
                    DataTypes.VALID: data.iloc[train_valid_boundary:valid_test_boundary, :],
                    DataTypes.TEST: data.iloc[valid_test_boundary:, :]}

        return data_map


class ShortMackeyGlassDataLoader(MackeyGlassDataLoader):

    INPUT_COLS = ['y_cur']  # output last

    OUTPUT_COLS = ['y_next']

    RNF_COLS = []  # Artefact from previous project

    COLS_OF_INTEREST = _sort_unique(INPUT_COLS + OUTPUT_COLS + RNF_COLS)
    use_standard_scaler = True
    output_first = False

    def __init__(self):
        self._name = 'short_mackey'
        trajectory = mg_configs.get_short_simulation()

        self.raw_data = pd.DataFrame({'y_cur': trajectory, 'y_next': trajectory.shift(-1)}).dropna()


class SimpleMackeyGlassDataLoader(MackeyGlassDataLoader):

    INPUT_COLS = ['y_cur']  # output last

    OUTPUT_COLS = ['y_next']

    RNF_COLS = []  # Artefact from previous project

    COLS_OF_INTEREST = _sort_unique(INPUT_COLS + OUTPUT_COLS + RNF_COLS)
    use_standard_scaler = True
    output_first = False

    def __init__(self):
        self._name = 'simple_mackey'
        trajectory = mg_configs.get_simple_simulation()

        self.raw_data = pd.DataFrame({'y_cur': trajectory, 'y_next': trajectory.shift(-1)}).dropna()


class ShortSimpleMackeyGlassDataLoader(MackeyGlassDataLoader):

    INPUT_COLS = ['y_cur']  # output last

    OUTPUT_COLS = ['y_next']

    RNF_COLS = []  # Artefact from previous project

    COLS_OF_INTEREST = _sort_unique(INPUT_COLS + OUTPUT_COLS + RNF_COLS)
    use_standard_scaler = True
    output_first = False

    def __init__(self):
        self._name = 'short_simple'
        trajectory = mg_configs.get_short_simple()

        self.raw_data = pd.DataFrame({'y_cur': trajectory, 'y_next': trajectory.shift(-1)}).dropna()


class NoisyMackeyGlassDataLoader(MackeyGlassDataLoader):

    INPUT_COLS = ['y_cur']  # output last

    OUTPUT_COLS = ['y_next']

    RNF_COLS = []  # Artefact from previous project

    COLS_OF_INTEREST = _sort_unique(INPUT_COLS + OUTPUT_COLS + RNF_COLS)
    use_standard_scaler = True
    output_first = False

    def __init__(self):
        self._name = 'noisy'
        trajectory = mg_configs.get_noisy_simulation()

        self.raw_data = pd.DataFrame({'y_cur': trajectory, 'y_next': trajectory.shift(-1)}).dropna()


