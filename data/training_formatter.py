"""
training_formatter.py


Created by limsi on 04/04/2019
"""

import numpy as np


class TrainingDataFormatter:


    @staticmethod
    def batch_arr(arr, minibatch_size):

        num_trajs = arr.shape[0]

        num_batches = int(np.ceil(num_trajs / minibatch_size))

        batched_data = []
        for i in range(num_batches):
            batched_data.append(arr[i:i + minibatch_size])

        return batched_data

    @classmethod
    def create_batched_training_data(cls, dataset, minibatch_size, randomise=False):

        first_key = list(dataset.keys())[0]
        num_trajectories = dataset[first_key].shape[0]
        index = [i for i in range(num_trajectories)]

        if randomise:  # randomise trajectory ordering
            index = np.random.permutation(index)

        # Batch data appropriately
        raw_data = {k: cls.batch_arr(dataset[k][index], minibatch_size) for k in dataset}

        # Convert form dictionary to list
        num_batches = len(raw_data[first_key])
        batch_map_list = []

        for i in range(num_batches):
            batch_map = {k: raw_data[k][i] for k in raw_data}

            batch_map_list.append(batch_map)

        return batch_map_list

    @staticmethod
    def normalise_trajectory_index(dataset, T):

        # create deep copy
        new_dataset = {k: dataset[k].copy() for k in dataset}

        # normalise
        new_dataset['trajectory_index'] = new_dataset['trajectory_index'] / T
        return new_dataset







