"""
base_data_loaders.py


Created by limsi on 24/10/2018
"""

from abc import ABC, abstractmethod
from enum import IntEnum
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from joblib import Parallel, delayed

# -----------------------------------------------------------------------------
class DataTypes(IntEnum):

    TRAIN = 1
    VALID = 2
    TEST = 3
    SCALER = 4

    @staticmethod
    def get_string_name():

        return {DataTypes.TRAIN: "train",
                DataTypes.VALID: "valid",
                DataTypes.TEST: "test",
                DataTypes.SCALER: "scaler"}


# -----------------------------------------------------------------------------
class BaseDataLoader(ABC):

    @property
    def name(self):
        return self._name

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Required methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @abstractmethod
    def get_input_size(self):
        pass

    @abstractmethod
    def get_output_size(self):
        pass

    @abstractmethod
    def make_rnn_tf_dataset(self, state_window):
        pass

    @abstractmethod
    def make_rnf_dataset(self, state_window):
        pass

    @abstractmethod
    def make_linear_dataset(self, lags):
        pass

    def make_single_run_dataset(self):
        pass



# -----------------------------------------------------------------------------
class BaseCsvLoader(BaseDataLoader):

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Required attributes
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @property
    def RNF_COLS(self):
        pass

    @property
    def INPUT_COLS(self):
        pass

    @property
    def OUTPUT_COLS(self):
        pass

    @property
    def COLS_OF_INTEREST(self):
        pass

    @property
    def use_standard_scaler(self):
        return True

    @property
    def output_first(self):  # remember that recursion is extremely important for state update
        return False


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Required methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @abstractmethod
    def load_data_from_file(self, data_types=[DataTypes.TRAIN,
                                              DataTypes.VALID,
                                              DataTypes.TEST]):
        raise NotImplementedError()

    def get_input_size(self):
        return len(self.INPUT_COLS)

    def get_output_size(self):
        return len(self.OUTPUT_COLS)

    def _get_ml_arrays(self,
                       all_data_types,
                       input_cols):

        data_map = self.load_data_from_file(all_data_types)
        use_standard_scaler = self.use_standard_scaler

        input_scaler = None
        output_scaler = None
        array_map = {}

        for k in all_data_types:

            df = data_map[k]

            # Unpack inputs and outputs
            inputs = df[input_cols].values
            outputs = df[self.OUTPUT_COLS].values

            # Scale inputs - fit scaler if necessary
            if input_scaler is None:
                if k == DataTypes.TRAIN:
                    if use_standard_scaler:
                        print("Normalising using Standard Scaler")
                        input_scaler = StandardScaler().fit(inputs)
                        output_scaler = StandardScaler().fit(outputs)
                        print(self.name, output_scaler.var_)
                    else:
                        print("Skipping Normalisation - use Min Max Scaler with unit range")
                        zero_inputs = inputs * 0.0
                        zero_inputs[-1, :] = 1.0
                        input_scaler = MinMaxScaler().fit(zero_inputs)
                        zero_outputs = outputs * 0.0
                        zero_outputs[-1, :] = 1.0
                        output_scaler = MinMaxScaler().fit(zero_outputs)
                        print(self.name, output_scaler.data_min_, output_scaler.data_max_)

                    array_map[DataTypes.SCALER] = (input_scaler, output_scaler)

                else:
                    raise ValueError("Illegal ordering of data types!!")

            inputs = input_scaler.transform(inputs)
            outputs = output_scaler.transform(outputs)
            array_map[k] = (inputs, outputs)

        return array_map

    def _make_rnn_tf_dataset(self, state_window, input_cols):
        print("Making RNN dataset")
        all_data_types = [DataTypes.TRAIN, DataTypes.VALID, DataTypes.TEST]

        array_maps = self._get_ml_arrays(all_data_types, input_cols=input_cols)
        tensorflow_map = {DataTypes.SCALER: array_maps[DataTypes.SCALER]}  # copy scaler for use later

        # Format data to use for RNN
        input_size = len(input_cols)  # self.get_input_size()
        output_size = self.get_output_size()
        for k in all_data_types:

            inputs, outputs = array_maps[k]

            # Zero-pad data as required - record number of zero paddings for sequence length
            total_time_steps = inputs.shape[0]
            additional_time_steps_required = state_window - (total_time_steps % state_window)

            if additional_time_steps_required > 0:
                inputs = np.concatenate([inputs, np.zeros((additional_time_steps_required, input_size))])
                outputs = np.concatenate([outputs, np.zeros((additional_time_steps_required, output_size))])

            # Reshape inputs now
            inputs = inputs.reshape(-1, state_window, input_size)
            outputs = outputs.reshape(-1, state_window, output_size)

            batch_size = inputs.shape[0]
            sequence_lengths = [(state_window if i != batch_size - 1 else state_window - additional_time_steps_required)
                                for i in range(batch_size)]

            # Setup active entries
            active_entries = np.ones((outputs.shape[0], outputs.shape[1], outputs.shape[2]))
            for i in range(outputs.shape[0]):
                active_entries[i, sequence_lengths[i]:, :] = 0

            # Remove any zero sequence lengths
            sequence_lengths = np.array(sequence_lengths, dtype=np.int)
            trajectory_index = np.array([[i for i in range(inputs.shape[0])]], dtype=np.int).T

            # Filter out empty entries
            good_elems = sequence_lengths != 0
            inputs = inputs[good_elems]
            outputs = outputs[good_elems]
            active_entries = active_entries[good_elems]
            trajectory_index = trajectory_index[good_elems]
            sequence_lengths = sequence_lengths[good_elems]

            # Package in to tensorflow dataset
            dataset = {'inputs': inputs,
                       'outputs': outputs,
                       'sequence_lengths': sequence_lengths,
                       'active_entries': active_entries,
                       'trajectory_index': trajectory_index}

            tensorflow_map[k] = dataset

        # Unscale trajectory indices
        num_trajectories = {k: tensorflow_map[k]['trajectory_index'].max()+1 for k in all_data_types}
        tensorflow_map[DataTypes.VALID]['trajectory_index'] += num_trajectories[DataTypes.TRAIN]
        tensorflow_map[DataTypes.TEST]['trajectory_index'] += num_trajectories[DataTypes.TRAIN] \
                                                              + num_trajectories[DataTypes.VALID]

        return tensorflow_map

    def make_rnn_tf_dataset(self, state_window):

        return self._make_rnn_tf_dataset(state_window, self.INPUT_COLS)

    def make_rnf_dataset(self, state_window):
        return self._make_rnn_tf_dataset(state_window, self.RNF_COLS)

    def _make_internal_linear_dataset(self, lags):

        if lags < 1:
            raise ValueError("Lags must be at least 1")

        print("Making linear dataset")

        all_data_types = [DataTypes.TRAIN, DataTypes.VALID, DataTypes.TEST]

        array_maps = self._get_ml_arrays(all_data_types, self.INPUT_COLS)
        output_map = {DataTypes.SCALER: array_maps[DataTypes.SCALER]}

        # Format data to use for RNN
        input_size = self.get_input_size()
        output_size = self.get_output_size()

        for k in all_data_types:
            inputs, outputs = array_maps[k]

            time_steps = inputs.shape[0]
            tmp = {"inputs": [inputs[i:time_steps - (lags-1) + i, :] for i in range(lags)],
                   "outputs": outputs[lags-1:, :]}

            output_map[k] = tmp

        return output_map

    def make_linear_dataset(self, lags):

        output_map = self._make_internal_linear_dataset(lags)

        for k in output_map:
            if k is not DataTypes.SCALER:
                input_list = output_map[k]["inputs"]
                input_list.reverse()  # to get newest at start of list (x_t = a*x_{t-1} + .. style)
                if len(input_list[0].shape) > 1:
                    output_map[k]["inputs"] = np.concatenate(input_list, axis=1)
                else:
                    output_map[k]["inputs"] = np.stack(input_list, axis=1)

        return output_map

    def make_single_run_dataset(self, is_rnf, all_data_types):

        print("Making single run dataset")

        #all_data_types = [DataTypes.TRAIN, DataTypes.VALID, DataTypes.TEST]

        array_maps = self._get_ml_arrays(all_data_types,
                                         self.RNF_COLS if is_rnf else self.INPUT_COLS)
        output_map = {DataTypes.SCALER: array_maps[DataTypes.SCALER]}

        # Format data to use for RNN

        for k in all_data_types:
            inputs, outputs = array_maps[k]

            if len(inputs.shape) < 3:
                inputs = inputs[np.newaxis, :, :]
                outputs = outputs[np.newaxis, :, :]

            # shape into rnn_type dataset
            time_steps = inputs.shape[1]

            tmp = {"inputs": inputs,
                   "outputs": outputs,
                   'sequence_lengths': np.array([time_steps]),
                   'active_entries': np.ones(outputs.shape)}

            output_map[k] = tmp

        return output_map


class BaseCombinationLoader(BaseDataLoader):

    rics = []
    data_loader_class = BaseDataLoader

    def get_input_size(self):
        return self.data_loader_class(self.rics[0]).get_input_size()

    def get_output_size(self):
        return self.data_loader_class(self.rics[0]).get_output_size()

    def _make_combined_tf_dataset(self, get_data_fxn):
        rics = self.rics

        # Load all the data first
        data_map = Parallel(n_jobs=5, prefer="threads")(delayed(get_data_fxn)(ric) for ric in rics)
        data_map = {rics[i]: data_map[i] for i in range(len(rics))}

        # Combine train, valid, test data into single map
        data_types = [DataTypes.TRAIN, DataTypes.VALID, DataTypes.TEST]
        combined_map = {k: {} for k in data_types}
        for data_type in data_types:
            for k in ["inputs", "outputs", "sequence_lengths", 'active_entries']:
                combined_map[data_type][k] = np.concatenate([data_map[ric][data_type][k] for ric in rics])

                # Free up some memory
                if data_type != DataTypes.TEST:
                    for ric in rics:
                        del data_map[ric][data_type][k]

        return combined_map

    def make_rnn_tf_dataset(self, state_window):

        LoaderClass = self.data_loader_class
        get_data_fxn = lambda r: LoaderClass(r).make_rnn_tf_dataset(state_window=state_window)

        return self._make_combined_tf_dataset(get_data_fxn)

    def make_rnf_dataset(self, state_window):

        LoaderClass = self.data_loader_class
        get_data_fxn = lambda r: LoaderClass(r).make_rnf_dataset(state_window=state_window)

        return self._make_combined_tf_dataset(get_data_fxn)

    def make_linear_dataset(self, lags):
        LoaderClass = self.data_loader_class
        get_data_fxn = lambda r: LoaderClass(r).make_linear_dataset(lags)

        rics = self.rics

        # Parallel datamap
        data_map = Parallel(n_jobs=5, prefer="threads")(delayed(get_data_fxn)(ric) for ric in rics)
        data_map = {rics[i]: data_map[i] for i in range(len(rics))}

        # Combine train and valid data
        train_map = {}
        test_map = {}
        keys = ["inputs", "outputs"]
        for k in keys:
            train_map[k] = np.concatenate([np.concatenate([data_map[ric][DataTypes.TRAIN][k],
                                                           data_map[ric][DataTypes.VALID][k]])
                                           for ric in rics])
            test_map[k] = np.concatenate([data_map[ric][DataTypes.TEST][k] for ric in rics])

            # Remove some unnecessary items to free up memory
            for ric in rics:
                del data_map[ric][DataTypes.TRAIN][k]
                del data_map[ric][DataTypes.VALID][k]

        return {DataTypes.TRAIN: train_map,
                DataTypes.TEST: test_map}