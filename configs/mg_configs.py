"""
mg_configs.py


Created by limsi on 03/04/2019
"""

import os
import numpy as np
import pandas as pd
from configs.common_configs import DATA_ROOT, MODEL_ROOT, PLOT_ROOT
import configs.common_configs as common


# DATA FOLDERS
_expt_name = "mackey"
DATA_FOLDER = os.path.join(DATA_ROOT, _expt_name)

# SERIALISATION FOLDERS
MODEL_FOLDER = os.path.join(MODEL_ROOT, _expt_name)
PLOT_FOLDER = PLOT_ROOT
RESULTS_FOLDER = os.path.join(common.NOTEBOOK_FOLDER, "results")

# Other constants
RANDOM_SEARCH_ITERATIONS = 100
MAX_TRAJECTORY_LENGTH = int(5e4)


# In[*]: Functions


def get_core_simulation():

    time_steps = MAX_TRAJECTORY_LENGTH*100
    obs_noise_std = 0.0
    seed = 63

    traj = _simulate_delayed_mackey_glass(time_steps, obs_noise_std, seed, pow=10)

    return pd.Series(traj).sort_index()

def get_short_simulation():
    time_steps = MAX_TRAJECTORY_LENGTH
    obs_noise_std = 0.0
    seed = 63

    traj = _simulate_delayed_mackey_glass(time_steps, obs_noise_std, seed, tau=50, pow=10, burnin=10000)

    return pd.Series(traj).sort_index()


def get_simple_simulation():
    time_steps = MAX_TRAJECTORY_LENGTH*100
    obs_noise_std = 0.0
    seed = 63

    traj = _simulate_delayed_mackey_glass(time_steps, obs_noise_std, seed, tau=50, pow=2, burnin=10000)

    return pd.Series(traj).sort_index()


def get_short_simple():

    time_steps = MAX_TRAJECTORY_LENGTH
    obs_noise_std = 0.0
    seed = 63

    traj = _simulate_delayed_mackey_glass(time_steps, obs_noise_std, seed, tau=50, pow=2, burnin=10000)

    return pd.Series(traj).sort_index()


def get_noisy_simulation():
    time_steps = MAX_TRAJECTORY_LENGTH
    obs_noise_std = 0.0
    seed = 63

    traj = _simulate_delayed_mackey_glass(time_steps, obs_noise_std, seed, tau=100, pow=3, burnin=10000,
                                          beta=-1.0, alpha=1.0, trans_noise_std=0.05)

    return pd.Series(traj).sort_index()


def _simulate_delayed_mackey_glass(time_steps, obs_noise_std=0.0, seed=100, tau=45, pow=10, burnin=None,
                                   beta=-0.1, alpha=0.2, trans_noise_std=0.01):

    """
    From: https://iopscience.iop.org/article/10.1088/1742-6596/720/1/012002/pdf
    Formula: dx/dt = b*x(t) + a*x(t-delay)/(1+x(t-delay)^10) + noise
    """

    np.random.seed(seed)
    delay = tau + 1  # offset by 1 extra to get y[i-1] and y[i-delay] to be spaced by 50 steps

    if burnin is None:
        burnin = delay

    transition_noise = trans_noise_std*np.random.rand(time_steps+burnin)
    y = np.zeros(time_steps+burnin)

    obs_noise = obs_noise_std*np.random.rand(time_steps)

    y[0] = 1.0  # start at 1.0

    for i in range(1, time_steps+burnin):

        if i < delay:
            y[i] = (1+beta) * y[i - 1] + transition_noise[i]
        else:
            y[i] = (1+beta)*y[i-1] + alpha * y[i-delay] / (1+y[i-delay]**pow) + transition_noise[i]

    y_final = y[burnin:] + obs_noise

    # plt.plot(y_final)
    # plt.show()

    return y_final


# In[*]: Test runtime
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = get_noisy_simulation()

    x.plot()
    plt.show()



