"""
common_configs.py


Created by limsi on 02/04/2019
"""


import os
import pandas as pd
import numpy as np

# Common
ROOT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
LIB_FOLDER = os.path.join(ROOT_FOLDER, "libs")

# Results
NOTEBOOK_FOLDER = os.path.join(ROOT_FOLDER, "notebooks")
PLOT_ROOT = os.path.join(NOTEBOOK_FOLDER, "plot data")
RESULTS_ROOT = os.path.join(ROOT_FOLDER, "results")

# External storage
_template = r"D:\{}"  # "/home/sjblim/Code/{}"   # r"/nfs/home/bryanl/{}"  #r"/nfs/home/bryanl/{}"  #
DATA_ROOT = _template.format("Data")
MODEL_ROOT = _template.format("Models")

# RNN Spec Params
# RNN_WINDOW = 50

