import numpy as np
from numpy import array
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScalermnist = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

LABELS = 10           # Number of labls(1-10)
IMAGE_WIDTH = 28      # Width/height if the image
COLOR_CHANNELS = 1    # Number of color channelsVALID_SIZE = 1000     # Size of the Validation dataEPOCHS = 20000        # Number of epochs to run
BATCH_SIZE = 32       # SGD Batch size
FILTER_SIZE = 5       # Filter size for kernel
DEPTH = 32            # Number of filters/templates
FC_NEURONS = 1024     # Number of neurons in the fully
                      # connected later
LR = 0.001            # Learning rate Alpha for SGD

