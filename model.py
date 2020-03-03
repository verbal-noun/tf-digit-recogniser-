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

labels = np.array(mnist.pop('label'))
labels = LabelEncoder().fit_transform(labels)[:, None]
labels = OneHotEncoder().fit_transform(labels).todense()
mnist = StandardScaler().fit_transform(np.float32(mnist.values))
mnist = mnist.reshape(-1, IMAGE_WIDTH, IMAGE_WIDTH, COLOR_CHANNELS)
train_data, valid_data = mnist[:-VALID_SIZE], mnist[-VALID_SIZE:]
train_labels, valid_labels = labels[:-VALID_SIZE], labels[-VALID_SIZE:]

tf_data = tf.placeholder(tf.float32, shape=(None, WIDTH, WIDTH, CHANNELS))
tf_labels = tf.placeholder(tf.float32, shape=(None, LABELS))