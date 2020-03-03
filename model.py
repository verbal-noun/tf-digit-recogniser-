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

#generates a weight variable of a given shape.
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  #generates a bias variable of a given shape.
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#returns a 2d convolution layer with full stride
def conv_2d(x, W):
    #down samples a feature map by 2X
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# First convolution layer - maps one grayscale image to 8 feature maps.
w1 = weight_variable([5, 5, 1, 32])
b1 = bias_variable([32])
# Pooling layer - downsamples by 2X.
layer_conv1 = tf.nn.relu(conv_2d(x, w1) + b1)
layer_pool1 = max_pool_2x2(layer_conv1)

# Second convolutional layer -- maps 32 feature maps to 64.
w2 = weight_variable([5, 5, 32, 64])
b2 = bias_variable([64])
# Second pooling layer.
layer_conv2 = tf.nn.relu(conv_2d(layer_pool1, w2) + b2)
layer_pool2 = max_pool_2x2(layer_conv2)

wfc1 = weight_variable([7*7*64, 1024])
bfc1 = bias_variable([1024])flatten_pool2 = tf.reshape(layer_pool2, [-1,7*7*64])
layer_fc1 = tf.nn.relu(tf.matmul(flatten_pool2, wfc1) + bfc1)