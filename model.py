import numpy as np
from numpy import array
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScalermnist = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

