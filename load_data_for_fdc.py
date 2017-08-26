import tensorflow as tf
from datasets import fast_depth_coding

import os

homedir = os.environ['HOME']

DATA_DIR = homedir + '/data/TFRecords/'

slim = tf.contrib.slim

# Selects the 'validation' dataset.
dataset = fast_depth_coding.get_split('validating', DATA_DIR)

# Creates a TF-Slim DataProvider which reads the dataset in the background
# during both training and testing.
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])
print('haha====>>>>>>>>')
print(type(image))
print(type(image.shape))
print(type(label))
