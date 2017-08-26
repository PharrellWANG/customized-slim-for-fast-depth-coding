# Copyright 2016 Pharrell. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

RESHAPE = 32

# _FILE_PATTERN = 'flowers_%s_*.tfrecord'
_FILE_PATTERN = '%sx%s_%s.tfrecord'
# 16x16_training.tfrecord

SPLITS_TO_SIZES = {'training': 62271, 'validating': 20757}

_NUM_CLASSES = 37

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [%s x %s x 1] gray scale depth image.' % (RESHAPE, RESHAPE),
    'label': 'A single integer between 0 and 36',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None,
              reshape=None, training_size=None, validating_size=None):
    """Gets a dataset tuple with instructions for reading flowers.

    Args:
      reshape: the size of the block
      split_name: A train/validation split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

      training_size: number of samples used for training
      validating_size: number of samples used for validating

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/validation split.
    """

    if not training_size:
        split_to_sizes = SPLITS_TO_SIZES
    else:
        split_to_sizes = {'training': training_size,
                          'validating': validating_size}

    if not reshape:
        reshape = RESHAPE

    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir,
                                file_pattern % (reshape, reshape, split_name))
    print('i show you the file pattern')
    print(file_pattern)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    # keys_to_features = {
    #     'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    #     'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
    #     'image/class/label': tf.FixedLenFeature(
    #         [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    # }
    #
    # items_to_handlers = {
    #     'image': slim.tfexample_decoder.Image(shape=[16, 16, 1], channels=1),
    #     'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
    # }
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=split_to_sizes[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names)
