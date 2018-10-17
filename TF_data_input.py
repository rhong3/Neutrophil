from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class DataSet(object):

    def __init__(self,
                 mode,
                 filename,
                 ep,
                 bs,
                 count):
        self._mode = mode
        self._filename = filename
        self._maxep = ep
        self._batchsize = bs
        self._index_in_epoch = 0
        self._num_examples = count

    def next_batch(self, batch_size):
        with tf.Session() as sess:
            feature = {self._mode + '/image': tf.FixedLenFeature([], tf.string),
                       self._mode + '/label': tf.FixedLenFeature([], tf.int64)}
            # Create a list of filenames and pass it to a queue
            filename_queue = tf.train.string_input_producer([self._filename], num_epochs=self._maxep)
            # Define a reader and read the next record
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            # Decode the record read by the reader
            features = tf.parse_single_example(serialized_example, features=feature)
            # Convert the image data from string back to the numbers
            image = tf.decode_raw(features[self._mode + '/image'], tf.float32)

            # Cast label data into int32
            label = tf.cast(features[self._mode + '/label'], tf.int32)
            # Reshape image data into the original shape
            image = tf.reshape(image, [299, 299, 3])

            if self._mode == 'train':
                # Creates batches by randomly shuffling tensors
                self.imgs, self.lbs = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=50000, num_threads=4,
                                                   min_after_dequeue=10000, allow_smaller_final_batch=True)
            else:
                self.imgs, self.lbs = tf.train.batch([image, label], batch_size=batch_size, capacity=50000, num_threads=4,
                                                             min_after_dequeue=10000, allow_smaller_final_batch=True)

            self._images = image
            self._labels = label

        return self.imgs, self.lbs, self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples
