from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class DataSet(object):

    def __init__(self,images,
                 bs,
                 count):
        self._batchsize = bs
        self._index_in_epoch = 0
        self._num_examples = count
        self._images = images

    def next_batch(self, Not_Realtest=False):
        batch_size = self._batchsize
        with tf.Session() as sess:
            self._images = tf.train.batch([self._images], batch_size=batch_size, capacity=5000,
                                                     num_threads=4,
                                                     allow_smaller_final_batch=True)
            return self._images, self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples
