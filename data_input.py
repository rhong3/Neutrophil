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
        features_placeholder = tf.placeholder(self._images.dtype, self._images.shape)
        dataset = tf.data.Dataset.from_tensor_slices(features_placeholder)
        dataset = dataset.repeat(1)
        batched_dataset = dataset.batch(batch_size, drop_remainder=False)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        with tf.Session() as sess:
            sess.run(iterator.initializer, feed_dict={features_placeholder: self._images})
            batch = sess.run(next_element)
        return batch

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples
