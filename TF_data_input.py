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
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = count

    def next_batch(self):
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
            image = tf.decode_raw(features['train/image'], tf.float32)

            # Cast label data into int32
            label = tf.cast(features['train/label'], tf.int32)
            # Reshape image data into the original shape
            image = tf.reshape(image, [299, 299, 3])

            # Creates batches by randomly shuffling tensors
            imgs, lbs = tf.train.shuffle_batch([image, label], batch_size=self._batchsize, capacity=50000, num_threads=4,
                                               min_after_dequeue=10000)

            self._images = image
            self._labels = label

            # Initialize all global and local variables
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            self.img, self.lbl = sess.run([imgs, lbs])
            self.img = self.img.astype(np.uint8)

            # Stop the threads
            coord.request_stop()

            # Wait for threads to stop
            coord.join(threads)
            sess.close()

        return self.img, self.lbl

        # start = self._index_in_epoch
        # self._index_in_epoch += batch_size
        # if self._index_in_epoch > self._num_examples:
        #     # Finished epoch
        #     self._epochs_completed += 1
        #     # Shuffle the data
        #     perm = np.arange(self._num_examples)
        #     np.random.shuffle(perm)
        #     self._images = self._images[perm]
        #     self._labels = self._labels[perm]
        #     # Start next epoch
        #     start = 0
        #     self._index_in_epoch = batch_size
        #     assert batch_size <= self._num_examples
        # end = self._index_in_epoch
        # return self._images[start:end], self._labels[start:end]

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed