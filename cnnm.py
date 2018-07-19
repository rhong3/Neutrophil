#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:02:45 2017

@author: lwk
modified by RH
"""

from datetime import datetime
import os
import sys

import numpy as np
import tensorflow as tf
import inception_v3

slim = tf.contrib.slim


class INCEPTION():
    """
    Use the InceptionV3 architecture

    """

    DEFAULTS = {
        "batch_size": 128,
        "dropout": 0.8,
        "learning_rate": 1E-3
    }

    RESTORE_KEY = "cnn_to_restore"

    def __init__(self, input_dim, d_hyperparams={},
                 save_graph_def=True, meta_graph=None,
                 log_dir="./log"):

        self.input_dim = input_dim
        self.__dict__.update(INCEPTION.DEFAULTS, **d_hyperparams)
        self.sesh = tf.Session()

        if meta_graph:  # load saved graph
            model_name = os.path.basename(meta_graph)
            meta_graph = os.path.abspath(meta_graph)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(
                self.sesh, meta_graph)
            handles = self.sesh.graph.get_collection(INCEPTION.RESTORE_KEY)


        else:  # build graph from scratch
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(INCEPTION.RESTORE_KEY, handle)
            self.sesh.run(tf.global_variables_initializer())

        # unpack handles for tensor ops to feed or fetch for lower layers
        (self.x_in, self.dropout_, self.is_train,
         self.y_in, self.logits, self.pred, self.pred_cost,
         self.global_step, self.train_op, self.merged_summary) = handles

        # print(self.batch_size,flush=True)
        # print(self.learning_rate,flush=True)

        if save_graph_def:  # tensorboard
            try:
                os.mkdir(log_dir + '/training')
                os.mkdir(log_dir + '/validation')

            except(FileExistsError):
                pass

            self.train_logger = tf.summary.FileWriter(log_dir + '/training', self.sesh.graph)
            self.valid_logger = tf.summary.FileWriter(log_dir + '/validation', self.sesh.graph)

    @property
    def step(self):
        return self.global_step.eval(session=self.sesh)

    def _buildGraph(self):
        x_in = tf.placeholder(tf.float32, shape=[None,  # enables variable batch size
                                                 self.input_dim[0]], name="x")
        x_in_reshape = tf.reshape(x_in, [-1, self.input_dim[1], self.input_dim[2], 3])

        dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

        y_in = tf.placeholder(dtype=tf.int8, name="y")

        onehot_labels = tf.one_hot(indices=tf.cast(y_in, tf.int32), depth=2)

        is_train = tf.placeholder_with_default(True, shape=[], name="is_train")

        logits, _ = inception_v3.inception_v3(x_in_reshape,
                                              num_classes=2,
                                              is_training=is_train,
                                              dropout_keep_prob=dropout,
                                              min_depth=16,
                                              depth_multiplier=1.0,
                                              prediction_fn=slim.softmax,
                                              spatial_squeeze=True,
                                              reuse=None,
                                              create_aux_logits=True,
                                              scope='InceptionV3',
                                              global_pool=False)

        pred = tf.nn.softmax(logits, name="prediction")

        global_step = tf.Variable(0, trainable=False)

        pred_cost = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

        tf.summary.scalar("InceptionV3_cost", pred_cost)

        train_op = tf.contrib.layers.optimize_loss(
            loss=pred_cost,
            learning_rate=self.learning_rate,
            global_step=global_step,
            optimizer="Adam")

        merged_summary = tf.summary.merge_all()

        return (x_in, dropout, is_train,
                y_in, logits, pred, pred_cost,
                global_step, train_op, merged_summary)

    def inference(self, x, train_status=False):
        feed_dict = {self.x_in: x, self.is_train: train_status}
        return self.sesh.run(self.pred, feed_dict=feed_dict)

    def train(self, X, max_iter=np.inf, max_epochs=np.inf, cross_validate=True,
              verbose=True, save=True, outdir="./out"):

        if save:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        try:
            err_train = 0
            now = datetime.now().isoformat()[11:]
            print("------- Training begin: {} -------\n".format(now), flush=True)

            while True:
                x, y = X.train.next_batch(self.batch_size)

                feed_dict = {self.x_in: x, self.y_in: y,
                             self.dropout_: self.dropout}

                fetches = [self.merged_summary, self.logits, self.pred,
                           self.pred_cost, self.global_step, self.train_op]

                # Benchmark the learning
                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()

                summary, logits, pred, cost, i, _ = self.sesh.run(fetches, feed_dict
                                                                  # options=run_options,
                                                                  # run_metadata=run_metadata
                                                                  )

                self.train_logger.add_summary(summary, i)

                # get runtime statistics every 1000 runs
                # if i%1000 == 0:
                # self.logger.add_run_metadata(run_metadata, 'step%d' % i)
                err_train += cost

                if i % 1000 == 0 and verbose:
                    # print("round {} --> avg cost: ".format(i), err_train / i, flush=True)
                    print("round {} --> cost: ".format(i), cost, flush=True)

                if i % 1000 == 0 and verbose:  # and i >= 10000:

                    if cross_validate:
                        x, y = X.validation.next_batch(self.batch_size)
                        feed_dict = {self.x_in: x, self.y_in: y}
                        fetches = [self.pred_cost, self.merged_summary]
                        valid_cost, valid_summary = self.sesh.run(fetches, feed_dict)

                        self.valid_logger.add_summary(valid_summary, i)

                        print("round {} --> CV cost: ".format(i), valid_cost, flush=True)

                """    
                if i%50000 == 0 and save:
                    interfile=os.path.join(os.path.abspath(outdir), "{}_cnn_{}".format(
                            self.datetime, "_".join(map(str, self.input_dim))))
                    saver.save(self.sesh, interfile, global_step=self.step)
                """

                if i >= max_iter or X.train.epochs_completed >= max_epochs:
                    print("final avg cost (@ step {} = epoch {}): {}".format(
                        i, X.train.epochs_completed, err_train / i), flush=True)

                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------\n".format(now), flush=True)

                    if save:
                        outfile = os.path.join(os.path.abspath(outdir), "inception3_{}".format("_".join(['dropout', str(self.dropout)])))
                        saver.save(self.sesh, outfile, global_step=None)
                    try:
                        self.train_logger.flush()
                        self.train_logger.close()
                        self.valid_logger.flush()
                        self.valid_logger.close()

                    except(AttributeError):  # not logging
                        continue
                    break

        except(KeyboardInterrupt):
            print("final avg cost (@ step {} = epoch {}): {}".format(
                i, X.train.epochs_completed, err_train / i), flush=True)

            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now), flush=True)

            if save:
                outfile = os.path.join(os.path.abspath(outdir), "inception3_{}".format("_".join(['dropout', str(self.dropout)])))
                saver.save(self.sesh, outfile, global_step=None)
            try:
                self.train_logger.flush()
                self.train_logger.close()
                self.valid_logger.flush()
                self.valid_logger.close()



            except(AttributeError):  # not logging
                print('Not logging', flush=True)

            sys.exit(0)
