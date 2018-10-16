#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/28/2018

@author: RH
"""
import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import tensorflow as tf
import TF_data_input
import cnnm2
import cnng2
import cnni2
import cnnt2
import cnnir12
import cnnir22
import pandas as pd
import cv2
import Sample_prep


dirr = sys.argv[1]
bs = sys.argv[2]
ep = sys.argv[3]
md = sys.argv[4]
bs = int(bs)
ep = int(ep)

IMG_DIM = 299

INPUT_DIM = [bs, IMG_DIM, IMG_DIM, 3]

HYPERPARAMS = {
    "batch_size": bs,
    "dropout": 0.8,
    "learning_rate": 1E-4
}

MAX_ITER = np.inf
MAX_EPOCHS = ep

img_dir = '../Neutrophil/All_Tiles_final'
LOG_DIR = "../Neutrophil/{}".format(dirr)
METAGRAPH_DIR = "../Neutrophil/{}".format(dirr)
data_dir = "../Neutrophil/{}/data".format(dirr)
out_dir = "../Neutrophil/{}/out".format(dirr)


def counters(totlist_dir):
    trlist = pd.read_csv(totlist_dir + '/tr_sample.csv', header=0)
    telist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    trcc = len(trlist['label']) - 1
    tecc = len(telist['label']) - 1
    trnumm = int(trcc/5000)+1
    tenumm = int(tecc/5000)+1

    return trcc, tecc, trnumm, tenumm


def load_image(addr):
    img = cv2.imread(addr)
    img = img.astype(np.float32)
    return img


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def loader(totlist_dir):
    trlist = pd.read_csv(totlist_dir+'/tr_sample.csv', header=0)
    telist = pd.read_csv(totlist_dir+'/te_sample.csv', header=0)
    trimlist = trlist['path'].values.tolist()
    trlblist = trlist['label'].values.tolist()
    teimlist = telist['path'].values.tolist()
    telblist = telist['label'].values.tolist()

    train_filename = data_dir+'/train.tfrecords'
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(trimlist)):
        if not i % 1000:
            sys.stdout.flush()
        # Load the image
        img = load_image(trimlist[i])
        label = trlblist[i]
        # Create a feature
        feature = {'train/label': _int64_feature(label),
                   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

    test_filename = data_dir+'/test.tfrecords'
    writer = tf.python_io.TFRecordWriter(test_filename)
    for i in range(len(teimlist)):
        if not i % 1000:
            sys.stdout.flush()
        # Load the image
        img = load_image(teimlist[i])
        label = telblist[i]
        # Create a feature
        feature = {'test/label': _int64_feature(label),
                   'tset/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def tfreloader(mode, ep, bs):
    filename = data_dir + '/' + mode + '.tfrecords'
    ctr, cte, _, _ = counters(img_dir)
    if mode == 'train':
        ct = ctr
    else:
        ct = cte
    class DataSets(object):
        pass

    datasets = DataSets()
    datasets.train = TF_data_input.DataSet(mode, filename, ep, bs, ct)
    datasets.validation = TF_data_input.DataSet(mode, filename, ep, bs, ct)

    return datasets, ct




def main(tenum, trnum, trc, tec, reITER=None, old_ITER=None, to_reload=None, test=None, log_dir=None):

    if test:  # restore

        if md == 'IG':
            m = cnng2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'I2':
            m = cnnt2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'I3':
            m = cnnm2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'I4':
            m = cnni2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'IR1':
            m = cnnir12.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'IR2':
            m = cnnir22.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        else:
            m = cnng2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)

        print("Loaded! Ready for test!", flush=True)

        HET = tfreloader('test', ep, bs)

        for a in range(tenum):

            aa = str(a+1)

            tdat_f = data_dir + '/data_test_{}.txt'.format(aa)

            tlab_f = data_dir + '/lab_test_{}.txt'.format(aa)

            HET = load_HE_data(train_dat_name=tdat_f,
                               train_lab_name=tlab_f,
                               valid_dat_name=tdat_f,
                               valid_lab_name=tlab_f)

            ppp = int(5000 / 1024)

            if tec > 5000:

                for b in range(ppp):

                    bb = str(b+1)

                    x, y = HET.validation.next_batch(1024)
                    print('Test:')
                    te, tenet, tew = m.inference(x)
                    CAM(tenet, tew, te, x, y, dirr, 'Test_{}'.format(bb))
                    metrics(te, y, dirr, 'Test_{}'.format(bb))

                tec = tec-5000


            elif tec in range(1024, 5001):
                mppp = int(tec/1024)

                for b in range(mppp):

                    bb = str(b+1+a*5)

                    x, y = HET.validation.next_batch(1024)
                    print('Test:')
                    te, tenet, tew = m.inference(x)
                    CAM(tenet, tew, te, x, y, dirr, 'Test_{}'.format(bb))
                    metrics(te, y, dirr, 'Test_{}'.format(bb))

            else:
                print("Not enough for a test batch!")


    elif to_reload:

        if md == 'IG':
            m = cnng2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'I2':
            m = cnnt2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'I3':
            m = cnnm2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'I4':
            m = cnni2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'IR1':
            m = cnnir12.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'IR2':
            m = cnnir22.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        else:
            m = cnng2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)

        print("Loaded! Restart training.", flush=True)

        for a in range(trnum):

            aa = str(a + 1)

            dat_f = data_dir + '/data_{}.txt'.format(aa)

            lab_f = data_dir + '/lab_{}.txt'.format(aa)


            HE, sz = load_HE_data(train_dat_name=dat_f,
                              train_lab_name=lab_f,
                              valid_dat_name=dat_f,
                              valid_lab_name=lab_f)

            old_ITER = m.get_global_step(HE)[0]

            if sz < 4998:
                reITER = int(sz * reITER/5000)
                MAX_ITER = old_ITER + reITER

            else:
                MAX_ITER = old_ITER + reITER

            if a == trnum-1:
                m.train(HE, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS,
                        verbose=True, save=True, outdir=METAGRAPH_DIR)
            else:
                m.train(HE, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS,
                        verbose=True, save=False, outdir=METAGRAPH_DIR)

            if trc > 1026:
                x, y = HE.validation.next_batch(1024)
                print('Generating metrics')
                tr, trnet, trw = m.inference(x)
                CAM(trnet, trw, tr, x, y, dirr, 'Train_{}'.format(aa))
                metrics(tr, y, dirr, 'Train_{}'.format(aa))
            elif trc in range(50, 1026):
                x, y = HE.validation.next_batch(trc)
                print('Generating metrics')
                tr, trnet, trw = m.inference(x)
                CAM(trnet, trw, tr, x, y, dirr, 'Train_{}'.format(aa))
                metrics(tr, y, dirr, 'Train_{}'.format(aa))
            else:
                print("The last training set is too small! No metrics generated.")

            trc -= 5000



        for at in range(tenum):

            aat = str(at+1)

            tdat_f = data_dir + '/data_test_{}.txt'.format(aat)

            tlab_f = data_dir + '/lab_test_{}.txt'.format(aat)

            HET, _ = load_HE_data(train_dat_name=tdat_f,
                               train_lab_name=tlab_f,
                               valid_dat_name=tdat_f,
                               valid_lab_name=tlab_f)

            ppp = int(5000 / 1024)

            if tec > 5000:

                for b in range(ppp):

                    bb = str(b+1)

                    x, y = HET.validation.next_batch(1024)
                    print('Test:')
                    te, tenet, tew = m.inference(x)
                    CAM(tenet, tew, te, x, y, dirr, 'Test_{}'.format(bb))
                    metrics(te, y, dirr, 'Test_{}'.format(bb))

                tec = tec-5000


            elif tec in range(1024, 5001):
                mppp = int(tec/1024)

                for b in range(mppp):

                    bb = str(b+1+at*5)

                    x, y = HET.validation.next_batch(1024)
                    print('Test:')
                    te, tenet, tew = m.inference(x)
                    CAM(tenet, tew, te, x, y, dirr, 'Test_{}'.format(bb))
                    metrics(te, y, dirr, 'Test_{}'.format(bb))

            else:
                print("Not enough for a test batch!")


    else:  # train

        if md == 'IG':
            m = cnng2.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'I2':
            m = cnnt2.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'I3':
            m = cnnm2.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'I4':
            m = cnni2.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'IR1':
            m = cnnir12.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'IR2':
            m = cnnir22.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        else:
            m = cnng2.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)

        print("Start training!")

        HE, ctt = tfreloader('train', ep, bs)
        itt = int(ctt*ep/bs)

        m.train(HE, max_iter=itt, max_epochs=MAX_EPOCHS, verbose=True, save=True, outdir=METAGRAPH_DIR)

        #     if trc > 1026:
        #         x, y = HE.validation.next_batch(1024)
        #         print('Generating metrics')
        #         tr, trnet, trw = m.inference(x)
        #         CAM(trnet, trw, tr, x, y, dirr, 'Train_{}'.format(aa))
        #         metrics(tr, y, dirr, 'Train_{}'.format(aa))
        #     elif trc in range(50, 1026):
        #         x, y = HE.validation.next_batch(trc)
        #         print('Generating metrics')
        #         tr, trnet, trw = m.inference(x)
        #         CAM(trnet, trw, tr, x, y, dirr, 'Train_{}'.format(aa))
        #         metrics(tr, y, dirr, 'Train_{}'.format(aa))
        #     else:
        #         print("The last training set is too small! No metrics generated.")
        #
        #     trc -= 5000
        #
        # for at in range(tenum):
        #
        #     aat = str(at + 1)
        #
        #     tdat_f = data_dir + '/data_test_{}.txt'.format(aat)
        #
        #     tlab_f = data_dir + '/lab_test_{}.txt'.format(aat)
        #
        #     HET, _ = load_HE_data(train_dat_name=tdat_f,
        #                        train_lab_name=tlab_f,
        #                        valid_dat_name=tdat_f,
        #                        valid_lab_name=tlab_f)
        #
        #     ppp = int(5000 / 1024)
        #
        #     if tec > 5000:
        #
        #         for b in range(ppp):
        #             bb = str(b + 1)
        #
        #             x, y = HET.validation.next_batch(1024)
        #             print('Test:')
        #             te, tenet, tew = m.inference(x)
        #             CAM(tenet, tew, te, x, y, dirr, 'Test_{}'.format(bb))
        #             metrics(te, y, dirr, 'Test_{}'.format(bb))
        #
        #         tec = tec - 5000
        #
        #
        #     elif tec in range(1024, 5001):
        #         mppp = int(tec / 1024)
        #
        #         for b in range(mppp):
        #             bb = str(b + 1 + at * 5)
        #
        #             x, y = HET.validation.next_batch(1024)
        #             print('Test:')
        #             te, tenet, tew = m.inference(x)
        #             CAM(tenet, tew, te, x, y, dirr, 'Test_{}'.format(bb))
        #             metrics(te, y, dirr, 'Test_{}'.format(bb))
        #
        #     else:
        #         print("Not enough for a test batch!")


if __name__ == "__main__":
    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR, data_dir, out_dir):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    _, _, _, tes, trs = Sample_prep.samplesum()
    tes.to_csv(img_dir+'/te_sample.csv', index=False)
    trs.to_csv(img_dir+'/tr_sample.csv', index=False)
    trc, tec, trnum, tenum = counters(img_dir)

    try:
        modeltoload = sys.argv[5]
        try:
            testmode = sys.argv[6]
            main(tenum, trnum, trc, tec, to_reload=modeltoload, log_dir=LOG_DIR, test=True)
        except(IndexError):
            main(tenum, trnum, trc, tec, reITER=iter, to_reload=modeltoload, log_dir=LOG_DIR)
    except(IndexError):
        if not os.path.isfile(data_dir + '/train.tfrecords'.format(str(tenum))):
            loader(img_dir)
        main(tenum, trnum, trc, tec, reITER=iter, old_ITER=0)


