#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:15:18 2017

@authors: lwk, RH
"""
import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import tensorflow as tf
import HE_data_input
import cnnm
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt

num = sys.argv[1]
dirr = sys.argv[2]
trn = sys.argv[3]
vln = sys.argv[4]
ttt = sys.argv[5]

IMG_DIM = 299

INPUT_DIM = [IMG_DIM ** 2 * 3,  # Default input for INCEPTION_V3 network, 299*299*3
             IMG_DIM, IMG_DIM]

HYPERPARAMS = {
    "batch_size": 128,
    "dropout": 0.8,
    "learning_rate": 1E-4
}

MAX_ITER = 2 ** 16
MAX_EPOCHS = np.inf

LOG_DIR = "../Neutrophil/{}".format(dirr)
METAGRAPH_DIR = "../Neutrophil/{}".format(dirr)


# to_load =

def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


def load_HE_data(train_dat_name, train_lab_name, valid_dat_name, valid_lab_name):
    train_dat = iter_loadtxt(train_dat_name, dtype=int, delimiter='\t')
    valid_dat = iter_loadtxt(valid_dat_name, dtype=int, delimiter='\t')
    train_lab = iter_loadtxt(train_lab_name, dtype=int, delimiter='\t')
    valid_lab = iter_loadtxt(valid_lab_name, dtype=int, delimiter='\t')

    class DataSets(object):
        pass

    data_sets = DataSets()

    data_sets.train = HE_data_input.DataSet(images=train_dat,
                                            labels=train_lab,
                                            reshape=False)

    data_sets.validation = HE_data_input.DataSet(images=valid_dat,
                                                 labels=valid_lab,
                                                 reshape=False)
    return data_sets


def metrics(pdx, tl, path, name):
    pdx = np.asmatrix(pdx)

    prl = (pdx[:,1] > 0.5).astype('uint8')
    prl = pd.DataFrame(prl, columns = ['Prediction'])
    out = pd.DataFrame(pdx, columns = ['neg_score', 'pos_score'])
    outtl = pd.DataFrame(tl, columns = ['True_label'])
    out = pd.concat([out,prl,outtl], join='outer')
    out.to_csv("../Neutrophil/{}/{}.csv".format(path, name), headers=0, index=False)

    y_score = pdx[:,1]
    auc = skl.metrics.roc_auc_score(tl, y_score)
    print('ROC-AUC:')
    print(skl.metrics.roc_auc_score(tl, y_score))
    fpr, tpr, _ = skl.metrics.roc_curve(tl, y_score)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of {}'.format(name))
    plt.legend(loc="lower right")
    plt.savefig("../Neutrophil/{}/{}_ROC.png".format(path, name))


def main(to_reload=None, test=None):
    dat_f = '../Neutrophil/{}_Tiles_final/slide{}_data_{}.txt'.format(trn, trn, num)

    lab_f = '../Neutrophil/{}_Tiles_final/slide{}_lab_{}.txt'.format(trn, trn, num)

    vdat_f = '../Neutrophil/{}_Tiles_final/slide{}_data_1.txt'.format(vln, vln)

    vlab_f = '../Neutrophil/{}_Tiles_final/slide{}_lab_1.txt'.format(vln, vln)

    tdat_f = '../Neutrophil/{}_Tiles_final/slide{}_data_1.txt'.format(ttt, ttt)

    tlab_f = '../Neutrophil/{}_Tiles_final/slide{}_lab_1.txt'.format(ttt, ttt)


    HE = load_HE_data(train_dat_name=dat_f,
                      train_lab_name=lab_f,
                      valid_dat_name=vdat_f,
                      valid_lab_name=vlab_f)


    HET = load_HE_data(train_dat_name=dat_f,
                      train_lab_name=lab_f,
                      valid_dat_name=tdat_f,
                      valid_lab_name=tlab_f)


    if to_reload:
        m = cnnm.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload)
        print("Loaded!", flush=True)
        m.train(HE, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS,
                verbose=True, save=True, outdir=METAGRAPH_DIR)

        x, y = HE.train.next_batch(HE.train._num_examples)
        print('Generating metrics')
        tr = m.inference(x)
        metrics(tr, y, dirr, 'Train_{}'.format(num))

        x, y = HE.validation.next_batch(HE.validation._num_examples)
        print('Validation:')
        va = m.inference(x)
        metrics(va, y, dirr, 'Validation')

        x, y = HET.validation.next_batch(HET.validation._num_examples)
        print('Test:')
        te = m.inference(x)
        metrics(te, y, dirr, 'Test')

    elif test:  # restore

        m = cnnm.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload)
        print("Loaded! Ready for test!", flush=True)

        x, y = HET.validation.next_batch(HET.validation._num_examples)
        print('Test:')
        te = m.inference(x)
        metrics(te, y, dirr, 'Test')


    else:  # train
        """to try cont'd training, load data from previously saved meta graph"""
        m = cnnm.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        m.train(HE, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS,
                verbose=True, save=True, outdir=METAGRAPH_DIR)
        print("Trained!", flush=True)

        x, y = HE.train.next_batch(HE.train._num_examples)
        print('Generating metrics')
        tr = m.inference(x)
        metrics(tr, y, dirr, 'Train_{}'.format(num))

        x, y = HE.validation.next_batch(HE.validation._num_examples)
        print('Validation:')
        va = m.inference(x)
        metrics(va, y, dirr, 'Validation')

        x, y = HET.validation.next_batch(HET.validation._num_examples)
        print('Test:')
        te = m.inference(x)
        metrics(te, y, dirr, 'Test')


if __name__ == "__main__":
    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    try:
        to_reload = sys.argv[6]
        main(to_reload=to_reload)
    except(IndexError):
        main()
