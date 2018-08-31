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
import cnng
import cnni
import cnnt
import cnnir1
import cnnir2
import cnnva
import cnnv16
import cnnv19
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
from PIL import Image
import cv2


num = sys.argv[1]
dirr = sys.argv[2]
bs = sys.argv[3]
iter = sys.argv[4]
md = sys.argv[5]
bs = int(bs)
iter = int(iter)

IMG_DIM = 299

INPUT_DIM = [IMG_DIM ** 2 * 3,  # Default input for INCEPTION_V3 network, 299*299*3
             IMG_DIM, IMG_DIM]

HYPERPARAMS = {
    "batch_size": bs,
    "dropout": 0.8,
    "learning_rate": 1E-4
}

MAX_ITER = iter
MAX_EPOCHS = np.inf

img_dir = '../Neutrophil/All_Tiles_final/tot_sample.csv'
LOG_DIR = "../Neutrophil/{}".format(dirr)
METAGRAPH_DIR = "../Neutrophil/{}".format(dirr)
data_dir = "../Neutrophil/{}/data".format(dirr)
out_dir = "../Neutrophil/{}/out".format(dirr)


def loader(totlist_dir):
    dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')
    tile_lab = []
    totlist = pd.read_csv(totlist_dir, header=0)
    f = 1
    for index, row in totlist.iterrows():
        image = Image.open(row['path'])
        pix = np.array(image)[:, :, 0:3]
        dat = np.vstack([dat, pix.flatten()])
        tile_lab.append(row['label'])
        if len(tile_lab) == 5000 and index != len(totlist['label']) - 1:
            np.savetxt(data_dir + '/data_{}.txt'.format(f), dat, fmt='%i', delimiter='\t')
            np.savetxt(data_dir + '/lab_{}.txt'.format(f), tile_lab, fmt='%i', delimiter='\t')
            dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')
            tile_lab = []
            f += 1
        elif index == len(totlist['label']) - 1:
            np.savetxt(data_dir + '/data_test.txt', dat, fmt='%i', delimiter='\t')
            np.savetxt(data_dir + '/lab_test.txt', tile_lab, fmt='%i', delimiter='\t')
            dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')
            tile_lab = []
            f += 1


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
    out = pd.concat([out,prl,outtl], axis=1)
    out.to_csv("../Neutrophil/{}/out/{}.csv".format(path, name), index=False)
    accu = 0
    tott = out.shape[0]
    for idx, row in out.iterrows():
        if row['Prediction'] == row['True_label']:
            accu += 1
    accur = accu/tott
    accur = round(accur,2)
    print('Accuracy:')
    print(accur)

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
    plt.savefig("../Neutrophil/{}/out/{}_ROC.png".format(path, name))

    average_precision = skl.metrics.average_precision_score(tl, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    plt.figure()
    precision, recall, _ = skl.metrics.precision_recall_curve(tl, y_score)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('{} Precision-Recall curve: AP={:0.2f}; Accu={}'.format(name, average_precision, accur))
    plt.savefig("../Neutrophil/{}/out/{}_PRC.png".format(path, name))


def py_returnCAMmap(activation, weights_LR):
    n_feat, w, h, n = activation.shape
    act_vec = np.reshape(activation, [n_feat, w*h])
    n_top = weights_LR.shape[0]
    out = np.zeros([w, h, n_top])

    for t in range(n_top):
        weights_vec = np.reshape(weights_LR[t], [1, weights_LR[t].shape[0]])
        heatmap_vec = np.dot(weights_vec,act_vec)
        heatmap = np.reshape( np.squeeze(heatmap_vec) , [w, h])
        out[:,:,t] = heatmap

    return out


def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


def py_map2jpg(imgmap, rang, colorMap):
    if rang is None:
        rang = [np.min(imgmap), np.max(imgmap)]

    heatmap_x = np.round(imgmap*255).astype(np.uint8)

    return cv2.applyColorMap(heatmap_x, cv2.COLORMAP_JET)


def CAM(net, w, pred, x, y, path, name):
    DIR = "../Neutrophil/{}/out/{}_posimg".format(path, name)
    DIRR = "../Neutrophil/{}/out/{}_negimg".format(path, name)

    try:
        os.mkdir(DIR)
    except(FileExistsError):
        pass

    try:
        os.mkdir(DIRR)
    except(FileExistsError):
        pass

    pdx = np.asmatrix(pred)

    prl = (pdx[:,1] > 0.5).astype('uint8')

    for ij in range(len(y)):

        if prl[ij] == 0:
            if y[ij] == 0:
                ddt = 'Correct'
            else:
                ddt = 'Wrong'

            weights_LR = w
            activation_lastconv = np.array([net[ij]])
            weights_LR = weights_LR.T
            activation_lastconv = activation_lastconv.T

            topNum = 1  # generate heatmap for top X prediction results
            scores = pred[ij]
            scoresMean = np.mean(scores, axis=0)
            ascending_order = np.argsort(scoresMean)
            IDX_category = ascending_order[::-1]  # [::-1] to sort in descending order
            curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[[0], :])
            for kk in range(topNum):
                curCAMmap_crops = curCAMmapAll[:, :, kk]
                curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (299, 299))
                curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops), (299, 299))  # this line is not doing much
                curHeatMap = im2double(curHeatMap)
                curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
                xim = x[ij].reshape(-1, 3)
                xim1 = xim[:, 0].reshape(-1, 299)
                xim2 = xim[:, 1].reshape(-1, 299)
                xim3 = xim[:, 2].reshape(-1, 299)
                image = np.empty([299,299,3])
                image[:, :, 0] = xim1
                image[:, :, 1] = xim2
                image[:, :, 2] = xim3
                a = im2double(image) * 255
                b = im2double(curHeatMap) * 255
                curHeatMap = a * 0.6 + b * 0.4
                ab = np.hstack((a,b))
                full = np.hstack((curHeatMap, ab))
                # imname = DIRR + '/' + ddt + str(ij) + '.png'
                # imname1 = DIRR + '/' + ddt + str(ij) + '_img.png'
                # imname2 = DIRR+ '/' + ddt + str(ij) + '_hm.png'
                imname3 = DIRR + '/' + ddt + str(ij) + '_full.png'
                # cv2.imwrite(imname, curHeatMap)
                # cv2.imwrite(imname1, a)
                # cv2.imwrite(imname2, b)
                cv2.imwrite(imname3, full)


        else:
            if y[ij] == 1:
                ddt = 'Correct'
            else:
                ddt = 'Wrong'

            weights_LR = w
            activation_lastconv = np.array([net[ij]])
            weights_LR = weights_LR.T
            activation_lastconv = activation_lastconv.T

            topNum = 1  # generate heatmap for top X prediction results
            scores = pred[ij]
            scoresMean = np.mean(scores, axis=0)
            ascending_order = np.argsort(scoresMean)
            IDX_category = ascending_order[::-1]  # [::-1] to sort in descending order
            curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[[1], :])
            for kk in range(topNum):
                curCAMmap_crops = curCAMmapAll[:, :, kk]
                curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (299, 299))
                curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops), (299, 299))  # this line is not doing much
                curHeatMap = im2double(curHeatMap)
                curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
                xim = x[ij].reshape(-1, 3)
                xim1 = xim[:, 0].reshape(-1, 299)
                xim2 = xim[:, 1].reshape(-1, 299)
                xim3 = xim[:, 2].reshape(-1, 299)
                image = np.empty([299,299,3])
                image[:, :, 0] = xim1
                image[:, :, 1] = xim2
                image[:, :, 2] = xim3
                a = im2double(image) * 255
                b = im2double(curHeatMap) * 255
                curHeatMap = a * 0.6 + b * 0.4
                ab = np.hstack((a,b))
                full = np.hstack((curHeatMap, ab))
                # imname = DIR + '/' + ddt + str(ij) + '.png'
                # imname1 = DIR + '/' + ddt + str(ij) + '_img.png'
                # imname2 = DIR + '/' + ddt + str(ij) + '_hm.png'
                imname3 = DIR + '/' + ddt + str(ij) + '_full.png'
                # cv2.imwrite(imname, curHeatMap)
                # cv2.imwrite(imname1, a)
                # cv2.imwrite(imname2, b)
                cv2.imwrite(imname3, full)


def main(to_reload=None, test=None, log_dir=None):
    dat_f = data_dir + '/data_{}.txt'.format(num)

    lab_f = data_dir + '/lab_{}.txt'.format(num)

    tdat_f = data_dir + '/data_test.txt'

    tlab_f = data_dir + '/lab_test.txt'


    HE = load_HE_data(train_dat_name=dat_f,
                      train_lab_name=lab_f,
                      valid_dat_name=dat_f,
                      valid_lab_name=lab_f)


    HET = load_HE_data(train_dat_name=dat_f,
                      train_lab_name=lab_f,
                      valid_dat_name=tdat_f,
                      valid_lab_name=tlab_f)


    if to_reload:
        if md == 'IG':
            m = cnng.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'I2':
            m = cnnt.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'I3':
            m = cnnm.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'I4':
            m = cnni.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'IR1':
            m = cnnir1.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'IR2':
            m = cnnir2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'VA':
            m = cnnva.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'V16':
            m = cnnv16.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)
        elif md == 'V19':
            m = cnnv19.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)

        print("Loaded!", flush=True)
        m.train(HE, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS,
                verbose=True, save=True, outdir=METAGRAPH_DIR)

        x, y = HE.train.next_batch(1024)
        print('Generating metrics')
        tr, trnet, trw = m.inference(x)
        CAM(trnet, trw, tr, x, y, dirr, 'Train_{}'.format(num))
        metrics(tr, y, dirr, 'Train_{}'.format(num))

        x, y = HET.validation.next_batch(1024)
        print('Test:')
        te, tenet, tew = m.inference(x)
        CAM(tenet, tew, te, x, y, dirr, 'Test_{}'.format(num))
        metrics(te, y, dirr, 'Test_{}'.format(num))

    elif test:  # restore
        if md == 'IG':
            m = cnng.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload)
        elif md == 'I2':
            m = cnnt.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload)
        elif md == 'I3':
            m = cnnm.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload)
        elif md == 'I4':
            m = cnni.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload)
        elif md == 'IR1':
            m = cnnir1.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload)
        elif md == 'IR2':
            m = cnnir2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload)
        elif md == 'VA':
            m = cnnva.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload)
        elif md == 'V16':
            m = cnnv16.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload)
        elif md == 'V19':
            m = cnnv19.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload)

        print("Loaded! Ready for test!", flush=True)

        x, y = HET.validation.next_batch(1024)
        print('Test:')
        te, tenet, tew = m.inference(x)
        CAM(tenet, tew, te, x, y, dirr, 'Test_{}'.format(num))
        metrics(te, y, dirr, 'Test')


    else:  # train
        """to try cont'd training, load data from previously saved meta graph"""
        if md == 'IG':
            m = cnng.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'I2':
            m = cnnm.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'I3':
            m = cnnm.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'I4':
            m = cnni.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'IR1':
            m = cnnir1.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'IR2':
            m = cnnir2.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'VA':
            m = cnnva.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'V16':
            m = cnnv16.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'V19':
            m = cnnv19.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)

        m.train(HE, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS,
                verbose=True, save=True, outdir=METAGRAPH_DIR)
        print("Trained!", flush=True)

        x, y = HE.train.next_batch(1024)
        print('Generating training metrics')
        tr, trnet, trw = m.inference(x)
        CAM(trnet, trw, tr, x, y, dirr, 'Train_{}'.format(num))
        metrics(tr, y, dirr, 'Train_{}'.format(num))

        x, y = HET.validation.next_batch(1024)
        print('Test:')
        te, tenet, tew = m.inference(x)
        CAM(tenet, tew, te, x, y, dirr, 'Test_{}'.format(num))
        metrics(te, y, dirr, 'Test_{}'.format(num))


if __name__ == "__main__":
    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR, data_dir, out_dir):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    try:
        to_reload = sys.argv[6]
        main(to_reload=to_reload, log_dir=LOG_DIR)
    except(IndexError):
        if not os.path.isfile(data_dir + '/lab_test.txt'):
            loader(img_dir)
        main()


