# Tile a real scn file, load a trained model and run the test.
"""
Created on 09/28/2018

@author: RH
"""

import get_tile
import time
import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import HE_data_input_te
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

dirr = sys.argv[1]
imgfile = sys.argv[2]
bs = sys.argv[3]
md = sys.argv[4]
modeltoload = sys.argv[5]
bs = int(bs)

IMG_DIM = 299

INPUT_DIM = [IMG_DIM ** 2 * 3,
             IMG_DIM, IMG_DIM]

HYPERPARAMS = {
    "batch_size": bs,
    "dropout": 0.8,
    "learning_rate": 1E-4
}
METAGRAPH_DIR = "../Neutrophil/{}".format(dirr)
data_dir = "../Neutrophil/{}/data".format(dirr)
out_dir = "../Neutrophil/{}/out".format(dirr)
LOG_DIR = "../Neutrophil/{}".format(dirr)

try:
    os.mkdir(METAGRAPH_DIR)
except(FileExistsError):
    pass

try:
    os.mkdir(data_dir)
except(FileExistsError):
    pass

try:
    os.mkdir(out_dir)
except(FileExistsError):
    pass

try:
    os.mkdir(LOG_DIR)
except(FileExistsError):
    pass


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


def load_HE_data(train_dat_name, valid_dat_name):
    train_dat = iter_loadtxt(train_dat_name, dtype=int, delimiter='\t')
    valid_dat = iter_loadtxt(valid_dat_name, dtype=int, delimiter='\t')

    class DataSets(object):
        pass

    data_sets = DataSets()

    data_sets.train = HE_data_input_te.DataSet(images=train_dat, reshape=False)

    data_sets.validation = HE_data_input_te.DataSet(images=valid_dat, reshape=False)
    return data_sets


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


def CAM(net, w, pred, x, path, name, prlista):
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

    prlista.extend(prl)

    newprlist = prlista

    for ij in range(len(prl)):

        if prl[ij] == 0:

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
                imname3 = DIRR + '/' + str(ij) + '_full.png'
                # cv2.imwrite(imname, curHeatMap)
                # cv2.imwrite(imname1, a)
                # cv2.imwrite(imname2, b)
                cv2.imwrite(imname3, full)


        else:

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
                imname3 = DIR + '/' + str(ij) + '_full.png'
                # cv2.imwrite(imname, curHeatMap)
                # cv2.imwrite(imname1, a)
                # cv2.imwrite(imname2, b)
                cv2.imwrite(imname3, full)
    return newprlist


def test(tenum, tec, to_reload=None):

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
    else:
        m = cnng.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR)

    print("Loaded! Ready for test!", flush=True)

    prlist = []

    for a in range(tenum):

        aa = str(a + 1)

        tdat_f = data_dir + '/data-{}.txt'.format(aa)

        HET = load_HE_data(train_dat_name=tdat_f, valid_dat_name=tdat_f)

        if tec >= 5000:

            for b in range(5):
                x = HET.validation.next_batch(1000)
                print('Test:')
                te, tenet, tew = m.inference(x)
                prlist = CAM(tenet, tew, te, x, dirr, 'Test', prlist)

        elif tec in range(1000, 5000):
            mppp = int(tec / 1000)+1

            for b in range(mppp):

                if b == mppp -1:
                    x = HET.validation.next_batch(tec%1000)
                else:
                    x = HET.validation.next_batch(1000)
                print('Test:')
                te, tenet, tew = m.inference(x)
                prlist = CAM(tenet, tew, te, x, dirr, 'Test', prlist)

        else:
            x, y = HET.validation.next_batch(tec)
            print('Test:')
            te, tenet, tew = m.inference(x)
            prlist = CAM(tenet, tew, te, x, dirr, 'Test', prlist)

    return prlist

# cut tiles with coordinates in the name (exclude white)

if not os.path.isfile(data_dir+'/dict.csv'):
    get_tile.tile(image_file=sys.argv[6], outdir = METAGRAPH_DIR)

dict = pd.read_csv(data_dir+'/dict.csv', header=0)
tec = len(dict["num"])
tenum = int(tec/5000)+1
listfinal = test(tenum, tec, to_reload=modeltoload)
se = pd.Series(listfinal)
dict['prediction'] = se
dict.to_csv(out_dir+'/finaldict.csv', index = False)

start_time = time.time()

print("--- %s seconds ---" % (time.time() - start_time))

# load 5000 each time until it is done



# output heat map of pos and neg; and output CAM and assemble them to a big graph.

