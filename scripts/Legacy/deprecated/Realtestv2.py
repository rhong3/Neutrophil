# Tile a real scn file, load a trained model and run the test.
"""
Created on 10/12/2018

@author: RH
"""

import get_tilev2
import time
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
import skimage.morphology as mph

dirr = sys.argv[1]
imgfile = sys.argv[2]
bs = sys.argv[3]
md = sys.argv[4]
modeltoload = sys.argv[5]
metadir = sys.argv[6]
bs = int(bs)

IMG_DIM = 299

INPUT_DIM = [bs, IMG_DIM, IMG_DIM, 3]

HYPERPARAMS = {
    "batch_size": bs,
    "dropout": 0.8,
    "learning_rate": 1E-4
}
LOG_DIR = "../Neutrophil"
data_dir = "../Neutrophil/{}/data".format(dirr)
out_dir = "../Neutrophil/{}/out".format(dirr)
METAGRAPH_DIR = "../Neutrophil/{}".format(metadir)
file_DIR = "../Neutrophil/{}".format(dirr)
img_dir = "../Neutrophil/{}-tiles".format(dirr)

try:
    os.mkdir(METAGRAPH_DIR)
except(FileExistsError):
    pass

try:
    os.mkdir(file_DIR)
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

try:
    os.mkdir(img_dir)
except(FileExistsError):
    pass

def load_image(addr):
    img = cv2.imread(addr)
    img = img.astype(np.float32)
    return img


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def loader(totlist_dir):
    telist = pd.read_csv(totlist_dir+'/dict.csv', header=0)
    teimlist = telist['Loc'].values.tolist()

    test_filename = data_dir+'/test.tfrecords'
    writer = tf.python_io.TFRecordWriter(test_filename)
    for i in range(len(teimlist)):
        if not i % 1000:
            sys.stdout.flush()
        # Load the image
        img = load_image(teimlist[i])
        # Create a feature
        feature = {'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def tfreloader():
    if not os.path.isfile(data_dir + '/test.tfrecords'):
        loader(img_dir)

    filename = data_dir + '/test.tfrecords'

    datasets = TF_data_input.DataSet('test', filename, 1, 1000, None)

    return datasets


def test(to_reload=None):

    if md == 'IG':
        m = cnng2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=METAGRAPH_DIR)
    elif md == 'I2':
        m = cnnt2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=METAGRAPH_DIR)
    elif md == 'I3':
        m = cnnm2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=METAGRAPH_DIR)
    elif md == 'I4':
        m = cnni2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=METAGRAPH_DIR)
    elif md == 'IR1':
        m = cnnir12.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=METAGRAPH_DIR)
    elif md == 'IR2':
        m = cnnir22.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=METAGRAPH_DIR)
    else:
        m = cnng2.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=METAGRAPH_DIR)

    print("Loaded! Ready for test!", flush=True)
    HE = tfreloader()
    m.inference(HE, dirr, Not_Realtest=False)

# cut tiles with coordinates in the name (exclude white)

if not os.path.isfile(img_dir+'/dict.csv'):
    n_x, n_y, raw_img, resx, resy = get_tilev2.tile(image_file = imgfile, outdir = img_dir)

else:
    n_x, n_y, raw_img, resx, resy = get_tilev2.sz(image_file = imgfile)

dict = pd.read_csv(img_dir+'/dict.csv', header=0)

print(len(dict["Num"]))

test(to_reload=modeltoload)

teresult = pd.read_csv(out_dir+'/Test.csv', header=0)

joined = pd.merge(dict, teresult, how='inner', on=['Num'])

joined.to_csv(out_dir+'/finaldict.csv', index=False)


# output heat map of pos and neg.
opt = np.full((n_x, n_y), 0)
hm_R = np.full((n_x, n_y), 0)
hm_G = np.full((n_x, n_y), 0)
hm_B = np.full((n_x, n_y), 0)

print(np.shape(opt))

poscsv = joined.loc[joined['Prediction'] == 1]
for index, row in poscsv.iterrows():
    opt[row["X_pos"], row["Y_pos"]] = 255
    hm_R[row["X_pos"], row["Y_pos"]] = 255
    hm_G[row["X_pos"], row["Y_pos"]] = int((1-(row["pos_score"]-0.5)*2)*255)
    hm_B[row["X_pos"], row["Y_pos"]] = int((1 - (row["pos_score"] - 0.5) * 2) * 255)

negcsv = joined.loc[joined['Prediction'] == 0]
for index, row in negcsv.iterrows():
    opt[row["X_pos"], row["Y_pos"]] = 255
    hm_B[row["X_pos"], row["Y_pos"]] = 255
    hm_G[row["X_pos"], row["Y_pos"]] = int((1-(row["neg_score"]-0.5)*2)*255)
    hm_R[row["X_pos"], row["Y_pos"]] = int((1 - (row["neg_score"] - 0.5) * 2) * 255)

opt = opt.repeat(5, axis=0).repeat(5, axis=1)
opt = mph.remove_small_objects(opt.astype(bool), min_size=500, connectivity=2).astype(np.uint8)

ori_img = cv2.resize(raw_img, (np.shape(opt)[0]+resx, np.shape(opt)[1]+resy))
ori_img = ori_img[:np.shape(opt)[1], :np.shape(opt)[0], :3]
tq = ori_img[:,:,0]
ori_img[:,:,0] = ori_img[:,:,2]
ori_img[:,:,2] = tq
cv2.imwrite(out_dir+'/Original_scaled.png', ori_img)

topt = np.transpose(opt)
opt = np.full((np.shape(topt)[0], np.shape(topt)[1], 3), 0)
opt[:,:,0] = topt
opt[:,:,1] = topt
opt[:,:,2] = topt
cv2.imwrite(out_dir+'/Mask.png', opt*255)

hm_R = np.transpose(hm_R)
hm_G = np.transpose(hm_G)
hm_B = np.transpose(hm_B)
hm_R = hm_R.repeat(5, axis=0).repeat(5, axis=1)
hm_G = hm_G.repeat(5, axis=0).repeat(5, axis=1)
hm_B = hm_B.repeat(5, axis=0).repeat(5, axis=1)
hm = np.dstack([hm_B, hm_G, hm_R])
hm = hm*opt
cv2.imwrite(out_dir+'/HM.png', hm)

ori_img = ori_img*opt
overlay = ori_img * 0.65 + hm * 0.35
cv2.imwrite(out_dir+'/Overlay.png', overlay)


# # Time measure tool
# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))





