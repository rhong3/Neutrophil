#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:06:47 2017

@author: lwk
"""

import os
import sys

import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt


#import plot
import HE_data_input
import cnn




LOG_DIR = "./log/0815"
METAGRAPH_DIR = "./out/0815"
PLOTS_DIR = "./png/0815"

IMG_DIM=32

ARCHITECTURE = [IMG_DIM**2*3, # 784 pixels
                32, 64,128,# number of convolutional filters in each layer
                5,1] # kernel size and stride
                # 50]
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 128,
    "dropout": 0.9,
    "learning_rate": 5E-4
    #"lambda_l2_reg": 1E-5,
    #"nonlinearity": tf.nn.elu,
    #"squashing": tf.nn.sigmoid
}


def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./mnist_data")

def load_HE(data_file,label_file=None):
    #dat=HE_data_input.get_data('26280_test.tif',32)
    dat=np.loadtxt(data_file,dtype='uint8',delimiter='\t')
    
    if label_file:
        labels=np.loadtxt(label_file,dtype='uint8',delimiter='\t')        
        Dat=HE_data_input.DataSet(images=dat,
                                  labels=labels,
                                  reshape=False)
    else:
        Dat=HE_data_input.DataSet(images=dat,
                                  labels=np.random.binomial(1,0.2, size=dat.shape[0]),
                                  reshape=False)
    return Dat


dat_f='/media/lwk/lab/HE_Image/slides/Slide76.scn_labeled_data.txt'
lab_f='/media/lwk/lab/HE_Image/slides/Slide76.scn_labels.txt'

HE = load_HE(data_file=dat_f,label_file=lab_f)
print("Data file: "+dat_f,flush=True)
print("Label file: "+lab_f,flush=True)

naive='/home/lwk/VAE/cnn/out/0817/naive/170817_1323_cnn_3072_32_64_128_5_1-5000'
pretrained='/home/lwk/VAE/cnn/out/0817/pretrained/170817_1247_cnn_3072_32_64_128_5_1-5000'
m1 = cnn.CNN(ARCHITECTURE,HYPERPARAMS, meta_graph=naive)
tf.reset_default_graph()
m2 = cnn.CNN(ARCHITECTURE,HYPERPARAMS, meta_graph=pretrained)
print("Loaded!",flush=True)

print('Test sample size: {}'.format(HE.num_examples))

def test_performance(model,test_data,batch_size=100):
    assert test_data.num_examples>batch_size, "Sample size is smaller than batch size!"
    n_batch=math.floor(test_data.num_examples/batch_size)
    print('Number of test samples: {}'.format(n_batch*batch_size))
    y_true=np.empty((0,1),dtype='int8')
    prob=np.empty((0,2),dtype='float32')
    for i in range(n_batch):
        x, y= test_data.next_batch(batch_size)
        pred=model.inference(x)
        y_true=np.append(y_true,y)
        prob=np.vstack((prob,pred))
        
    precision, recall, _ = precision_recall_curve(y_true, prob[:,1])
    fpr, tpr, _ = roc_curve(y_true, prob[:,1])
    
    return y_true, prob, precision, recall, fpr, tpr




y1, pred1, precision1, recall1, fpr1, tpr1 = test_performance(m1, HE)
y2, pred2, precision2, recall2, fpr2, tpr2 = test_performance(m2, HE)

ave_precision1 = average_precision_score(y1, pred1[:,1])
ave_precision2 = average_precision_score(y2, pred2[:,1])

print(ave_precision1)
print(ave_precision2)

#print(auc(fpr1,tpr1))
#print(auc(fpr2,tpr2))

#plt.step(recall1, precision1, color='b', alpha=0.8,where='post')
#plt.fill_between(recall, precision, step='post', alpha=0.2,
                 #color='b')


plt.plot(recall1, precision1, alpha=0.5,
         label='naive (AP = %0.2f)' % (ave_precision1))
plt.plot(recall2, precision2, alpha=0.5,
         label='pre-trained (AP = %0.2f)' % (ave_precision2))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc="lower right")
#plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(ave_precision1))
plt.title('Precision-Recall Curve')
plt.savefig('precision_recall2.png')
plt.close()


plt.plot(fpr1, tpr1, lw=1, alpha=0.5,label='naive (ROC AUC = %0.2f)' % (auc(fpr1,tpr1)))
plt.plot(fpr2, tpr2, lw=1, alpha=0.5,label='pre-trained (ROC AUC = %0.2f)' % (auc(fpr2,tpr2)))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc="lower right")
plt.title("Receiver Operating Characteristic Curve")
plt.savefig('ROC_curve.png')

