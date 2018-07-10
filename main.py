#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:15:18 2017

@author: lwk
"""

import os
import sys

import numpy as np
import tensorflow as tf

#import plot
import HE_data_input
import cnn


IMG_DIM = 299

ARCHITECTURE = [IMG_DIM**2*3, # 784 pixels
                IMG_DIM,IMG_DIM]
               # 5,1] # kernel size and stride
                # 50]
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 1,
    "dropout": 0.9,
    "learning_rate": 1E-4,
    "lambda_l2_reg": 1E-5
    #"nonlinearity": tf.nn.elu,
    #"squashing": tf.nn.sigmoid
}

MAX_ITER = 100#2**16
MAX_EPOCHS = np.inf

LOG_DIR = "./log/0621"
METAGRAPH_DIR = "./out/0621"
PLOTS_DIR = "./png/0621"

def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./mnist_data")

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

def load_HE(data_file,label_file=None):
    #dat=HE_data_input.get_data('26280_test.tif',32)
    dat=iter_loadtxt(data_file,dtype=int,delimiter='\t')
    
    if label_file:
        labels=np.loadtxt(label_file,dtype=int,delimiter='\t')        
        Dat=HE_data_input.DataSet(images=dat,
                                  labels=labels,
                                  reshape=False)
    else:
        Dat=HE_data_input.DataSet(images=dat,
                                  labels=np.random.binomial(1,0.2, size=dat.shape[0]),
                                  reshape=False)
    return Dat

def main(to_reload=None):
    
    #dat_f='/ifs/home/liuw09/HE_Image/26280_nuc/26280_all_nuc.txt'
    #dat_f='/home/lwk/VAE/nucleus/auto_label_nucim.txt'
    #dat_f='/home/lwk/HE_Image/26280_auto_labeled_cells_rgb.txt'
    #dat_f='/home/lwk/HE_Image/26280_auto_labeled_cells_rgb_100.txt'
    #dat_f='/ifs/home/liuw09/HE_Image/26280_cell/temp/26280_all_cells_rgb.txt'
    #dat_f='/ifs/home/liuw09/VAE/conv_vae/26280_auto_labeled_cells_rgb_100.txt'
    dat_f='/media/lwk/data/HE_Image/slides/slide80_mini_data.txt'
    
    lab_f='/media/lwk/data/HE_Image/slides/slide80_mini_lab.txt'
    #pretrained='/home/lwk/VAE/conv_vae/out/0816/170816_1737_conv_vae_3072_32_64_128_30-200000'
    pretrained=None
    
    HE = load_HE(data_file=dat_f,label_file=lab_f)
    print("Data file: "+dat_f,flush=True)
    print("Label file: "+lab_f,flush=True)
    print("Architecture:",flush=True)
    print(ARCHITECTURE,flush=True)
    print(HYPERPARAMS,flush=True)
    print("Pre-trained model:",flush=True)
    print(pretrained,flush=True)


    if to_reload: # restore
        #HE=load_HE('auto_label_nucim.txt')
        v = cnn.INCEPTION(ARCHITECTURE,HYPERPARAMS, meta_graph=to_reload)
        print("Loaded!",flush=True)
        x, y=HE.next_batch(100)
        print(v.inference(x))
        print(y)
        

    else: # train
        """to try cont'd training, load data from previously saved meta graph"""
        v = cnn.INCEPTION(ARCHITECTURE,HYPERPARAMS, log_dir=LOG_DIR,
                    pre_trained=pretrained
                    )
        v.train(HE, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS,
                verbose=True, save=True, outdir=METAGRAPH_DIR, plots_outdir=PLOTS_DIR)
        print("Trained!",flush=True)
       
    """    
    x, _ = HE.next_batch(10)
    #x=HE.images[0:11,:]
    x_reconstructed = v.vae(x)
    #np.savetxt('./x.txt',np.transpose(x),delimiter='\t')
    plot.plotSubset(v, x, x_reconstructed, n=10, name='train',
                        outdir=PLOTS_DIR)
    """    
    
    #z1=np.random.normal(size=20)
    #z2=np.random.normal(size=20)
    """
    x1=np.reshape(x[2,:],[1,784])
    x2=np.reshape(x[4,:],[1,784])
    z1=v.encode(x1)[0]
    z2=v.encode(x2)[0]
    z1=z1.T
    z2=z2.T
    plot.interpolate(v,z1,z2,n=10)
    """
    
if __name__ == "__main__":
    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR, PLOTS_DIR):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    try:
        to_reload = sys.argv[1]
        main(to_reload=to_reload)
    except(IndexError):
        main()
