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
import HE_data_input
import cnn


IMG_DIM = 299

INPUT_DIM = [IMG_DIM**2*3, # Default input for INCEPTION_V3 network, 299*299*3
                IMG_DIM,IMG_DIM]
               

HYPERPARAMS = {
    "batch_size": 1,
    "dropout": 0.8,
    "learning_rate": 1E-4
}

MAX_ITER = 100#2**16
MAX_EPOCHS = np.inf

LOG_DIR = "./log/0710"
METAGRAPH_DIR = "./out/0710"


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


def load_HE_data(train_dat_name,train_lab_name,valid_dat_name,valid_lab_name):
 
    train_dat=iter_loadtxt(train_dat_name,dtype=int,delimiter='\t')
    valid_dat=iter_loadtxt(valid_dat_name,dtype=int,delimiter='\t')
    train_lab=iter_loadtxt(train_lab_name,dtype=int,delimiter='\t')
    valid_lab=iter_loadtxt(valid_lab_name,dtype=int,delimiter='\t')
    
    class DataSets(object):
        pass
    
    data_sets=DataSets()
    
    data_sets.train=HE_data_input.DataSet(images=train_dat,
                                       labels=train_lab,
                                       reshape=False)
    
    data_sets.validation=HE_data_input.DataSet(images=valid_dat,
                                       labels=valid_lab,
                                       reshape=False)
    return data_sets

def main(to_reload=None):
    
    dat_f='/media/lwk/data/HE_Image/slides/slide80_mini_data.txt'
    
    lab_f='/media/lwk/data/HE_Image/slides/slide80_mini_lab.txt'
    
    
    HE = load_HE_data(train_dat_name=dat_f,
                      train_lab_name=lab_f,
                      valid_dat_name=dat_f,
                      valid_lab_name=lab_f)


    if to_reload: # restore
        
        m = cnn.INCEPTION(INPUT_DIM,HYPERPARAMS, meta_graph=to_reload)
        print("Loaded!",flush=True)
        
        x, y=HE.train.next_batch(1)
        print(m.inference(x))
        print(y)
        

    else: # train
        """to try cont'd training, load data from previously saved meta graph"""
        m = cnn.INCEPTION(INPUT_DIM,HYPERPARAMS, log_dir=LOG_DIR)
        m.train(HE, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS,
                verbose=True, save=True, outdir=METAGRAPH_DIR)
        print("Trained!",flush=True)
          
        x, y=HE.train.next_batch(1)
        print(m.inference(x))
        print(y)


    
if __name__ == "__main__":
    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    try:
        to_reload = sys.argv[1]
        main(to_reload=to_reload)
    except(IndexError):
        main()
