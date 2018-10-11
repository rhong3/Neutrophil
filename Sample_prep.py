'''
Prepare pd from image tiles
RH 0717
'''

import os
import pandas as pd
import sklearn.utils as sku
import numpy as np


pos_path = '../Neutrophil/All_Tiles_final/pos'
neg_path = '../Neutrophil/All_Tiles_final/neg'
pos_pattern = '../Neutrophil/All_Tiles_final/pos/{}'
neg_pattern = '../Neutrophil/All_Tiles_final/neg/{}'


def image_ids_in(root_dir, ignore=['.DS_Store']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


def samplesum():
    poslist = image_ids_in(pos_path)
    poslist = sorted(poslist)
    neglist = image_ids_in(neg_path)
    neglist = sorted(neglist)
    postenum = int(len(poslist)*0.1)
    negtenum = int(len(neglist)*0.1)
    totpd = []
    pospd = []
    negpd = []
    telist = []
    trlist = []
    postemplist = []
    negtemplist = []

    for i in poslist:
        posdir = pos_pattern.format(i)
        pdp = [posdir, 1]
        pospd.append(pdp)
        totpd.append(pdp)
        postemplist.append(pdp)
        if len(postemplist) == 17:
            if len(telist) < postenum:
                s = np.random.random_sample()
                if s > 0.85:
                    telist.extend(postemplist)
                else:
                    trlist.extend(postemplist)
            else:
                trlist.extend(postemplist)
            postemplist = []

    for j in neglist:
        negdir = neg_pattern.format(j)
        pdn = [negdir, 0]
        negpd.append(pdn)
        totpd.append(pdn)
        negtemplist.append(pdn)
        if len(negtemplist) == 4:
            if len(telist) < negtenum+postenum:
                s = np.random.random_sample()
                if s > 0.85:
                    telist.extend(negtemplist)
                else:
                    trlist.extend(negtemplist)
            else:
                trlist.extend(negtemplist)
            negtemplist = []

    totpd = pd.DataFrame(totpd, columns = ['path', 'label'])
    pospd = pd.DataFrame(pospd, columns = ['path', 'label'])
    negpd = pd.DataFrame(negpd, columns = ['path', 'label'])
    tepd = pd.DataFrame(telist, columns = ['path', 'label'])
    trpd = pd.DataFrame(trlist, columns=['path', 'label'])
    totpd = sku.shuffle(totpd)
    pospd = sku.shuffle(pospd)
    negpd = sku.shuffle(negpd)
    tepd = sku.shuffle(tepd)
    trpd = sku.shuffle(trpd)

    return totpd, pospd, negpd, tepd, trpd


tot, pos, neg, te, tr = samplesum()

tot.to_csv('../Neutrophil/All_Tiles_final/tot_sample.csv', index = False)
pos.to_csv('../Neutrophil/All_Tiles_final/pos_sample.csv', index = False)
neg.to_csv('../Neutrophil/All_Tiles_final/neg_sample.csv', index = False)
tr.to_csv('../Neutrophil/All_Tiles_final/tr_sample.csv', index = False)
te.to_csv('../Neutrophil/All_Tiles_final/te_sample.csv', index = False)