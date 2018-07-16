import os
import pandas as pd
import sklearn.utils as sku


pos_path = '../Neutrophil/Tiles_final/pos'
neg_path = '../Neutrophil/Tiles_final/neg'
pos_pattern = '../Neutrophil/Tiles_final/pos/{}'
neg_pattern = '../Neutrophil/Tiles_final/neg/{}'


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
    neglist = image_ids_in(neg_path)
    totpd = []
    pospd = []
    negpd = []
    for i in poslist:
        posdir = pos_pattern.format(i)
        pdp = [posdir, 1]
        pospd.append(pdp)
        totpd.append(pdp)
    for j in neglist:
        negdir = neg_pattern.format(j)
        pdn = [negdir, 0]
        negpd.append(pdn)
        totpd.append(pdn)
    totpd = pd.DataFrame(totpd, columns = ['path', 'label'])
    pospd = pd.DataFrame(pospd, columns = ['path', 'label'])
    negpd = pd.DataFrame(negpd, columns = ['path', 'label'])
    totpd = sku.shuffle(totpd)
    pospd = sku.shuffle(pospd)
    negpd = sku.shuffle(negpd)

    return totpd, pospd, negpd


tot, pos, neg = samplesum()

tot.to_csv('../Neutrophil/Tiles_final/tot_sample.csv', index = False)
pos.to_csv('../Neutrophil/Tiles_final/pos_sample.csv', index = False)
neg.to_csv('../Neutrophil/Tiles_final/neg_sample.csv', index = False)