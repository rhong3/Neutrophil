"""
Prepare sample split

Created on 05/07/2020

@author: RH

"""
import pandas as pd
import numpy as np
import os


def set_sep(path, cut=0.3):
    trlist = []
    telist = []
    valist = []
    dic = pd.read_csv('../../sampled_label_ready.csv', header=0)
    unq = list(dic.slide.unique())
    validation = unq[:int(len(unq) * cut / 2)]
    valist.append(dic[dic['slide'].isin(validation)])
    test = unq[int(len(unq) * cut / 2):int(len(unq) * cut)]
    telist.append(dic[dic['slide'].isin(test)])
    train = unq[int(len(unq) * cut):]
    trlist.append(dic[dic['slide'].isin(train)])

    test = pd.concat(telist)
    train = pd.concat(trlist)
    validation = pd.concat(valist)

    tepd = pd.DataFrame(test, columns=['slide', 'path', 'label'])
    tepd = tepd[['slide', 'path', 'label']]
    vapd = pd.DataFrame(validation.sample(frac=1), columns=['slide', 'path', 'label'])
    vapd = vapd[['slide', 'path', 'label']]
    trpd = pd.DataFrame(train.sample(frac=1), columns=['slide', 'path', 'label'])
    trpd = trpd[['slide', 'path', 'label']]

    tepd.to_csv(path + '/te_sample.csv', header=True, index=False)
    trpd.to_csv(path + '/tr_sample.csv', header=True, index=False)
    vapd.to_csv(path + '/va_sample.csv', header=True, index=False)