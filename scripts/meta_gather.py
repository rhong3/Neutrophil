"""
Gathering meta results

Created on 05/08/2020

@author: RH
"""

import pandas as pd
import os

metapd = pd.DataFrame(columns=['model', 'state', 'best epoch', 'test loss', 'TP_tile',
                                                     'TN_tile', 'FN_tile', 'FP_tile', 'precision_tile',
                                                     'recall_tile', 'F1_tile', 'accuracy_tile', 'AUROC_tile',
                                                     'AUPRC_tile',	'TP_slide', 'TN_slide', 'FN_slide',
                                                     'FP_slide', 'precision_slide',	'recall_slide', 'F1_slide',
                                                     'accuracy_slide'])
for dir in os.listdir('../Results'):
    try:
        meta = pd.read_csv('../Results/{}/out/meta.csv'.format(dir), header=0)
        metapd = pd.concat([metapd, meta])
    except FileNotFoundError:
        pass

metapd = metapd.sort(['AUROC_tile', 'AUPRC_tile'], ascending=[0, 0])
metapd.to_csv('../Results/meta_summary.csv', index=False)

