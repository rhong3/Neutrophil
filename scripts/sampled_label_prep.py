"""
Prepare the label file for sampled tiles

Created on 05/05/2020

@author: RH
"""
import pandas as pd
import numpy as np

raw = pd.read_csv('../../sampled_label.csv', header=0)
label = []

for itr, row in raw.iterrows():
    if 'negative' in row['Label']:
        label.append([str(row['External ID'].split('_')[0]),
                      str('../../sampled_tiles/'+row['External ID']), 0])
    elif 'Neutrophil' in row['Label']:
        label.append([str(row['External ID'].split('_')[0]),
                      str('../../sampled_tiles/' + row['External ID']), 1])
    else:
        label.append([str(row['External ID'].split('_')[0]),
                      str('../../sampled_tiles/' + row['External ID']), np.nan])

labelpd = pd.DataFrame(label, columns=['slide', 'path', 'label'])
labelpd.to_csv('../../sampled_label_ready.csv', index=False, header=True)

labelpd = labelpd.fillna(0.5)
labelsld = labelpd.groupby(['slide']).mean()
labelsld.to_csv('../../sampled_label_slide.csv', index=True, header=True)

