"""
Prepare the label file for sampled tiles

Created on 05/05/2020

@author: RH
"""
import pandas as pd
import numpy as np
import shutil
import os

# raw = pd.read_csv('../../sampled_label.csv', header=0)
# label = []
#
# for itr, row in raw.iterrows():
#     if 'negative' in row['Label']:
#         label.append([str(row['External ID'].split('_')[0]),
#                       str('../../sampled_tiles/'+row['External ID']), 0])
#     elif 'Neutrophil' in row['Label']:
#         label.append([str(row['External ID'].split('_')[0]),
#                       str('../../sampled_tiles/' + row['External ID']), 1])
#     else:
#         label.append([str(row['External ID'].split('_')[0]),
#                       str('../../sampled_tiles/' + row['External ID']), np.nan])
#
# labelpd = pd.DataFrame(label, columns=['slide', 'path', 'label'])
# labelpd.to_csv('../../sampled_label_ready.csv', index=False, header=True)
#
# labelpd = labelpd.fillna(0.5)
# labelsld = labelpd.groupby(['slide']).mean()
# labelsld.to_csv('../../sampled_label_slide.csv', index=True, header=True)

labelpd = pd.read_csv('../../sampled_label_ready.csv', header=0)

todolist = ['76_Tiles_final', '79_Tiles_final', '80_Tiles_final']

extlist = []

for sld in todolist:
    for img in os.listdir('../../{}/pos'.format(sld)):
        if '_0_rot0' in img:
            ss = sld.split('_')[0]
            imgg = img.split('_0_rot')[0]
            shutil.copy('../../{}/pos/{}'.format(sld, img), '../../sampled_tiles/00000262{}_{}.png'.format(ss, imgg))
            extlist.append([str('00000262'+ss), '../../sampled_tiles/00000262{}_{}.png'.format(ss, imgg), 1])

    for img in os.listdir('../../{}/neg'.format(sld)):
        if 'rot0' in img:
            ss = sld.split('_')[0]
            imgg = img.split('rot')[0]
            shutil.copy('../../{}/neg/{}'.format(sld, img), '../../sampled_tiles/00000262{}_{}.png'.format(ss, imgg))
            extlist.append([str('00000262'+ss), '../../sampled_tiles/00000262{}_{}.png'.format(ss, imgg), 0])


labelpdex = pd.DataFrame(extlist, columns=['slide', 'path', 'label'])
output = pd.concat([labelpd, labelpdex], axis=0)
# output.to_csv('../../sampled_label_ready.csv', index=False, header=True)

labelpd = output.fillna(0.5)
labelsld = labelpd.groupby(['slide']).mean()
labelsld.to_csv('../../sampled_label_slide.csv', index=True, header=True)
