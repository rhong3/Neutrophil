import pandas as pd
import numpy as np
from PIL import Image

dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')
tile_lab = []
totlist = pd.read_csv('../Neutrophil/Tiles_final/tot_sample.csv', header = 0)
f = 1
for index, row in totlist.iterrows():
    image = Image.open(row['path'])
    pix = np.array(image)[:, :, 0:3]
    dat = np.vstack([dat, pix.flatten()])
    tile_lab.append(1)
    if len(tile_lab) == 5000 or index == len(totlist['label'])-1:
        np.savetxt('../Neutrophil/Tiles_final/slide80_data_{}.txt'.format(f), dat, fmt='%i', delimiter='\t')
        np.savetxt('../Neutrophil/Tiles_final/slide80_lab_{}.txt'.format(f), tile_lab, fmt='%i', delimiter='\t')
        dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')
        tile_lab = []
        f+=1

