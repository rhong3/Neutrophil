from openslide import OpenSlide
from scipy import signal
import numpy as np
import pandas as pd
import os


def ckpt(X,Y,POS):
    check = True
    for indexp, rowp in POS.iterrows():
        if X in range(rowp['X']-151, rowp['X']+151) and Y in range(rowp['Y']-151, rowp['Y']+151):
            check = False
            break
    return check


if not os.path.exists('../Neutrophil/Tiles_grow'):
        os.makedirs('../Neutrophil/Tiles_grow')

if not os.path.exists('../Neutrophil/Tiles_grow/pos'):
        os.makedirs('../Neutrophil/Tiles_grow/pos')

if not os.path.exists('../Neutrophil/Tiles_grow/neg'):
        os.makedirs('../Neutrophil/Tiles_grow/neg')

coords = pd.read_excel('../Neutrophil/all_features_circa_July.xlsx', header=0)
pos = coords.loc[(coords['Review'] == '+') & (coords['Slide'] == 'Slide80.scn')]
neg = coords.loc[(coords['Review'] == '-') & (coords['Slide'] == 'Slide80.scn')]
pos.to_csv('../Neutrophil/Tiles_grow/pos.csv', header = 0, index = False)
neg.to_csv('../Neutrophil/Tiles_grow/neg.csv', header = 0, index = False)

slide = OpenSlide("../Neutrophil/ImageCollection_0000026280_2016-10-27 14_13_01.scn")

assert 'openslide.bounds-height' in slide.properties
assert 'openslide.bounds-width' in slide.properties
assert 'openslide.bounds-x' in slide.properties
assert 'openslide.bounds-y' in slide.properties

xo = int(slide.properties['openslide.bounds-x'])
yo = int(slide.properties['openslide.bounds-y'])
bounds_height = int(slide.properties['openslide.bounds-height'])
bounds_width = int(slide.properties['openslide.bounds-width'])

pos.loc[:, 'X'] = pos.loc[:, 'X'] + xo
pos.loc[:, 'Y'] = pos.loc[:, 'Y'] + yo
neg.loc[:, 'X'] = neg.loc[:, 'X'] + xo
neg.loc[:, 'Y'] = neg.loc[:, 'Y'] + yo

pos.to_csv('../Neutrophil/Tiles_grow/pos_realloc.csv', header = 0, index = False)
neg.to_csv('../Neutrophil/Tiles_grow/neg_realloc.csv', header = 0, index = False)

half_width_region = 149
full_width_region = 2 * half_width_region + 1
dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')

for index, row in pos.iterrows():
    the_image = slide.read_region((row['X']-149, row['Y']-149), 0, (full_width_region, full_width_region))
    the_image.save("../Neutrophil/Tiles_grow/pos/region_x{}_y{}.png".format(format(row['X']-xo), format(row['Y']-yo)))
    pix = np.array(the_image)[:, :, 0:3]
    dat = np.vstack([dat, pix.flatten()])
    for rot in range(10,175,10):
        the_image_rot = the_image.rotate(rot)
        the_image_rot.save("../Neutrophil/Tiles_grow/pos/region_x{}_y{}_rot{}.png".format(format(row['X']-xo), format(row['Y']-yo), format(rot)))
        pix = np.array(the_image)[:, :, 0:3]
        dat = np.vstack([dat, pix.flatten()])
np.savetxt('../Neutrophil/Tiles_grow/slide80_pos_data.txt', dat, fmt='%i', delimiter='\t')


dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')
for indexn, rown in neg.iterrows():
    checked = ckpt(rown['X'], rown['Y'], pos)
    print(checked)
    if checked:
        the_image = slide.read_region((rown['X'] - 149, rown['Y'] - 149), 0, (full_width_region, full_width_region))
        the_image.save(
            "../Neutrophil/Tiles_grow/neg/region_x{}_y{}.png".format(format(rown['X'] - xo), format(rown['Y'] - yo)))
        pix = np.array(the_image)[:, :, 0:3]
        dat = np.vstack([dat, pix.flatten()])
np.savetxt('../Neutrophil/Tiles_grow/slide80_neg_data.txt', dat, fmt='%i', delimiter='\t')

