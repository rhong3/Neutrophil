from openslide import OpenSlide
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


if not os.path.exists('../Neutrophil/Tiles_final'):
        os.makedirs('../Neutrophil/Tiles_final')

if not os.path.exists('../Neutrophil/Tiles_final/pos'):
        os.makedirs('../Neutrophil/Tiles_final/pos')

if not os.path.exists('../Neutrophil/Tiles_final/neg'):
        os.makedirs('../Neutrophil/Tiles_final/neg')

coords = pd.read_excel('../Neutrophil/all_features_circa_July.xlsx', header=0)
sample = coords.loc[(coords['Slide'] == 'Slide80.scn')]
sample = sample.loc[(coords['Review'] == '+') | (coords['Review'] == '-')]
sample = sample.sample(frac=1).reset_index(drop=True)
pos = coords.loc[(coords['Review'] == '+') & (coords['Slide'] == 'Slide80.scn')]

sample.to_csv('../Neutrophil/Tiles_final/sample.csv', header = 0, index = False)
pos.to_csv('../Neutrophil/Tiles_final/pos.csv', header = 0, index = False)

slide = OpenSlide("../Neutrophil/ImageCollection_0000026280_2016-10-27 14_13_01.scn")

assert 'openslide.bounds-height' in slide.properties
assert 'openslide.bounds-width' in slide.properties
assert 'openslide.bounds-x' in slide.properties
assert 'openslide.bounds-y' in slide.properties

xo = int(slide.properties['openslide.bounds-x'])
yo = int(slide.properties['openslide.bounds-y'])
bounds_height = int(slide.properties['openslide.bounds-height'])
bounds_width = int(slide.properties['openslide.bounds-width'])

sample.loc[:, 'X'] = sample.loc[:, 'X'] + xo
sample.loc[:, 'Y'] = sample.loc[:, 'Y'] + yo
pos.loc[:, 'X'] = pos.loc[:, 'X'] + xo
pos.loc[:, 'Y'] = pos.loc[:, 'Y'] + yo

sample.to_csv('../Neutrophil/Tiles_final/sample_realloc.csv', header = 0, index = False)
pos.to_csv('../Neutrophil/Tiles_final/pos_realloc.csv', header = 0, index = False)

half_width_region = 149
full_width_region = 2 * half_width_region + 1
dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')
tile_lab = []

for index, row in sample.iterrows():
    print(index)
    if row['Review'] == '+':
        the_image = slide.read_region((row['X'] - 149, row['Y'] - 149), 0, (full_width_region, full_width_region))
        the_image.save("../Neutrophil/Tiles_final/pos/region_x{}_y{}_0_rot0.png".format(format(row['X'] - xo),
                                                                                         format(row['Y'] - yo)))
        pix = np.array(the_image)[:, :, 0:3]
        dat = np.vstack([dat, pix.flatten()])
        tile_lab.append(1)
        for i in range(1,5):
            intx = np.random.randint(low=50, high=250, size=1)[0]
            inty = np.random.randint(low=50, high=250, size=1)[0]
            the_image = slide.read_region((row['X']-intx, row['Y']-inty), 0, (full_width_region, full_width_region))
            the_image.save("../Neutrophil/Tiles_final/pos/region_x{}_y{}_{}_rot0.png".format(format(row['X']-xo), format(row['Y']-yo), format(i)))
            pix = np.array(the_image)[:, :, 0:3]
            dat = np.vstack([dat, pix.flatten()])
            tile_lab.append(1)
            for rot in range(90,355,90):
                the_image_rot = the_image.rotate(rot)
                the_image_rot.save("../Neutrophil/Tiles_final/pos/region_x{}_y{}_{}_rot{}.png".format(format(row['X']-xo), format(row['Y']-yo), format(i), format(rot)))
                pix = np.array(the_image)[:, :, 0:3]
                dat = np.vstack([dat, pix.flatten()])
                tile_lab.append(1)
    elif row['Review'] == '-':
        checked = ckpt(row['X'], row['Y'], pos)
        if checked:
            the_image = slide.read_region((row['X'] - 149, row['Y'] - 149), 0, (full_width_region, full_width_region))
            the_image.save(
                "../Neutrophil/Tiles_final/neg/region_x{}_y{}_rot0.png".format(format(row['X'] - xo),
                                                                         format(row['Y'] - yo)))
            pix = np.array(the_image)[:, :, 0:3]
            dat = np.vstack([dat, pix.flatten()])
            tile_lab.append(0)
            for rot in range(90, 355, 90):
                the_image_rot = the_image.rotate(rot)
                the_image_rot.save(
                    "../Neutrophil/Tiles_final/neg/region_x{}_y{}_rot{}.png".format(format(row['X'] - xo),
                                                                                    format(row['Y'] - yo), format(rot)))
                pix = np.array(the_image)[:, :, 0:3]
                dat = np.vstack([dat, pix.flatten()])
                tile_lab.append(0)
np.savetxt('../Neutrophil/Tiles_final/slide80_data.txt', dat, fmt='%i', delimiter='\t')
np.savetxt('../Neutrophil/Tiles_final/slide80_lab.txt', tile_lab, fmt='%i',delimiter='\t')