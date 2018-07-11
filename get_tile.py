from openslide import OpenSlide
from scipy import signal
import numpy as np
import pandas as pd
import os

slide = OpenSlide("../Neutrophil/ImageCollection_0000026280_2016-10-27 14_13_01.scn")

assert 'openslide.bounds-height' in slide.properties
assert 'openslide.bounds-width' in slide.properties
assert 'openslide.bounds-x' in slide.properties
assert 'openslide.bounds-y' in slide.properties

x = int(slide.properties['openslide.bounds-x'])
y = int(slide.properties['openslide.bounds-y'])
bounds_height = int(slide.properties['openslide.bounds-height'])
bounds_width = int(slide.properties['openslide.bounds-width'])

bounds_x = x + bounds_width
bounds_y = y + bounds_height

half_width_region = 149
full_width_region = 2 * half_width_region + 1

n_x = int((bounds_width - 1) / half_width_region)
n_y = int((bounds_height - 1) / half_width_region)

coords = pd.read_table('../Neutrophil/all_users_reduced_curation_compressed.txt', delimiter='\t',
                       header=0)
pos = coords.loc[(coords['State'] == '+') & (coords['Slide'] == 'Slide80.scn')]

x_edge = np.arange(n_x + 1) * 149
y_edge = np.arange(n_y + 1) * 149

lab, _, _ = np.histogram2d(x=np.asarray(pos['X']),
                           y=np.asarray(pos['Y']),
                           bins=[x_edge, y_edge])

lab_res = signal.convolve2d(lab, np.ones((2, 2)), mode='valid').astype(int)

dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')
tile_lab = []

if not os.path.exists('../Neutrophil/Tiles'):
        os.makedirs('../Neutrophil/Tiles')

for i in range(n_x - 1):
    for j in range(n_y - 1):
        target_x = half_width_region * i
        target_y = half_width_region * j

        image_x = target_x + x
        image_y = target_y + y

        the_image = slide.read_region((image_x, image_y), 0, (full_width_region, full_width_region))
        the_imagea = np.array(the_image)[:,:,:3]

        mask = (the_imagea[:,:,:3] > 200).astype(np.uint8)
        mask = mask[:,:,0]*mask[:,:,1]*mask[:,:,2]
        white = np.sum(mask)/(299*299)
        # print(white)

        if white < 0.5:
            the_image.save("../Neutrophil/Tiles/region_x{}_y{}_{}.png".format(format(i, '02d'),
                                                                                                    format(j, '02d'),
                                                                                                    format(
                                                                                                        lab_res[i, j],
                                                                                                        '02d')))
            pix = np.array(the_image)[:, :, 0:3]
            dat = np.vstack([dat, pix.flatten()])
            tile_lab.append((lab_res[i, j] > 0))
        else:
            print('Ignore white!')

tile_lab = np.asarray(tile_lab).astype(int)
np.savetxt('../Neutrophil/slide80_mini_data.txt', dat, fmt='%i', delimiter='\t')
np.savetxt('../Neutrophil/slide80_mini_lab.txt', tile_lab, fmt='%i', delimiter='\t')

# the_image.save("region.png")