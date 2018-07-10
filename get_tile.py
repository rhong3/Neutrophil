from openslide import OpenSlide
from scipy import signal
import numpy as np
import pandas as pd

slide = OpenSlide("../Neutrophil/ImageCollection_0000026280_2016-10-27 14_13_01.scn")

assert 'openslide.bounds-height' in slide.properties
assert 'openslide.bounds-width' in slide.properties
assert 'openslide.bounds-x' in slide.properties
assert 'openslide.bounds-y' in slide.properties

bounds_height = int(slide.properties['openslide.bounds-height'])
bounds_width = int(slide.properties['openslide.bounds-width'])

# bounds_height = 3000
# bounds_width = 3000

bounds_x = int(slide.properties['openslide.bounds-x']) + 60000
bounds_y = int(slide.properties['openslide.bounds-y']) + 30000

# target_x = 17501
# target_y = 22703


target_x = bounds_width - 299
target_y = bounds_height - 299

half_width_region = 149
full_width_region = 2 * half_width_region + 1

n_x = int((bounds_width - 1) / half_width_region)
n_y = int((bounds_height - 1) / half_width_region)

coords = pd.read_table('../Neutrophil/all_users_reduced_curation_compressed.txt', delimiter='\t',
                       header=0)
pos = coords.loc[(coords['State'] == '+') & (coords['Slide'] == 'Slide80.scn')]

x_edge = np.arange(n_x + 1) * 149 + 60000
y_edge = np.arange(n_y + 1) * 149 + 30000

lab, _, _ = np.histogram2d(x=np.asarray(pos['X']),
                           y=np.asarray(pos['Y']),
                           bins=[x_edge, y_edge])

lab_res = signal.convolve2d(lab, np.ones((2, 2)), mode='valid').astype(int)

dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')
tile_lab = []

for i in range(n_x - 1):
    for j in range(n_y - 1):
        target_x = half_width_region * i
        target_y = half_width_region * j

        image_x = target_x + bounds_x
        image_y = target_y + bounds_y

        the_image = slide.read_region((image_x, image_y), 0, (full_width_region, full_width_region))

        the_image.save("../Neutrophil/mini_data_pic/region_x{}_y{}_{}.png".format(format(i, '02d'),
                                                                                                    format(j, '02d'),
                                                                                                    format(
                                                                                                        lab_res[i, j],
                                                                                                        '02d')))
        pix = np.array(the_image)[:, :, 0:3]
        dat = np.vstack([dat, pix.flatten()])
        tile_lab.append((lab_res[i, j] > 0))

tile_lab = np.asarray(tile_lab).astype(int)
np.savetxt('../Neutrophil/slide80_mini_data.txt', dat, fmt='%i', delimiter='\t')
np.savetxt('../Neutrophil/slide80_mini_lab.txt', tile_lab, fmt='%i', delimiter='\t')

# the_image.save("region.png")