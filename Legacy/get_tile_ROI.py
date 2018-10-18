from openslide import OpenSlide
from scipy import signal
import numpy as np
import pandas as pd
import os

coords = pd.read_excel('../Neutrophil/all_features_circa_July.xlsx', header=0)
pos = coords.loc[(coords['Review'] == '+') & (coords['Slide'] == 'Slide80.scn')]
# print('total pos:')
# print(len(pos.index))

xmax = int(int(pos['X'].max())+299)
xmin = int(int(pos['X'].min())-299)
ymax = int(int(pos['Y'].max())+299)
ymin = int(int(pos['Y'].min())-299)
xrange = xmax-xmin
yrange = ymax-ymin

# print(xmax)
# print(xmin)
# print(ymax)
# print(ymin)

slide = OpenSlide("../Neutrophil/ImageCollection_0000026280_2016-10-27 14_13_01.scn")

assert 'openslide.bounds-height' in slide.properties
assert 'openslide.bounds-width' in slide.properties
assert 'openslide.bounds-x' in slide.properties
assert 'openslide.bounds-y' in slide.properties


xo = int(slide.properties['openslide.bounds-x'])
yo = int(slide.properties['openslide.bounds-y'])
bounds_height = int(slide.properties['openslide.bounds-height'])
bounds_width = int(slide.properties['openslide.bounds-width'])

# print(xo)
# print(yo)
# print(bounds_height)
# print(bounds_width)

x = xo
y = yo
# if x < xmin:
#     print('x true')
#     x = xmin
# if y < ymin:
#     print('y true')
#     y = ymin
if yrange < bounds_height:
    print('yr true')
    bounds_height = ymax
if xrange < bounds_width:
    print('xr true')
    bounds_width = xmax

# print(x)
# print(y)
# print(bounds_height)
# print(bounds_width)
# bounds_x = x + bounds_width
# bounds_y = y + bounds_height

half_width_region = 149
full_width_region = 2 * half_width_region + 1

n_x = int((bounds_width - 1) / half_width_region)
n_y = int((bounds_height - 1) / half_width_region)
# print(n_x)
# print(n_y)

x_edge = np.arange(n_x + 1) * 149
y_edge = np.arange(n_y + 1) * 149
print(x_edge)
print(y_edge)
lab, _, _ = np.histogram2d(x=np.asarray(pos['X']),
                           y=np.asarray(pos['Y']),
                           bins=[x_edge, y_edge])

lab_res = signal.convolve2d(lab, np.ones((2, 2)), mode='valid').astype(int)

dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')
tile_lab = []

if not os.path.exists('../Neutrophil/Tiles_ROI'):
        os.makedirs('../Neutrophil/Tiles_ROI')

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
            the_image.save("../Neutrophil/Tiles_ROI/region_x{}_y{}_{}.png".format(format(i, '02d'),
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
np.savetxt('../Neutrophil/slide80_ROI_data.txt', dat, fmt='%i', delimiter='\t')
np.savetxt('../Neutrophil/slide80_ROI_lab.txt', tile_lab, fmt='%i', delimiter='\t')

# the_image.save("region.png")