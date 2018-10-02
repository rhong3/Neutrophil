from openslide import OpenSlide
import numpy as np
import pandas as pd
import os

def tile(path_to_slide = "../Neutrophil/", image_file = "ImageCollection_0000026280_2016-10-27 14_13_01.scn"):
    slide = OpenSlide(path_to_slide+image_file)

    assert 'openslide.bounds-height' in slide.properties
    assert 'openslide.bounds-width' in slide.properties
    assert 'openslide.bounds-x' in slide.properties
    assert 'openslide.bounds-y' in slide.properties

    x = int(slide.properties['openslide.bounds-x'])
    y = int(slide.properties['openslide.bounds-y'])
    bounds_height = int(slide.properties['openslide.bounds-height'])
    bounds_width = int(slide.properties['openslide.bounds-width'])

    half_width_region = 49
    full_width_region = 299
    stepsize = full_width_region - half_width_region

    n_x = int((bounds_width - 1) / stepsize)
    n_y = int((bounds_height - 1) / stepsize)

    dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')

    imloc = []
    counter = 0
    svcounter = 0

    if not os.path.exists(path_to_slide + 'Tiles'):
            os.makedirs(path_to_slide + 'Tiles')

    if not os.path.exists(path_to_slide + 'Outputs'):
            os.makedirs(path_to_slide + 'Outputs')

    for i in range(n_x - 1):
        for j in range(n_y - 1):
            target_x = stepsize * i
            target_y = stepsize * j

            image_x = target_x + x
            image_y = target_y + y

            the_image = slide.read_region((image_x, image_y), 0, (full_width_region, full_width_region))
            the_imagea = np.array(the_image)[:,:,:3]

            mask = (the_imagea[:,:,:3] > 200).astype(np.uint8)
            mask = mask[:,:,0]*mask[:,:,1]*mask[:,:,2]
            white = np.sum(mask)/(299*299)

            if white < 0.5:
                the_image.save(path_to_slide+"Tiles/region_x-{}-y-{}.png".format(target_x, target_y))
                imloc.append([svcounter, counter, target_x, target_y])
                if svcounter % 5000 == 0:
                    np.savetxt(path_to_slide+'outputs/data-{}.txt'.format(svcounter), dat, fmt='%i', delimiter='\t')
                    dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')
                pix = np.array(the_image)[:, :, 0:3]
                dat = np.vstack([dat, pix.flatten()])
                svcounter += 1
            else:
                print('Ignore white!')

            counter += 1

    np.savetxt(path_to_slide+'outputs/data-{}.txt'.format(svcounter), dat, fmt='%i', delimiter='\t')
    dat = np.empty((0, int(299 ** 2 * 3)), dtype='uint8')
    imlocpd = pd.DataFrame(imloc, columns = ["Num", "Count", "X", "Y"])
    imlocpd.to_csv(path_to_slide+"outputs/dict.csv", index = False)
