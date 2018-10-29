from openslide import OpenSlide
import numpy as np
import pandas as pd
import os


def tile(image_file, outdir, path_to_slide = "../Neutrophil/"):
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

    residue_x = int((bounds_width - n_x * stepsize)/50)
    residue_y = int((bounds_height - n_y * stepsize)/50)
    lowres = slide.read_region((x, y), 2, (int(n_x*stepsize/16), int(n_y*stepsize/16)))
    lowres = np.array(lowres)[:,:,:3]

    imloc = []
    counter = 0
    svcounter = 0

    if not os.path.exists(outdir):
            os.makedirs(outdir)

    for i in range(n_x - 1):
        for j in range(n_y - 1):
            target_x = stepsize * i
            target_y = stepsize * j

            image_x = target_x + x
            image_y = target_y + y

            the_image = slide.read_region((image_x, image_y), 0, (full_width_region, full_width_region))
            the_imagea = np.array(the_image)[:,:,:3]
            the_imagea = np.nan_to_num(the_imagea)
            mask = (the_imagea[:,:,:3] > 200).astype(np.uint8)
            maskb = (the_imagea[:,:,:3] < 5).astype(np.uint8)
            mask = mask[:,:,0]*mask[:,:,1]*mask[:,:,2]
            maskb = maskb[:, :, 0] * maskb[:, :, 1] * maskb[:, :, 2]
            white = (np.sum(mask)+np.sum(maskb))/(299*299)

            if white < 0.5:
                the_image.save(outdir + "/region_x-{}-y-{}.png".format(target_x, target_y))
                strr = outdir + "/region_x-{}-y-{}.png".format(target_x, target_y)
                imloc.append([svcounter, counter, target_x, target_y, i, j, strr])
                svcounter += 1
            else:
                pass
                # print('Ignore white!')

            counter += 1

    imlocpd = pd.DataFrame(imloc, columns = ["Num", "Count", "X", "Y", "X_pos", "Y_pos", "Loc"])
    imlocpd.to_csv(outdir + "/dict.csv", index = False)

    return n_x, n_y, lowres, residue_x, residue_y


def sz(image_file, path_to_slide = "../Neutrophil/"):
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

    residue_x = int((bounds_width - n_x * stepsize)/50)
    residue_y = int((bounds_height - n_y * stepsize)/50)
    lowres = slide.read_region((x, y), 2, (int(n_x * stepsize/16), int(n_y * stepsize/16)))
    lowres = np.array(lowres)[:, :, :3]

    return n_x, n_y, lowres, residue_x, residue_y

