"""
Tile real scn/svs files; used by Cutter.py

Created on 11/19/2018

@author: RH
"""
from openslide import OpenSlide
import numpy as np
import pandas as pd
import multiprocessing as mp
import staintools
from PIL import Image


# check if a tile is background or not; return a blank pixel percentage score
def bgcheck(img, ts):
    the_imagea = np.array(img)[:, :, :3]
    the_imagea = np.nan_to_num(the_imagea)
    mask = (the_imagea[:, :, :3] > 200).astype(np.uint8)
    maskb = (the_imagea[:, :, :3] < 50).astype(np.uint8)
    mask = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]
    maskb = maskb[:, :, 0] * maskb[:, :, 1] * maskb[:, :, 2]
    white = (np.sum(mask) + np.sum(maskb)) / (ts * ts)

    return white


# Tile color normalization
def normalization(img, sttd):
    img = np.array(img)[:, :, :3]
    img = staintools.LuminosityStandardizer.standardize(img)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(sttd)
    img = normalizer.transform(img)
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img


# tile method; slp is the scn/svs image; n_y is the number of tiles can be cut on y column to be cut;
# x and y are the upper left position of each tile; tile_size is tile size; stepsize of each step; x0 is the row to cut.
# outdir is the output directory for images;
# imloc record each tile's relative and absolute coordinates; imlist is a list of cut tiles.
def v_slide(slp, n_y, x, y, tile_size, stepsize, x0, outdir, std):
    # pid = os.getpid()
    # print('{}: start working'.format(pid))
    slide = OpenSlide(slp)
    imloc = []
    imlist = []
    y0 = 0
    target_x = x0 * stepsize
    image_x = target_x + x
    while y0 < n_y:
        target_y = y0 * stepsize
        image_y = target_y + y
        img = slide.read_region((image_x, image_y), 0, (tile_size, tile_size))
        wscore = bgcheck(img, tile_size)
        if wscore < 0.3:
            img = img.resize((tile_size, tile_size))
            # try:
            #     img = normalization(img, std)
            # except staintools.miscellaneous.exceptions.TissueMaskException:
            #     print("Empty tissue mask computed: region_x-{}-y-{}".format(image_x, image_y))
            #     y0 += 1
            #     continue
            # except:
            #     print('An error occurred: region_x-{}-y-{}'.format(image_x, image_y))
            #     y0 += 1
            #     continue
            img.save(outdir + "/region_x-{}-y-{}.png".format(image_x, image_y))
            strr = outdir + "/region_x-{}-y-{}.png".format(image_x, image_y)
            imloc.append([x0, y0, image_x, image_y, target_x, target_y, strr])
            # imlist.append(np.array(img)[:, :, :3])
        y0 += 1
    slide.close()
    return imloc, imlist


# image_file is the scn/svs name; outdir is the output directory; path_to_slide is where the scn/svs stored.
# First open the slide, determine how many tiles can be cut, record the residue edges width,
# and calculate the final output prediction heat map size should be. Then, using multithread to cut tiles, and stack up
# tiles and their position dictionaries.
def tile(image_file, outdir, std_img, stepsize, full_width_region, path_to_slide="../images/"):
    slide = OpenSlide(path_to_slide+image_file)
    slp = str(path_to_slide+image_file)

    assert 'openslide.bounds-height' in slide.properties
    assert 'openslide.bounds-width' in slide.properties
    assert 'openslide.bounds-x' in slide.properties
    assert 'openslide.bounds-y' in slide.properties

    x = int(slide.properties['openslide.bounds-x'])
    y = int(slide.properties['openslide.bounds-y'])
    bounds_height = int(slide.properties['openslide.bounds-height'])
    bounds_width = int(slide.properties['openslide.bounds-width'])

    n_x = int((bounds_width - 1) / stepsize)
    n_y = int((bounds_height - 1) / stepsize)

    residue_x = int((bounds_width - n_x * stepsize)/50)
    residue_y = int((bounds_height - n_y * stepsize)/50)
    lowres = slide.read_region((x, y), 2, (int(n_x*stepsize/16), int(n_y*stepsize/16)))
    lowres = np.array(lowres)[:,:,:3]

    x0 = 0
    # create multiporcessing pool
    print(mp.cpu_count())
    pool = mp.Pool(processes=8)
    tasks = []
    while x0 < n_x:
        task = tuple((slp, n_y, x, y, full_width_region, stepsize, x0, outdir, std_img))
        tasks.append(task)
        x0 += 1
    # slice images with multiprocessing
    temp = pool.starmap(v_slide, tasks)
    tempdict = list(zip(*temp))[0]
    tempimglist = list(zip(*temp))[1]
    temp = None
    pool.close()
    pool.join()

    tempdict = list(filter(None, tempdict))
    imloc = []
    list(map(imloc.extend, tempdict))
    imlocpd = pd.DataFrame(imloc, columns=["X_pos", "Y_pos", "X", "Y", "X_relative", "Y_relative", "file"])
    imlocpd = imlocpd.sort_values(["X_pos", "Y_pos"], ascending=[True, True])
    imlocpd = imlocpd.reset_index(drop=True)
    imlocpd = imlocpd.reset_index(drop=False)
    imlocpd.columns = ["Num", "X_pos", "Y_pos", "X", "Y", "X_relative", "Y_relative", "file"]
    imlocpd.to_csv(outdir + "/dict.csv", index=False)
    tempdict = None

    tempimglist = list(filter(None, tempimglist))
    imglist = []
    list(map(imglist.extend, tempimglist))
    ct = len(imloc)
    tempimglist = None
    imglist = np.asarray(imglist)

    return n_x, n_y, lowres, residue_x, residue_y, imglist, ct
