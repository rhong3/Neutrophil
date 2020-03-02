"""
Tile svs/scn files

Created on 11/01/2018

@author: RH
"""
import time
import matplotlib
import os
import shutil
import pandas as pd
matplotlib.use('Agg')
import Slicer
import staintools
import re


# Get all images in the root directory
def image_ids_in(root_dir, ignore=['.DS_Store', 'dict.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            dirname = id.split('.')[0]
            ids.append((id, dirname))
    return ids


def cut(stepsize, tilesize, path='../images/'):
    start_time = time.time()
    # load standard image for normalization
    std = staintools.read_image("../colorstandard.png")
    std = staintools.LuminosityStandardizer.standardize(std)
    imlist = image_ids_in(path)
    for i in imlist:
        begin_time = time.time()
        print(i)
        otdir = "../tiles/{}".format(i[1])
        try:
            os.mkdir(otdir)
        except FileExistsError:
            pass
        if os.path.exists("../tiles/{}/dict.csv".format(i[1])):
            print("{} exist!".format(i[1]))
            pass
        else:
            try:
                n_x, n_y, raw_img, residue_x, residue_y, imglist, ct = Slicer.tile(image_file=i[0], outdir=otdir,
                                                                                   std_img=std, stepsize=stepsize,
                                                                                   full_width_region=tilesize,
                                                                                   path_to_slide=path)
                print(ct)
            except IndexError:
                pass
        if len(os.listdir(otdir)) < 2:
            shutil.rmtree(otdir, ignore_errors=True)
        print("--- %s seconds ---" % (time.time() - begin_time))
    print("--- %s seconds ---" % (time.time() - start_time))


# Run as main
if __name__ == "__main__":
    if not os.path.isdir('../tiles'):
        os.mkdir('../tiles')
    cut(250, 299)

