#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:29:08 2016

@author: lwk


"""

#import openslide
#from openslide.deepzoom import DeepZoomGenerator
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io
from PIL import Image
from skimage.segmentation import mark_boundaries
from skimage.segmentation import clear_border
from skimage.color import rgb2hed
from skimage.measure import label, regionprops
from matplotlib.colors import LinearSegmentedColormap
from skimage.filters import threshold_otsu, threshold_adaptive, rank
from scipy import ndimage as ndi
from skimage import morphology
from skimage.util import img_as_ubyte


indir_=sys.argv[1]
file_=sys.argv[2]

if len(sys.argv)>3:
    outdir_=sys.argv[3]
else:
    outdir_=indir_


#im=Image.open('/home/lwk/HE_Image/test.tif')
#im=Image.open('/home/lwk/HE_Image/26279_pos/ImageCollection_0000026279_2016-10-27 14_07_46__Tile_7_18.tif')
#im=Image.open('/home/lwk/HE_Image/annotated_slices/Slide_80_2__Slice_73282_19067_2000_1000 (neutrophils).tif')
im=Image.open(indir_+'/'+file_)
pix=np.array(im)
pix=pix[:,:,0:3]

#pixsub=pix#[2**10*4:2**10*4+127,2**10*4:2**10*4+127,0:3]

#plt.imshow(pixsub)
#plt.show()


cmap_hema=LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
subh=rgb2hed(pix)[:,:,0]
subh_min=np.amin(subh)
subh_range=np.ptp(subh)
hf=(subh-subh_min)/subh_range
hf=hf*2.0-1.0
hb=img_as_ubyte(hf)

#plt.imshow(hb,cmap=cmap_hema)
#plt.show()

"""

# global otsu method, not optimal
thresh=threshold_otsu(hf)
binary=hf>thresh

# local otsu method, not optimal, very slow
radius = 2**5
selem = morphology.disk(radius)
local_otsu = rank.otsu(hb, selem)
binary=hb>local_otsu

"""

block_size = 65
binary = threshold_adaptive(hb, block_size,method='mean')

#plt.imshow(binary,cmap=plt.cm.gray)
#plt.show()


clean_bw = morphology.closing(binary)
clean_bw=clear_border(clean_bw)
clean_bw=morphology.remove_small_objects(clean_bw, 20)
image_label = label(clean_bw)

#plt.imshow(clean_bw,cmap=plt.cm.gray)
#plt.show()

#plt.imshow(mark_boundaries(pix,clean_bw))
fig,ax = plt.subplots(1)
pixSize=list(pix.shape)
fig.set_size_inches(pixSize[1]/fig.dpi,pixSize[0]/fig.dpi)

#ax=plt.imshow(mark_boundaries(pix,clean_bw))
#ax.imshow(mark_boundaries(pix,clean_bw))
ax.imshow(pix)

nucprop=regionprops(image_label,intensity_image=hf)


for region in nucprop:
    eigList=list(region.inertia_tensor_eigvals)
    eigRate=eigList[0]/eigList[1]
    if (eigRate<10)&(region.solidity<0.8)&(region.mean_intensity>0.2)&(region.area>100)&(region.major_axis_length<40):
        minr, minc, maxr, maxc=region.bbox
        rect=mpatches.Rectangle((minc-5,minr-5),maxc-minc+10,maxr-minr+10,
                                fill=False, edgecolor='cyan',linewidth=2)
        ax.add_patch(rect)
    
    else:
        continue
    
  
plt.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.savefig(outdir_+'/'+file_+'_auto_labeled'+'.tif',dpi=fig.dpi,pad_inches=0)


