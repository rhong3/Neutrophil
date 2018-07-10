from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
#import pandas as pd
from PIL import Image
from skimage import morphology
from skimage.color import rgb2hed
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.filters import threshold_adaptive
from skimage.util import img_as_ubyte
#import matplotlib.pyplot as plt

#from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from six.moves import xrange
    
def get_data(imagename,dim,out=False,outdir='./'):
    """ imagename: the file name to read
        dim: dimension of the roi tile in pixels
    """
    im=Image.open(imagename)
    pix=np.array(im)
    pix=pix[:,:,0:3]
    nrow, ncol, _ = pix.shape    
    subh=rgb2hed(pix)[:,:,0]
    subh_min=np.amin(subh)
    subh_range=np.ptp(subh)
    hf=(subh-subh_min)/subh_range
    hf=hf*2.0-1.0
    hb=img_as_ubyte(hf)
    block_size = 65
    binary = threshold_adaptive(hb, block_size,method='mean')
    clean_bw = morphology.closing(binary)
    clean_bw=clear_border(clean_bw)
    clean_bw=morphology.remove_small_objects(clean_bw, 20)
    image_label = label(clean_bw)
    nucprop=regionprops(image_label,intensity_image=hf)
   
    dat=np.empty((0,int(dim**2)),dtype='uint8')
    dim2=int(dim/2)
    #i=0
    for region in nucprop:  
        coords=list(region.centroid)
        coords=list(map(int,coords))
        if ((dim2<coords[0]<(nrow-dim2)) & (dim2<coords[1]<(ncol-dim2))):           
            #coords_list=pd.read_table(coords,sep='\t',header=None,names=['x','y','creator','time'])   
            """normalize the image data to float [0,1] will be carried out in the training step"""
            tile=hb[(coords[0]-dim2):(coords[0]+dim2),(coords[1]-dim2):(coords[1]+dim2)]
            tile=tile.astype(np.uint8)
            #print(len(tile.flatten()))
            dat=np.vstack([dat,tile.flatten()])
            """
            fig,ax = plt.subplots(1)
            pixSize=list(pix.shape)
            fig.set_size_inches(pixSize[1]/fig.dpi,pixSize[0]/fig.dpi)
            ax.imshow(tile,cmap='Greys')
            plt.axis('off')
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.savefig(imagename+str(i)+'_roi'+'.tif',dpi=fig.dpi,pad_inches=0)
            i=i+1
            """
        else:
            continue
    if out:
        np.savetxt(outdir+imagename+'_data.txt',dat,fmt='%i',delimiter='\t')
    else:
        return dat


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        if np.amax(images)>1:
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


