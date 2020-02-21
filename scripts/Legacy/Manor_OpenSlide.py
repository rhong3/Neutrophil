#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 12:48:13 2017

@author: lwk
"""


from openslide import OpenSlide
import numpy as np

slide = OpenSlide("Slide80.scn")

assert 'openslide.bounds-height' in slide.properties
assert 'openslide.bounds-width' in slide.properties
assert 'openslide.bounds-x' in slide.properties
assert 'openslide.bounds-y' in slide.properties

bounds_height = int(slide.properties['openslide.bounds-height'])
bounds_width = int(slide.properties['openslide.bounds-width'])
bounds_x = int(slide.properties['openslide.bounds-x'])
bounds_y = int(slide.properties['openslide.bounds-y'])

target_x = 25097
target_y = 44401

image_x = target_x + bounds_x
image_y = target_y + bounds_y

half_width_region = 50
full_width_region = 2 * half_width_region + 1

the_image = slide.read_region((image_x - half_width_region, image_y - half_width_region), 0, (full_width_region, full_width_region))
the_image.save("region.png")
the_image = the_image.convert("L")
the_image.save("grayscale.png")
the_image = the_image.rotate(90)
the_image.save("rotated90.png")
the_image = the_image.rotate(20)
the_image.save("rotated110.png")

half_width_feature = 15
full_width_feature = 2 * half_width_feature + 1

center_x = half_width_region + 1
center_y = half_width_region + 1

the_image = the_image.crop((center_x - half_width_feature, center_y - half_width_feature, center_x + half_width_feature + 1, center_y + half_width_feature + 1))
the_image.save("feature.png")

