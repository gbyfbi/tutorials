#!/usr/bin/env python
# see 1) https://github.com/BVLC/caffe/issues/290 (to read binaryproto)
# see 2) https://github.com/hughperkins/pytorch/issues/7 (LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgfortran.so.3.0.0) for me just conda install libgfortran

import cv2
import sys
import numpy as np
import lutorpy as lua
import os
torch = lua.require("torch")
loadcaffe = lua.require('loadcaffe')
image = lua.require('image')
# setup runtime and use zero-based index(optional, enabled by default)
# lua.LuaRuntime(zero_based_index=True)

imageHeight = 224
imageWidth  = 224
def preprocess(img_bgr, img_mean):
  img_bgr = image.scale(img_bgr,imageHeight,imageWidth,'bilinear')
  img_mean_bgr = img_mean._squeeze()
  if img_mean is not None:
    for i in range(1, 4):
        img_bgr[i]._add(-img_mean_bgr[i])
  return img_bgr

net = loadcaffe.load('/home/gao/Hadoop/hduser/data/models/VGG_CNN_S_deploy.prototxt', '/home/gao/Hadoop/hduser/data/models/VGG_CNN_S.caffemodel', 'nn')
net._evaluate();  #This sets the mode of the Module (or sub-modules) to train=false. This is useful for modules like Dropout that have a different behaviour during training vs evaluation.

synset_words = []
fo = open('/home/gao/Hadoop/hduser/data/synset_words.txt', 'r')
for line in fo.readlines():
    synset_words.append(line.rstrip())

image_mean = np.load('/home/gao/Hadoop/hduser/data/models/VGG_mean.npy').squeeze()
for line in sys.stdin:
    image_name = line.strip()
    im = cv2.imread(image_name)
    img_bgr = np.array(cv2.resize(im, dsize=(imageWidth, imageHeight)), dtype=np.double)
    img_bgr = np.rollaxis(img_bgr, 2)

    img_bgr -= image_mean
    I = torch.fromNumpyArray(img_bgr)
    score, classes = net._forward(I)._view(-1)._sort(True)
    print("%s\t%f" % (synset_words[classes[0]-1], score[0]))

