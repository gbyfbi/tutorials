# see 1) https://github.com/BVLC/caffe/issues/290 (to read binaryproto)
# see 2) https://github.com/hughperkins/pytorch/issues/7 (LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgfortran.so.3.0.0) for me just conda install libgfortran

import cv2
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
# image_url = 'https://images-eu.ssl-images-amazon.com/images/I/61AhXv6rOBL._SL1200_.jpg'
image_url = 'http://upload.wikimedia.org/wikipedia/commons/e/e9/Goldfish3.jpg'
# image_url = 'https://upload.wikimedia.org/wikipedia/commons/d/d9/Audio-technica_erji.jpg'
image_url = 'http://animal-dream.com/data_images/horse/horse6.jpg'
# image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/85/Points_of_a_horse.jpg'
# image_url = 'http://r.ddmcdn.com/s_f/o_1/cx_633/cy_0/cw_1725/ch_1725/w_720/APL/uploads/2014/11/too-cute-doggone-it-video-playlist.jpg'
image_url = 'http://cdn1-www.dogtime.com/assets/uploads/2011/01/file_23262_entlebucher-mountain-dog-460x290.jpg'
image_url = 'http://cdn1-www.dogtime.com/assets/uploads/2011/01/file_23218_cockapoo-dog-breed-300x189.jpg'
image_url = 'http://mardaloopdoggiedaycare.com/wp-content/uploads/2014/12/cute-dog2.jpg'
image_url = 'http://up1.goumin.com/attachments/photo/0/0/16/4166/1066624.jpg'
image_url = 'http://grfx.cstv.com/schools/kty/blog/Strieby_action.jpg'
image_name = os.path.basename(image_url)
print(image_name)
if not os.path.isfile(image_name):
    os.system('wget '+image_url)
print ('==> Loading network')
net = loadcaffe.load('models/VGG_CNN_S_deploy.prototxt', 'models/VGG_CNN_S.caffemodel', 'cudnn')
# net = loadcaffe.load('models/VGG_ILSVRC_16_layers_deploy.prototxt', 'models/VGG_ILSVRC_16_layers.caffemodel', 'cudnn')
net._evaluate();  #This sets the mode of the Module (or sub-modules) to train=false. This is useful for modules like Dropout that have a different behaviour during training vs evaluation.
print(net)
print '==> Loading synsets'
print 'Loads mapping from net outputs to human readable labels'
synset_words = []
fo = open('synset_words.txt', 'r')
for line in fo.readlines():
    synset_words.append(line.rstrip())
print (synset_words)
#
print '==> Loading image and imagenet mean'
# im = image.load(image_name, 3, 'byte'):double()
im = cv2.imread(image_name)
print type(im)
print im.shape
img_bgr = np.array(cv2.resize(im, dsize=(imageWidth, imageHeight)), dtype=np.double)
img_bgr = np.rollaxis(img_bgr, 2)

image_mean = np.load('models/VGG_mean.npy').squeeze()
img_bgr -= image_mean
print type(img_bgr)
#
print '==> Preprocessing'
#
print 'Propagate through the network, sort outputs in decreasing order and show 5 best classes'
I = torch.fromNumpyArray(img_bgr)
score, classes = net._forward(I._cuda())._view(-1)._sort(True)
# print (net:get(net:size()-1).output)
for i in range(0, 10):
    print('score: %f, predicted class %d: %-60s' % (score[i], i, synset_words[classes[i]-1]))

