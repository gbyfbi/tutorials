-- Imagenet classification with Torch7 demo
--
-- Will be using Network-in-Network trained in Torch-7 with batch normalization
-- more information on it here
-- 
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'loadcaffe'
-- Rescales and normalizes the image
imageHeight = 224
imageWidth  = 224
function preprocess(im, img_mean)
  -- rescale the image
  local im3 = image.scale(im,imageHeight,imageWidth,'bilinear')
--  im3 = im3
  img_mean_bgr = img_mean:squeeze()
  img_bgr = torch.cat({im3[{{3}}], im3[{{2}}], im3[{{1}}]}, 1)
  -- subtract imagenet mean and divide by std
  if img_mean ~= nil then
    for i=1,3 do img_bgr[i]:add(-img_mean_bgr[i]) end
  end
  return img_bgr
end


print '==> Downloading image and network'
--local image_url = 'http://upload.wikimedia.org/wikipedia/commons/e/e9/Goldfish3.jpg'
--local image_url = 'http://cdn.bgr.com/2016/01/iphone-6s-34.jpg'
--local image_url = 'https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcTbhJHHhfDIlkpDmGkGzzqtJx89hOv6yXUKaXWsthnIsZIVM10p'
--local image_url = 'https://upload.wikimedia.org/wikipedia/commons/d/d9/Audio-technica_erji.jpg'
--local image_url = 'https://images-eu.ssl-images-amazon.com/images/I/61AhXv6rOBL._SL1200_.jpg'
local image_url = 'http://animal-dream.com/data_images/horse/horse6.jpg'
local image_name = paths.basename(image_url)
if not paths.filep(image_name) then os.execute('wget '..image_url)   end
--if not paths.filep(network_name) then os.execute('wget '..network_url)   end

print '==> Loading network'
--local net = torch.load(network_name):unpack():float()
--local net = loadcaffe.load('models/VGG_CNN_M_2048_deploy.prototxt', 'models/VGG_CNN_M_2048.caffemodel', 'cudnn')
local net = loadcaffe.load('models/VGG_CNN_S_deploy.prototxt', 'models/VGG_CNN_S.caffemodel', 'cudnn')
net:evaluate();  --This sets the mode of the Module (or sub-modules) to train=false. This is useful for modules like Dropout that have a different behaviour during training vs evaluation.
print(net)
image_mean = torch.load("models/VGG_mean.t7")
print '==> Loading synsets'
print 'Loads mapping from net outputs to human readable labels'
local synset_words = {}
for line in io.lines'synset_words.txt' do table.insert(synset_words, line:sub(11)); end

print '==> Loading image and imagenet mean'
local im = image.load(image_name, 3, 'byte'):double()

print '==> Preprocessing'
-- Our network has mean and std saved in net.transform
print('ImageNet ', net.transform)
local I = preprocess(im, image_mean):view(1,3,imageHeight,imageWidth):float()

--inputNode = net:get(1)()
--outputNode = net(inputNode)
--nodeGraph = nn.gModule({inputNode}, {outputNode})
--graph.dot(nodeGraph.fg, "forward imagenet")

print 'Propagate through the network, sort outputs in decreasing order and show 5 best classes'
local score,classes = net:forward(I:cuda()):view(-1):sort(true)
print (net:get(net:size()-1).output)
for i=1,10 do
  print('predicted class '..tostring(i)..': ', synset_words[classes[i] ], score[i])
end
