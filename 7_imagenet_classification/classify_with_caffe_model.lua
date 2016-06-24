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
imageHeight = 227
imageWidth  = 227
function preprocess(im, img_mean)
  -- rescale the image
  local im3 = image.scale(im,imageHeight,imageWidth,'bilinear')
  -- subtract imagenet mean and divide by std
  if img_mean ~= nil then
    for i=1,3 do im3[i]:add(-img_mean.mean[i]):div(img_mean.std[i]) end
  end
  return im3
end


print '==> Downloading image and network'
local image_url = 'http://upload.wikimedia.org/wikipedia/commons/e/e9/Goldfish3.jpg'
--local network_url = 'https://www.dropbox.com/s/npmr5egvjbg7ovb/nin_nobn_final.t7'
local image_name = paths.basename(image_url)
--local network_name = paths.basename(network_url)
if not paths.filep(image_name) then os.execute('wget '..image_url)   end
--if not paths.filep(network_name) then os.execute('wget '..network_url)   end

print '==> Loading network'
--local net = torch.load(network_name):unpack():float()
local net = loadcaffe.load('deploy.prototxt', 'AlexNet_SalObjSub.caffemodel', 'cudnn')
net:evaluate()
print(net)

print '==> Loading synsets'
print 'Loads mapping from net outputs to human readable labels'
local synset_words = {}
for line in io.lines'synset_words.txt' do table.insert(synset_words, line:sub(11)) end

print '==> Loading image and imagenet mean'
local im = image.load(image_name)

print '==> Preprocessing'
-- Our network has mean and std saved in net.transform
print('ImageNet ', net.transform)
local I = preprocess(im, net.transform):view(1,3,imageHeight,imageWidth):float()

inputNode = net:get(1)()
outputNode = net(inputNode)
nodeGraph = nn.gModule({inputNode}, {outputNode})
graph.dot(nodeGraph.fg, "forward imagenet")

print 'Propagate through the network, sort outputs in decreasing order and show 5 best classes'
local score,classes = net:forward(I:cuda()):view(-1):sort(true)
print (net:get(net:size()-1).output)
for i=1,5 do
  print('predicted class '..tostring(i)..': ', synset_words[classes[i] ], score[i])
end
