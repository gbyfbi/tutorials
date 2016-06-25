# see 1) https://github.com/BVLC/caffe/issues/290
# see 2) https://github.com/hughperkins/pytorch/issues/7 (LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgfortran.so.3.0.0)
import caffe
import numpy as np
import lutorpy as lua
# setup runtime and use zero-based index(optional, enabled by default)
lua.LuaRuntime(zero_based_index=True)
torch = lua.require("torch")

blob = caffe.proto.caffe_pb2.BlobProto()
in_binary_proto_path = "models/VGG_mean.binaryproto"
out_path = "out.npy"
data = open(in_binary_proto_path , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
out = arr[0]
np.save(out_path, out)
t = torch.DoubleTensor(11, 11)._fill(1)
torch.save("test_save_from_python.t7", t)
image_mean_torch_tensor = torch.fromNumpyArray(arr)
torch.save("models/VGG_mean.t7", image_mean_torch_tensor)
np.save("models/VGG_mean.npy", arr)


