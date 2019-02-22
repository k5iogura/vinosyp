# encoding: utf-8
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import argparse
from lib.utils import *
from lib.image_generator import *
from yolov2_network_v5 import *

parser = argparse.ArgumentParser(description="指定したパスのweightsファイルを読み込み、chainerモデルへ変換する")
#parser.add_argument('file', help="オリジナルのyolov2のweightsファイルへのパスを指定")
args = parser.parse_args()

#print("loading", args.file)
#file = open(args.file, "rb")
#dat=np.fromfile(file, dtype=np.float32)[4:] # skip header(4xint)
#dat =np.zeros((300000000),dtype=np.float32)[4:]
headerWords = 5
datO=np.zeros((300000000),dtype=np.float32)
dat =datO[headerWords:]
datO[0] = 0
datO[1] = 2
datO[2] = 0

# load model
print("loading initial model...")
n_classes = 1
n_boxes = 5
last_out = (n_classes + 5) * n_boxes

yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
yolov2.train = True
yolov2.finetune = False
model = YOLOv2Predictor(yolov2)

#serializers.load_npz("YOLO_network.npz",model)
serializers.load_hdf5("YOLO_network.model",model)

layers=[
    [3, 16, 3], 
    [16, 32, 3], 
    [32, 64, 3], 
    [64, 32, 1], 
    [32, 64, 3], 
    [64, 128, 3], 
    [128, 64, 1], 
    [64, 128, 3], 
    [128, 256, 3], 
    [256, 128, 1], 
    [128, 256, 3], 
    [256, 128, 1], 
    [128, 256, 3], 
    [256, 512, 3], 
    [512, 256, 1], 
    [256, 512, 3], 
    [512, 256, 1], 
    [256, 512, 3], 
    [512, 512, 3], 
    [512, 512, 3], 
    [1536, 512, 3],     # reorg -1 -4
]

print("To weights as dat ndarray")

offset=0
for i, l in enumerate(layers):
    in_ch = l[0]
    out_ch = l[1]
    ksize = l[2]

    start_offset = offset
    # load bias(Bias.bはout_chと同じサイズ)
    txt = "dat[%d:%d] = yolov2.bias%d.b.data" % (offset, offset+out_ch, i+1)
    offset+=out_ch
    exec(txt)

    # load bn(BatchNormalization.gammaはout_chと同じサイズ)
    txt = "dat[%d:%d] = yolov2.bn%d.gamma.data" % (offset, offset+out_ch, i+1)
    offset+=out_ch
    exec(txt)

    # (BatchNormalization.avg_meanはout_chと同じサイズ)
    txt = "dat[%d:%d] = yolov2.bn%d.avg_mean" % (offset, offset+out_ch, i+1)
    offset+=out_ch
    exec(txt)

    # (BatchNormalization.avg_varはout_chと同じサイズ)
    txt = "dat[%d:%d] = yolov2.bn%d.avg_var" % (offset, offset+out_ch, i+1)
    offset+=out_ch
    exec(txt)

    # load convolution weight(Convolution2D.Wは、outch * in_ch * フィルタサイズ。これを(out_ch, in_ch, 3, 3)にreshapeする)
    txt = "dat[%d:%d] = yolov2.conv%d.W.data.reshape(-1)" % (offset, offset+(out_ch*in_ch*ksize*ksize), i+1)
    exec(txt)
    print(dat[offset:offset+4], np.mean(dat[start_offset:start_offset+out_ch*4+out_ch*in_ch*ksize*ksize]))
    offset+= (out_ch*in_ch*ksize*ksize)
    print(i+1, offset)

# load last convolution weight(BiasとConvolution2Dのみロードする)
in_ch = 512
out_ch = last_out
ksize = 1

txt = "dat[%d:%d] = yolov2.bias%d.b.data" % (offset, offset+out_ch, i+2)
offset+=out_ch
exec(txt)

txt = "dat[%d:%d] = yolov2.conv%d.W.data.reshape(-1)" % (offset, offset+(out_ch*in_ch*ksize*ksize), i+2)
exec(txt)
print(dat[offset:offset+4])
offset+=out_ch*in_ch*ksize*ksize
print(i+2, offset)

# start dump chainer model as weights format
print("start dump chainer model as weights format")
print("dat size %d to used size %d"%(len(dat),offset))
datO[headerWords:headerWords+offset] = dat[:offset]
datO = datO[:headerWords+offset]
print("became dat size to ",len(datO), " saving")
datO.tofile("yolov2_network_v5.weights")
print("finished")

#print("save weights file to yolov2_darknet_hdf5.model")
#serializers.save_hdf5("yolov2_darknet_hdf5.model", yolov2)
#print("save weights file to yolov2_darknet.model")
#serializers.save_npz("yolov2_darknet.model", yolov2)

