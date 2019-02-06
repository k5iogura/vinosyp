#!/usr/bin/env python3
#STEP-1
from pdb import *
import sys,os
import numpy as np
from time import time
import cv2
import argparse
#from postscript import *
from openvino.inference_engine import IENetwork, IEPlugin

def save_as_txt(data,outfile):
    a1 = data.reshape(-1)
    print("a1.shape=",a1.shape)
    with open(outfile,"w") as fp:
        for i in range(len(a1)):
            fp.write("%.7f\n"%a1[i])

def embed_image(source, dest, dx, dy):
    (h,w,c)=source.shape
    for k in range(0,c):
        for y in range(0,h):
            for x in range(0,w):
                val = source[y][x][k]
                dest[dy+y][dx+x][k] = val

def letterbox_image(im, w, h):
    # HWC im
    (im_h, im_w, im_c) = im.shape
    new_w = im_w
    new_h = im_h
    if 1.0*w/new_w < 1.0*h/new_h:
        new_w = w
        new_h =int((im_h * w)/ im_w)
    else:
        new_h = h
        new_w =int((im_w * h)/ im_h)
    resized = cv2.resize(im, (new_w,new_h))
    boxed   = np.full((h,w,3),0.5)
    print("embed_image(resized, boxed,",int((w-new_w)//2), int((h-new_h)//2))
    embed_image(resized, boxed, int((w-new_w)//2), int((h-new_h)//2))
    for i in range(46208,46218): print(resized.transpose((2,0,1)).reshape(-1)[i])
    return boxed

def softmax(a):
    exp_a = np.exp( a )
    sum_exp_a = np.sum(exp_a)
    y= exp_a / sum_exp_a
    return y

args = argparse.ArgumentParser()
args.add_argument("images", nargs='*', type=str)
args.add_argument("-d", "--device"   , type=str, default="MYRIAD", help="Default MYRIAD or CPU")
args = args.parse_args()

input_image_size=(300,300)

data_type="FP16_2"
if args.device == "CPU": data_type="FP32"

#STEP-2
model_xml='tfnet/'+data_type+'/yolov2-voc.xml'
model_bin='tfnet/'+data_type+'/yolov2-voc.bin'
model_xml = os.environ['HOME'] + "/" + model_xml
model_bin = os.environ['HOME'] + "/" + model_bin
net = IENetwork(model=model_xml, weights=model_bin)

#STEP-3
print(model_bin, "on", args.device)
plugin = IEPlugin(device=args.device, plugin_dirs=None)
exec_net = plugin.load(network=net, num_requests=1)

#STEP-4
input_blob = next(iter(net.inputs))  #input_blob = 'data'
out_blob   = next(iter(net.outputs)) #out_blob   = 'detection_out'
model_n, model_c, model_h, model_w = net.inputs[input_blob].shape #Tool kit R4
print("n/c/h/w (from xml)= %d %d %d %d"%(model_n, model_c, model_h, model_w))
print("input_blob : out_blob =[",input_blob,":",out_blob,"]")
print(net.outputs[out_blob].shape)

del net

#STEP-5
# VOC YOLOv2 output layer structure
# res.shape = (1, 21125)
# 21125     = 13*13*125
# 125       = 25 (=xywh+conf+class)  * 5    5:region-layer.num in cfg
# 13        = 416(=input_image_size) / 32  32:down-sampling ratio
# threshold = 0.6                         0.6:region-layer.thresh in cfg
# :x1:y1:x2:y2:conf:20*classes_prob:

thresh=0.6
files=[]
if len(args.images)>0:files = args.images
for f in files:
    print("input image = %s"%f)
    frame = cv2.imread(f)   # HWC
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    save_as_txt((frame/255.).transpose(2,0,1),"dog_im.txt")
#    for i in range(10):
#        print(i,frame.transpose((2,0,1)).reshape(-1)[i]/255.0)
    frame =frame/255.0
    print("orig shape:", frame.shape)
    frame =letterbox_image(frame, model_w, model_h)
    flat_ = frame.transpose(2,0,1).reshape(-1)
    save_as_txt(flat_,"dog_sized.txt")
#    j=0
#    for i in range(frame.shape[0]*frame.shape[1]):
#        if flat_[i] != 0.5:
#            print(i,flat_[i])
#            j+=1
#        if j>=10:break

    #STEP-6
    in_frame = cv2.resize(frame, (model_w, model_h))
    in_frame = in_frame[np.newaxis,:,:,:]
    in_frame = in_frame.transpose((0, 3, 1, 2))  # Change data layout from HWC to CHW
    print("in-frame",in_frame.shape)

    start = time()
    #STEP-7
    #exec_net.start_async(request_id=0, inputs={input_blob: in_frame})
    res = exec_net.infer(inputs={input_blob: in_frame})
    for outkey in res.keys(): print("outkey=",outkey)
    x   = [ res[outkey][0][i] for i in range(0,res[outkey].shape[-1],85) ]
    y   = [ res[outkey][0][i] for i in range(1,res[outkey].shape[-1],85) ]
    w   = [ res[outkey][0][i] for i in range(2,res[outkey].shape[-1],85) ]
    h   = [ res[outkey][0][i] for i in range(3,res[outkey].shape[-1],85) ]
    conf= [ res[outkey][0][i] for i in range(4,res[outkey].shape[-1],85) ]
    prob= [ softmax(res[outkey][0][i:i+20]) for i in range(5,res[outkey].shape[-1],85) ]
    #for i in range(5,res[outkey].shape[-1],85):
        #res[outkey][0][i:i+20] = softmax(res[outkey][0][i:i+20])
    save_as_txt(res[outkey][0],"dog_region.txt")

#    print("res",res.keys(),res['output/YoloRegion'][0].shape)
#    j=5+100*85
#    for i in range(1,10):
#        print("conf",j,res['output/YoloRegion'][0][j])
#        j += i*85
#    break

    biases =[[1.3221 ,1.73145],[3.19275,4.00944],
             [5.05587,8.09892],[9.47112,4.84053],
             [11.2364,10.0071]]
    if exec_net.requests[0].wait(-1) == 0:
        sec = time()-start
        res = exec_net.requests[0].outputs[out_blob]
        res2=res.reshape(-1,125)
        print("res",res.shape)
        print("fin",res2.shape)
        print("top")
        for j in range(res2.shape[0]):
            if res2[j][4] > thresh:
                print(res2[j][4])
                print(res2[j][0:4])
                print(res2[j][4]*res2[j][5:25])
                break
        print("elapse = %.3fmsec %.3fFPS"%(1000.0*sec,1.0/sec))
#        boxes = get_boxes(res, 13, 13, 5, 20, 0.8, frame.shape[0], frame.shape[1], biases)
    else:
        print("error")

#STEP-10
del exec_net
del plugin
