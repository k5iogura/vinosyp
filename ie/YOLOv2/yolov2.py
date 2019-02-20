#!/usr/bin/env python3
#STEP-1
from pdb import *
import sys,os
import numpy as np
from time import time
import cv2
import argparse
from postscript import *
from openvino.inference_engine import IENetwork, IEPlugin


args = argparse.ArgumentParser()
args.add_argument("images", nargs='*', type=str)
args.add_argument("-d", "--device"   , type=str, default="MYRIAD", help="Default MYRIAD or CPU")
args = args.parse_args()

input_image_size=(300,300)

data_type="FP16"
if args.device == "CPU": data_type="FP32"

#STEP-2
model_xml='darknet/built_graph/'+data_type+'/yolov2.xml'
model_bin='darknet/built_graph/'+data_type+'/yolov2.bin'
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
    frame = cv2.imread(f)
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #frame = cv2.resize(frame,input_image_size)
    #frame = frame.astype(dtype=np.float)
    #frame = frame[np.newaxis, :, :, :]
    #frame = frame.transpose(0,3,1,2)
    #frame/=255.0

    #STEP-6
    in_frame = cv2.resize(frame, (model_w, model_h))
    in_frame = in_frame[np.newaxis,:,:,:]
    in_frame = in_frame.transpose((0, 3, 1, 2))  # Change data layout from HWC to CHW
    #in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    #in_frame = in_frame.reshape((model_n, model_c, model_h, model_w))
    print("in-frame",in_frame.shape)

    start = time()
    #STEP-7
    #exec_net.start_async(request_id=0, inputs={input_blob: in_frame})
    res = exec_net.infer(inputs={input_blob: in_frame})
    print("res",res.keys())
    break

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
        boxes = get_boxes(res, 13, 13, 5, 20, 0.8, frame.shape[0], frame.shape[1], biases)
    else:
        print("error")

#STEP-10
del exec_net
del plugin
