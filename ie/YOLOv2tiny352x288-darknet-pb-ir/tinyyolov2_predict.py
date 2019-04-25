#!/usr/bin/env python3
#STEP-1
from pdb import *
import sys,os
from math import sqrt
from time import time
import cv2
import numpy as np
import argparse
from openvino.inference_engine import IENetwork, IEPlugin

from test import preprocessing, postprocessing

args = argparse.ArgumentParser()
args.add_argument("images", nargs='*', type=str)
args.add_argument("-d", "--device"   , type=str, default="MYRIAD", help="Default MYRIAD or CPU")
args = args.parse_args()

data_type="FP16"
if args.device == "CPU": data_type="FP32"

model_xml='./'+data_type+'/y.xml'
model_bin='./'+data_type+'/y.bin'
plugin = IEPlugin(device=args.device, plugin_dirs=None)
extension = "/inference_engine_samples/intel64/Release/lib/libcpu_extension.so"
extension = os.environ['HOME']+extension
if args.device == "CPU":plugin.add_cpu_extension(extension)
net = IENetwork(model=model_xml, weights=model_bin)	# R5

print(model_bin, "on", args.device)
exec_net = plugin.load(network=net, num_requests=1)

input_blob = next(iter(net.inputs))  #input_blob = 'data'
out_blob   = next(iter(net.outputs)) #out_blob   = 'detection_out'
print(net.inputs[input_blob].shape)
model_n, model_c, model_h, model_w = net.inputs[input_blob].shape #Tool kit R4
print("input_blob shape(from xml)", net.inputs[input_blob].shape)
print("name (input_blob : out_blob) =",input_blob,":",out_blob)

score_threshold = 0.3
iou_threshold = 0.3

total_sec  = 1e-10
done_image = 0
for input_img_path in args.images:
    print(input_img_path)
    outfile = os.path.splitext(os.path.basename(input_img_path))[0]+"_result.png"
    in_frame = preprocessing(input_img_path,model_h,model_w)
    in_frame = in_frame.transpose((0,3,1,2))

    start = time()
    exec_net.start_async(request_id=0, inputs={input_blob: in_frame})

    if exec_net.requests[0].wait(-1) == 0:
        sec = time() - start
        total_sec+=sec
        done_image+=1
        res = exec_net.requests[0].outputs[out_blob]
        res = res.transpose((0,2,3,1))
        res_image = postprocessing(res,input_img_path,score_threshold,iou_threshold,model_h,model_w)
        cv2.imwrite(outfile,res_image)
    else:
        print("error")

print("%.3fFPS"%(done_image/total_sec))
del net
del exec_net
del plugin

sys.exit(-1)
