#!/usr/bin/env python3
import os, sys
import argparse
import numpy as np
import cv2
from tflite_runtime import interpreter as tf
from time import time
from pdb import set_trace
from fbnnpp import *

args = argparse.ArgumentParser()
def chF(f): return f if os.path.exists(f) else sys.exit(-1)
args.add_argument('-t',"--tflite",       type=chF, default='y.tflite')
args.add_argument('-i',"--image",        type=chF, default='dog.jpg')
args.add_argument('-v',"--verbose",      action='store_true')
args.add_argument('-q',"--quantization", action='store_true')
args = args.parse_args()

org = cv2.imread(args.image)
org_h, org_w = org.shape[:2]
img = cv2.resize(org, (416,416))
img = img[np.newaxis,:,:,:]
if args.quantization:
    img  = img.astype(np.uint8)
else:
    img  = img.astype(np.float32)
    img /= 255.

print("input image size:",org.shape,img.dtype,img.max(),img.min(),img.mean())

ip = tf.Interpreter(model_path=args.tflite)
ip.allocate_tensors()

infoi=ip.get_input_details()
infoo=ip.get_output_details()

indexi=infoi[0]['index']
indexo=infoo[0]['index']

start=time()
ip.set_tensor(indexi, img)
ip.invoke()

if args.quantization:
    predictions = 1.19607841969*(ip.get_tensor(indexo) - 42.)
    result_img = postprocessing(predictions,'dog.jpg',0.005,0.005,416,416)
else:
    predictions = ip.get_tensor(indexo)
    result_img = postprocessing(predictions,'dog.jpg',0.3,0.3,416,416)

cv2.imwrite('result.jpg',result_img)

def view(idx):
    mx=ip.get_tensor(idx).max()
    mn=ip.get_tensor(idx).min()
    me=ip.get_tensor(idx).mean()
    sd=ip.get_tensor(idx).std()
    print("min/max/mean = {:.3f}/{:3f}/{:3f}:{:.6f}".format(mn,mx,me,sd))
