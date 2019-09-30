# -*- coding: utf-8 -*-
import os, sys, re
import numpy as np
from   pdb import set_trace
from   inspect import getmembers

import struct

import tflite
from   tflite.Model import Model
import tflite.BuiltinOptions
import tflite.TensorType

import tflite.AddOptions
import tflite.CallOptions
import tflite.ConcatenationOptions
import tflite.Conv2DOptions
import tflite.DepthwiseConv2DOptions
import tflite.FullyConnectedOptions
import tflite.L2NormOptions
import tflite.Pool2DOptions
import tflite.QuantizationParameters
import tflite.RNNOptions
import tflite.ReshapeOptions
import tflite.ResizeBilinearOptions
import tflite.SoftmaxOptions

import tflite.OperatorCode
import tflite.BuiltinOperator
import tflite.ActivationFunctionType

import cv2
from   flags import flags

from   fbnnop import DEPTHWISE_CONV_2D, MAX_POOL_2D, CONV_2D, RELUx
from   fbnnpp import *
from   fbapix import *

import argparse
args = argparse.ArgumentParser()
def chF(f): return f if os.path.exists(f) else sys.exit(-1)
args.add_argument('-t',"--tflite",       type=chF, default='y.tflite')
args.add_argument('-i',"--image",        type=chF, default='dog.jpg')
args.add_argument('-v',"--verbose",      action='store_true')
args.add_argument('-q',"--quantization", action='store_true')
args = args.parse_args()

img = preprocessing(args.image, 416, 416)
if args.quantization:
    flags.floating_infer=False
    img *= 255
    img  = img.astype(np.uint8)
    print(flags.floating_infer)

g = graph(tflite=args.tflite, verbose=args.verbose)
g.allocate_graph(verbose=args.verbose)

g.tensors[g.inputs[0]].set(img)
predictions = g.invoke(verbose=False)

if args.quantization:
    predictions = g.tensors[g.outputs[0]].data
    result_img = postprocessing(predictions,args.image,0.015,0.025,416,416)
#    result_img = postprocessing(predictions,args.image,0.150,0.125,416,416)
    print("realize from Quantization")
else:
    predictions = g.tensors[g.outputs[0]].data
    result_img = postprocessing(predictions,args.image,0.3,0.3,416,416)
    print("realize from Floating")

cv2.imwrite('result.jpg',result_img)

