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

from   fbnnop import DEPTHWISE_CONV_2D, MAX_POOL_2D, CONV_2D, RELUx
from   fbnnpp import *
from   fbapix import *

if __name__=='__main__':
    import argparse
    args = argparse.ArgumentParser()
    def chF(f): return f if os.path.exists(f) else sys.exit(-1)
    args.add_argument('-t',"--tflite",       type=chF, default='y.tflite')
    args.add_argument('-v',"--verbose",      action='store_true')
    args.add_argument('-q',"--quantization", action='store_true')
    args = args.parse_args()

    #import tensorflow.examples.tutorials.mnist.input_data as input_data
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    floating_model = True
    img = preprocessing('dog.jpg', 416, 416)
    if args.quantization:
        img *= 255
        img  = img.astype(np.uint8)
        floating_model = False

    g = graph(tflite=args.tflite, floating=floating_model, verbose=args.verbose)
    g.allocate_graph(verbose=args.verbose)

    g.tensors[g.inputs[0]].set(img)
    predictions = g.invoke(verbose=False)

    if args.quantization:
        predictions = 1.19607841969*g.tensors[g.outputs[0]].data
        result_img = postprocessing(predictions,'dog.jpg',0.05,0.03,416,416)
        print("realize from Quantization")
    else:
        predictions = g.tensors[g.outputs[0]].data
        result_img = postprocessing(predictions,'dog.jpg',0.3,0.3,416,416)
        print("realize from Floating")

    cv2.imwrite('result.jpg',result_img)

