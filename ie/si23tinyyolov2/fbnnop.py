import numpy as np
import sys,os
import math
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from pdb import *
from flags import flags

# NHWC : input  tensor shape   = ( batch,  h, w, in_ch )
# NHWC : output tensor shape   = ( batch,  h, w, in_ch )
# 1HWC : filter tensor shape   = ( 1,      k, k, in_ch ) at DEPTHWISE
# CHWC : filter tensor shape   = ( out_ch, k, k, in_ch ) at CONV_2D
# C    : bias   tensor shape   = ( in_ch               )

def RELUx(numpy_in, val=0, leaky=None):
    assert numpy_in.dtype != np.uint8,"RELU not supports {}".format(np.uint8)
    numpy_out = numpy_in.copy()
    if val > 1:                    # RELUx
        numpy_out[numpy_out < 0]   = 0
        numpy_out[numpy_out > val] = val
    elif val == 1:                 # RELU1
        numpy_out[numpy_out < -1]  = -1
        numpy_out[numpy_out > val] =  1
    elif leaky is not None:        # LEAKY RELU
        numpy_out[numpy_out < 0]  *= leaky
    else:                          # RELU
        numpy_out[numpy_out < 0]   = 0
    return numpy_out

# MultiplyByQuantizedMultiplier instead of tensorflow-lite reference code
def MBQM(acc, multiplier_fx, shift):
    f1 = (multiplier_fx * acc)
    lsb= 1 if f1 & (1<<(shift - 1)) else 0
    f1 = f1 >> shift
    f1+=lsb
    return f1

def mbqm(acc, multiplier_fx, shift):
    f1 = (multiplier_fx * acc)
    # 1 or 0 to get round with ndarray
    lsb= (f1 & (1<<(shift - 1)))>>(shift - 1)
    f1 = f1 >> shift
    f1+=lsb
    return f1

def CONV_2D(operator, outputs, inputs, verbose=True):
    _floating_infer = flags.floating_infer
    (padding, stride, strideh, _activation_) = operator.Builtin_Options()
    (tensor_idx_input, tensor_idx_filter, tensor_idx_bias) = inputs
    tensor_input     = operator.tensors[tensor_idx_input]
    tensor_idx_output= outputs[0]
    tensor_output    = operator.tensors[tensor_idx_output]
    tensor_filter    = operator.tensors[tensor_idx_filter]
    tensor_bias      = operator.tensors[tensor_idx_bias]
    filter_size      = tensor_filter.data.shape[1] # kernel height NHWC

    patches = []
    output_ = []
    input_shape = tensor_input.data.shape
    output_height, output_width = tensor_output.data.shape[1:3]
    output_ch   = tensor_filter.shape[0]
    
    D = tensor_input.data.copy()
    if not _floating_infer: D -= tensor_input.zero_point
    # stride 1
    # output 1,14,14,64
    # input  1,14,14,32
    # filter 64,5,5,32
    # bias   64

    # <by padding>
    _pad = ((output_height - 1)*stride - input_shape[1] + filter_size)/2.
    _pad = int(math.ceil(_pad))
    operator.padding = _pad
    #_pad = int(math.ceil(((output_height - 1)*stride - input_shape[1] + filter_size)/2))
    # Padding along height and width
    if _pad > 0:
        D = np.pad(
            D,
            ((0,0),(_pad,_pad),(_pad,_pad),(0,0)),
            mode='constant', constant_values=(0,0)
        )
    elif _pad < 0:
        operator.view("Invalid padding size",cont=True)
        set_trace()
    # output 1,14,14,64
    # input  1,14,14,32
    # filter 64,5,5,32
    # bias   64
    
    B = tensor_bias.data
    F = tensor_filter.data
    
    for row in range(int(output_height)):
        for col in range(int(output_width)):     
            row_start = row*stride
            row_end = row_start + filter_size
            col_start = col*stride
            col_end = col_start + filter_size
            #patches.append(tensor_input.data[:, row_start:row_end, col_start:col_end, :]) ##M
            # apatch 1,5,5,32
            apatch=D[:, row_start:row_end, col_start:col_end, :]
            if apatch.shape[1:]!=F.shape[1:]:
                set_trace()
            assert apatch.shape[1:]==F.shape[1:],"Failed {} {}".format(apatch.shape, F.shape)
            patches.append(apatch)
    # patches 14*14,5,5,32
    patches = np.concatenate(patches, axis=0)

    # FF 64,14*14,5,5,32
    # Q  64,14*14,5,5,32
    if False:
  #  if output_ch<512:
        if False:
            FF = []
            for i in range(output_ch):
                temp_ = []
                for j in range(output_height*output_width):
                    temp_.append(F[i][np.newaxis,:])
                FF.append(np.concatenate(temp_, axis=0)[np.newaxis,:])
            FF= np.concatenate(FF,axis=0)

            Q = FF.copy()
            for i in range(output_ch):
                Q[i] *= patches
            Q = np.sum(Q, axis=(2,3,4)).reshape(-1, output_height, output_width)
            for i in range(output_ch):
                Q[i] += B[i]
            output_ = mbqm(Q, operator.factor_fx, 16) if not _floating_infer else Q
        else:
            # FX = np.tile(F.reshape(st),(1,output_height*output_width,1,1,1))
            #
            # patches  1, 14*14, 5, 5, 32
            # FX      64,     1, 5, 5, 32
            # tt      64, 14*14, 5, 5, 32   by broadcast
            sl= list(F.shape)
            sl.insert(1,1)
            st = tuple(sl)
            FX = F.reshape(st)
            tt = patches[np.newaxis,:] * FX
            tt = np.sum(tt, axis=(2,3,4)).reshape(-1, output_height, output_width)
            tt+= B[:,np.newaxis,np.newaxis]
            output_ = mbqm(tt, operator.factor_fx, 16) if not _floating_infer else tt
    else:
        # temp_ = []  # for DepthWiseConv
        outputX = []
        if True:
            if not _floating_infer:
                F       = F.astype(np.int16)
                patches = patches.astype(np.int16)
            if output_ch < 256:
                for filter_, bias in zip(F, B):
                    # Fx          1,5,5,32
                    # patches 14*14,5,5,32
                    # tt      14*14,5,5,32
                    # tsum(f) 14*14
                    Fx    = filter_[np.newaxis,:]
                    tt    = patches * Fx
                    tsum  = np.sum(tt, axis=(1,2,3))
                    tsum += bias
                    tsum = mbqm(tsum, operator.factor_fx, 16) if not _floating_infer else tsum
                    output_.append(tsum.reshape(output_height, output_width))
                output_ = np.array(output_)
            else:
                for patche_ in patches:
                    # patche_  1,5,5,32
                    # F       64,5,5,32
                    # tsum    64
                    tt = F * patche_[np.newaxis,:]
                    tsum  = np.sum(tt, axis=(1,2,3))
                    tsum += B
                    tsum = mbqm(tsum, operator.factor_fx, 16) if not _floating_infer else tsum
                    output_.append(tsum)
                output_ = np.array(output_).reshape(output_height, output_width, -1)
                output_ = np.transpose(output_,(2,0,1))
        else:
            for filter_, bias in zip(F, B):
                temp_ = []  # for CONV
                for patch_idx, patch_ in enumerate(patches):
                    # patch_ 5,5,32
                    conv = (np.sum(patch_ * filter_) + bias)              # for CONV as scaler
                    #conv = (np.sum(patch_ * filter_, axis=(0,1)) + bias)   # for DepthWiseConv
                    if not _floating_infer: conv = MBQM(conv, operator.factor_fx, 16)
                    temp_.append(conv)
                #temp_ 14*14
                output_.append(np.array(temp_).reshape(output_height, output_width)) # for CONV
            output_ = np.array(output_)
    # output_ 64,14,14
    output_ = np.transpose(output_, (1,2,0)) # for CONV
    if not _floating_infer: output_+= tensor_output.zero_point
    if not _floating_infer: output_ = np.clip(output_, 0, np.int32(tensor_output.max))
    # output_ 1,14,14,64
    output_ = output_[np.newaxis, :]
    #output_ = np.asarray(temp_).reshape((1, output_height, output_width, -1)) # for DepthWiseConv
    if _activation_ is not None:
        if   "RELU"  in _activation_: output_ = RELUx(output_, 0)
        elif "RELU1" in _activation_: output_ = RELUx(output_, 1)
        elif "RELU6" in _activation_: output_ = RELUx(output_, 6)
        else: print(_activation_+' not supported')
    assert output_.shape == tensor_output.data.shape,"Mismatch {} {}".format(
                            output_.shape,tensor_output.data.shape)
    tensor_output.data = output_
    return output_

def DEPTHWISE_CONV_2D(operator, outputs, inputs, verbose=True):
    (padding, stride, strideh, _activation_,depth_multiplier) = operator.Builtin_Options()
    (tensor_idx_input, tensor_idx_filter, tensor_idx_bias)    = inputs
    tensor_input     = operator.tensors[tensor_idx_input]
    tensor_idx_output= outputs[0]
    tensor_output    = operator.tensors[tensor_idx_output]
    tensor_filter    = operator.tensors[tensor_idx_filter]
    tensor_bias      = operator.tensors[tensor_idx_bias]
    filter_size      = tensor_filter.data.shape[1] # kernel height NHWC

    patches = []
    output_ = []
    input_shape = tensor_input.data.shape
    output_height, output_width = tensor_output.data.shape[1:3]
    
    D = tensor_input.data.copy()
    # <by depth_multiplier>
    # output 1,28,28,32
    # input  1,28,28,1  (depth_multiplier==32)
    # filter 1,5,5,32
    # bias   32
    if depth_multiplier>0:
        np_concat = []
        for m in range(depth_multiplier):
            np_concat.append(D)
        D = np.concatenate(np_concat,axis=3)
    # output 1,28,28,32
    # input  1,28,28,32 <= changed
    # filter 1,5,5,32
    # bias   32

    # <by padding>
    _pad = ((output_height - 1)*stride - input_shape[1] + filter_size)/2.
    _pad = int(math.ceil(_pad))
    operator.padding = _pad
    #_pad = int(math.ceil(((output_height - 1)*stride - input_shape[1] + filter_size)/2))
    # Padding along height and width
    if _pad > 0:
        D = np.pad(
            D,
            ((0,0),(_pad,_pad),(_pad,_pad),(0,0)),
            mode='constant', constant_values=(0,0)
        )
    elif _pad < 0:
        operator.view("Invalid padding size",cont=False)
    # output 1,28,28,32
    # input  1,34,34,32 <= changed
    # filter 1,5,5,32
    # bias   32
    
    B = tensor_bias.data
    F = tensor_filter.data
    
    for row in range(int(output_height)):
        for col in range(int(output_width)):     
            row_start = row*stride
            row_end = row_start + filter_size
            col_start = col*stride
            col_end = col_start + filter_size
            #patches.append(tensor_input.data[:, row_start:row_end, col_start:col_end, :]) ##M
            # apatch 1,5,5,32
            apatch=D[:, row_start:row_end, col_start:col_end, :]
            assert apatch.shape == F.shape,"Failed {} {}".format(apatch.shape, F.shape)
            patches.append(apatch)
    # patches N,5,5,32
    patches = np.concatenate(patches, axis=0)
    temp_ = []  # for DepthWiseConv
    for filter_, bias in zip(F, B):
        # temp_ = []  # for CONV
        # filter_ 5,5,32
        for patch_idx, patch_ in enumerate(patches):
            # patch_ 5,5,32
            #conv = (np.sum(patch_ * filter_) + bias)              # for CONV
            conv = (np.sum(patch_ * filter_, axis=(0,1)) + bias)   # for DepthWiseConv
            temp_.append(conv)
        # output_.append(np.array(temp_).reshape(int(output_height), int(output_width))) # for CONV
    #output_ = np.transpose(np.array(output_), (1,2,0)) # for CONV
    output_ = np.asarray(temp_).reshape((1, output_height, output_width, -1))
    if _activation_ is not None:
        if   "RELU"  in _activation_: output_ = RELUx(output_, 0)
        elif "RELU1" in _activation_: output_ = RELUx(output_, 1)
        elif "RELU6" in _activation_: output_ = RELUx(output_, 6)
        else: print(_activation_+' not supported')
    assert output_.shape == tensor_output.data.shape,"Mismatch {} {}".format(
                            output_.shape,tensor_output.data.shape)
    tensor_output.data = output_
    return output_

def MAX_POOL_2D(operator, outputs, inputs, verbose=True):
    (padding, stride, strideh, _activation_, filter_size, filterheight) = operator.Builtin_Options()
    tensor_idx_input = inputs[0]
    tensor_input     = operator.tensors[tensor_idx_input]
    tensor_idx_output= outputs[0]
    tensor_output    = operator.tensors[tensor_idx_output]

    patches = []
    output_ = []
    input_shape = tensor_input.data.shape
    output_height, output_width = tensor_output.data.shape[1:3]
    
    D = tensor_input.data.copy()
    # input  1,28,28,32
    # output 1,14,14,32

    # <by padding>
    _pad = int(math.ceil(((output_height - 1)*stride - input_shape[1] + filter_size)/2))
    operator.padding = _pad
    # Padding along height and width
    if _pad > 0:
        D = np.pad(
            D,
            ((0,0),(_pad,_pad),(_pad,_pad),(0,0)),
            mode='constant', constant_values=(0,0)
        )
    elif _pad < 0:
        operator.view("Invalid padding size",cont=False)
    # input  1,28,28,32
    # output 1,14,14,32
    for row in range(int(output_height)):
        for col in range(int(output_width)):     
            row_start = row*stride
            row_end = row_start + filter_size
            col_start = col*stride
            col_end = col_start + filter_size
            # apatch N,2,2,32
            apatch=D[:, row_start:row_end, col_start:col_end, :]
            mpatch=np.max(apatch, axis=(1), keepdims=True)   # N,1,2,32
            mpatch=np.max(mpatch, axis=(2), keepdims=False)  # N,1,32
            # mpatch N,14*14,32
            patches.append(mpatch)
    # patches N,14*14,32
    patches = np.concatenate(patches, axis=1)
    # patches N,14,14,32
    patches = patches.reshape(-1, output_height, output_width, patches.shape[-1])
    output_ = patches
    if _activation_ is not None:
        if   "RELU"  in _activation_: output_ = RELUx(output_, 0)
        elif "RELU1" in _activation_: output_ = RELUx(output_, 1)
        elif "RELU6" in _activation_: output_ = RELUx(output_, 6)
        else: print(_activation_+' not supported')
    assert output_.shape == tensor_output.data.shape
    tensor_output.data = output_
    return output_

