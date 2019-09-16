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
#from   fbnnpp import *

_floating_infer = False
dati_dtype      = np.int32

def floating_on(verbose=False):
    global _floating_infer
    _floating_infer = True
    if verbose: print("floating inference On {}".format(_floating_infer))

def read_tflite_model(file):
    buf = open(file, "rb").read()
    buf = bytearray(buf)
    model = Model.GetRootAsModel(buf, 0)
    return model

class operator():
    def __init__(self, operator_idx, operator_fb, operator_codes_fb, tensors):
        self.idx     = operator_idx
        self.Operator= operator_fb
        self.tensors = tensors
        self.inputs  = list( operator_fb.InputsAsNumpy() )
        self.outputs = list( operator_fb.OutputsAsNumpy() )
        self.opcode_index    = operator_fb.OpcodeIndex()
        self.builtin_options = operator_fb.BuiltinOptions()
        self.operator_codes_fb = operator_codes_fb

        self.name    = self.opcode_name = self.BuiltinCode2String(self.opcode_index)
        self.nick    = "{:5s}".format((self.name[:2]+re.sub('[_AIUEO0-9]','',self.name[2:]))[:5])
        self.padding = 0

        self.denomi  = denomiC = None
        if len(self.inputs)==3:
            ( scale_a, max_a, min_a, zero_point_a ) = self.tensors[self.inputs[0]].Quantization_Options()
            ( scale_b, max_b, min_b, zero_point_b ) = self.tensors[self.inputs[1]].Quantization_Options()
            ( scale_c, max_c, min_c, zero_point_c ) = self.tensors[self.inputs[2]].Quantization_Options()
            if scale_a is not None and scale_b is not None:
                self.denomi = dati_dtype((scale_a*scale_b)**-1)
                assert self.denomi > 0,"Invalid Denominator {}".format(self.denomi)
            if scale_c is not None:
                denomiC     = dati_dtype((scale_c)**-1)
                assert self.denomi == denomiC,"Unsupports Denominator {} != {}".format(self.denomi,denomiC)
        elif len(self.inputs)==2 and self.name=='MUL':
            ( scale_a, max_a, min_a, zero_point_a ) = self.tensors[self.inputs[0]].Quantization_Options()
            ( scale_b, max_b, min_b, zero_point_b ) = self.tensors[self.inputs[1]].Quantization_Options()
            if scale_a is not None and scale_b is not None:
                self.denomi = dati_dtype((scale_a*scale_b)**-1)
                assert self.denomi > 0,"operator-{} Invalid Denominator {}(1/({}*{}))".format(self.nick,self.denomi,scale_a,scale_b)

    def Builtin_Options(self, verbose=False):
        def funcno2name(funcno):
            if funcno == 0:return None
            if funcno == 1:return "RELU"
            if funcno == 2:return "RELU1"
            if funcno == 3:return "RELU6"
            if funcno == 4:return "TANH"
            if funcno == 5:return "SIGN_BIT"
            assert False,"Unknown Fused Function"

        op = self.Operator
        option_type = op.BuiltinOptionsType()
        if option_type == tflite.BuiltinOptions.BuiltinOptions.Conv2DOptions:
            opt = tflite.Conv2DOptions.Conv2DOptions()
            opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            padding = opt.Padding()
            strideh = opt.StrideH()
            stridew = opt.StrideW()
            _activation_ = opt.FusedActivationFunction()
            _activation_ = funcno2name(_activation_)
            if verbose: print("Conv2DOptions")
            return (padding, stridew, strideh, _activation_)

        elif option_type == tflite.BuiltinOptions.BuiltinOptions.DepthwiseConv2DOptions:
            opt = tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptions()
            opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            padding = opt.Padding()
            strideh = opt.StrideH()
            stridew = opt.StrideW()
            _activation_ = opt.FusedActivationFunction()
            _activation_ = funcno2name(_activation_)
            depthmultiplier = opt.DepthMultiplier()
            if verbose: print("DepthwiseConv2DOptions")
            return (padding, stridew, strideh, _activation_,depthmultiplier)

        elif option_type == tflite.BuiltinOptions.BuiltinOptions.FullyConnectedOptions:
            opt = tflite.FullyConnectedOptions.FullyConnectedOptions()
            opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            _activation_ = opt.FusedActivationFunction()
            _activation_ = funcno2name(_activation_)
            if verbose: print("FullyConnectedOptions")
            return (_activation_)

        elif option_type == tflite.BuiltinOptions.BuiltinOptions.SoftmaxOptions:
            opt = tflite.SoftmaxOptions.SoftmaxOptions()
            opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            beta = opt.Beta()
            if verbose: print("SoftmaxOptions")
            return (beta)

        elif option_type == tflite.BuiltinOptions.BuiltinOptions.ReshapeOptions:
            opt = tflite.ReshapeOptions.ReshapeOptions()
            opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            newshape = list(opt.NewShapeAsNumpy())
            if verbose: print("ReshapeOptions")
            return (newshape)

        elif option_type == tflite.BuiltinOptions.BuiltinOptions.Pool2DOptions:
            opt = tflite.Pool2DOptions.Pool2DOptions()
            opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            padding = opt.Padding()
            strideh = opt.StrideH()
            stridew = opt.StrideW()
            filterwidth = opt.FilterWidth()
            filterheight = opt.FilterHeight()
            _activation_ = opt.FusedActivationFunction()
            _activation_ = funcno2name(_activation_)
            if verbose: print("Pool2DOptions")
            return (padding, stridew, strideh, _activation_,filterwidth, filterheight)
        else:
            return ()

    def BuiltinCode2String(self, opcode_index):
        builtin_code = self.operator_codes_fb(opcode_index).BuiltinCode()
        custom_code  = self.operator_codes_fb(opcode_index).CustomCode()
        #print("operator code {} builtin_code/custom_code = {}/{}".format(opcode_index,builtin_code,custom_code))
        if builtin_code == tflite.BuiltinOperator.BuiltinOperator.CONCATENATION:
            return "CONCATENATION"
        elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.CONV_2D:
            return "CONV_2D"
        elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.DEPTHWISE_CONV_2D:
            return "DEPTHWISE_CONV_2D"
        elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.FULLY_CONNECTED:
            return "FULLY_CONNECTED"
        elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.LOGISTIC:
            return "LOGISTIC"
        elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.MAX_POOL_2D:
            return "MAX_POOL_2D"
        elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.RELU:
            return "RELU"
        elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.RELU6:
            return "RELU6"
        elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.RESHAPE:
            return "RESHAPE"
        elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.SOFTMAX:
            return "SOFTMAX"
        elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.CUSTOM:
            return "CUSTOM"
        elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.MUL:
            return "MUL"
        elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.MAXIMUM:
            return "MAXIMUM"
        else:
            print("Unknown builtin {} custom {}".format(builtin_code, custom_code))
            return "UNKNOWN{}{}".format(builtin_code, custom_code)

    def unsupported(self):
        print(self.name+" IS NOT SUPPORTED",self.outputs,self.name,self.inputs)
        #sys.exit(-1)

    def fully_connected(self, x, W, b):
        # x : width  x height
        # W : height x width
        # b : height
        y = np.sum(x*W,axis=1)
        z = y+b
        return z

    def clipping(self, tensor_idx):
        return
        tensor = self.tensors[tensor_idx[0]]
        if tensor.min is not None and tensor.max is not None:
            tensor.data = np.clip(tensor.data, tensor.min, tensor.max)
        elif tensor.min is not None:
            tensor.data = np.maximum(tensor.data, tensor.max)
        elif tensor.max is not None:
            tensor.data = np.minimum(tensor.data, tensor.min)

    def eval(self):
        name = self.name
        if   name == 'ADD':     # Untested yet
            r = self.tensors[0].data
            for i in self.inputs[1:]:
                assert self.tensors[0].shape == self.tensors[i].shape,"Unmatch {} {}".format(
                                                    r.shape, self.tensors[i].shape)
                r += self.tensors[i].data
            self.clipping(self.outputs)
            return r
        elif name == 'AVERAGE_POOL_2D':   self.unsupported()
        elif name == 'CONCATENATION':
            _axis  = self.Builtin_Options()
            if _axis_ is None:self.view('Invalid conatenation axis',cont=False)
            temp_ = []
            for t in self.inputs:
                temp_.append(self.tensors[t].data.tolist())
            assert len(temp_) > 0, "Invalid concatenation list"
            r = self.tensors[self.outputs[0]].data = np.concatenate(temp_, axis = _axis_)
            return r
        elif name == 'CONV_2D':
            CONV_2D(self, self.outputs, self.inputs)
            self.clipping(self.outputs)
            return self.tensors[self.outputs[0]].data
        elif name == 'DEPTHWISE_CONV_2D':
            DEPTHWISE_CONV_2D(self, self.outputs, self.inputs)
            self.clipping(self.outputs)
            return self.tensors[self.outputs[0]].data
        elif name == 'EMBEDDING_LOOKUP':  self.unsupported()
        elif name == 'FULLY_CONNECTED':
            x = self.tensors[self.inputs[0]].data.reshape(-1)
            w = self.tensors[self.inputs[1]].data
            b = self.tensors[self.inputs[2]].data
            r = self.fully_connected(x,w,b)
            _activation_ = self.Builtin_Options()
            if _activation_ is not None:
                if   "RELU"  in _activation_: r = RELUx(r, 0)
                elif "RELU1" in _activation_: r = RELUx(r, 1)
                elif "RELU6" in _activation_: r = RELUx(r, 6)
                else: print(_activation_+' not supported')
            self.tensors[self.outputs[0]].data = r
            self.clipping(self.outputs)
            return r
        elif name == 'HASHTABLE_LOOKUP':  self.unsupported()
        elif name == 'L2_NORMALIZATION':  self.unsupported()
        elif name == 'L2_POOL_2D':        self.unsupported()
        elif name == 'LOCAL_RESPONSE_NORMALIZATION': self.unsupported()
        elif name == 'LOGISTIC':
            sigmoid = lambda x : 1 / (1 + np.exp(-x))
            x = self.tensors[self.inputs[0]].data
            r = self.tensors[self.outputs[0]].data = sigmoid(np.clip(x,-100,100))
            return r
        elif name == 'LSH_PROJECTION':    self.unsupported()
        elif name == 'LSTM':              self.unsupported()
        elif name == 'MAX_POOL_2D':
            MAX_POOL_2D(self, self.outputs, self.inputs)
            self.clipping(self.outputs)
            return self.tensors[self.outputs[0]].data
        elif name == 'RELU':
            x = self.tensors[self.inputs[0]].data
            r = self.tensors[self.outputs[0]].data = RELUx(x, 0)
            return r
        elif name == 'RELU6':
            x = self.tensors[self.inputs[0]].data
            r = self.tensors[self.outputs[0]].data = RELUx(x, 6)
            return r
        elif name == 'RESHAPE':
            s = self.Builtin_Options()
            x = self.tensors[self.inputs[0]].data
            if s is None: s = self.tensors[self.inputs[1]].data
            r = self.tensors[self.outputs[0]].data = x.reshape(tuple(s))
            return r
        elif name == 'RESIZE_BILINEAR':   self.unsupported()
        elif name == 'RNN':               self.unsupported()
        elif name == 'SOFTMAX':
            assert len(self.inputs) == 1, "SOFTMAX not support dim {}".format(self.inputs)
            beta = self.Builtin_Options()
            assert beta != 0, "SOFTMAX not support beta {}".format(beta)
            # x  = np.exp(self.tensors[self.inputs[0]].data - np.max(self.tensors[self.inputs[0]].data))
            input_tensor = self.tensors[self.inputs[0]]
            x  = np.exp(beta*(input_tensor.data - np.max(input_tensor.data)))
            r  = self.tensors[self.outputs[0]].data = x/np.sum(x)
            return r
        elif name == 'SPACE_TO_DEPTH':    self.unsupported()
        elif name == 'SVDF':              self.unsupported()
        elif name == 'TANH':              self.unsupported()
        elif name == 'CONCAT_EMBEDDINGS': self.unsupported()
        elif name == 'SKIP_GRAM':         self.unsupported()
        elif name == 'CALL':              self.unsupported()
        elif name == 'CUSTOM':            self.unsupported()
        elif name == 'MUL':               # 18 additional support for schema_v3.fbs
            a = self.tensors[self.inputs[0]].data
            x = self.tensors[self.inputs[1]].data
            r = self.tensors[self.outputs[0]].data = a * x
            assert _floating_infer,"Quantization inference now, `MUL` supports Floating inference only {}".format(a)
            return r
        elif name == 'MAXIMUM':           # 55 additional support for schema_v3.fbs
            x0= self.tensors[self.inputs[0]].data
            x1= self.tensors[self.inputs[1]].data
            r = self.tensors[self.outputs[0]].data = np.maximum(x0, x1)
            return r
        else:                             self.unsupported()

    def view(self, msg=None, cont=True):
        if msg is not None: print("\n***\n*** "+msg+"\n***")
        print("operator[{}]({}:{}) outputs {} inpus {}".format(self.idx, self.nick, self.opcode_index, self.outputs, self.inputs))
        #print("  builtin_options : {} padding@run {}".format(self.builtin_options, self.padding))
        print("  builtin_options : {} padding@run {}".format(self.Builtin_Options(), self.padding))
        for o in self.outputs: self.tensors[o].view()
        for i in self.inputs:  self.tensors[i].view()
        assert cont,"Fatal Error occurrence at operator"

class tensor():
    def __init__(self, tensor_idx, tensor_fb, buffers_fb):
        global _floating_infer
        self.idx    = tensor_idx
        self.Tensor = tensor_fb
        self.shape  = list(tensor_fb.ShapeAsNumpy())
        self.type   = self.TensorType2String(tensor_fb.Type())
        self.name   = tensor_fb.Name()
        self.buffer = tensor_fb.Buffer()
        self.warn_convert = 1

        assert self.buffer>=0,"Invalid tensor.Buffer() {}".format(self.buffer)
        if self.type   == 'FLOAT32': dtype_string = 'f4'
        elif self.type == 'FLOAT16': dtype_string = 'f2'
        elif self.type == 'INT32'  : dtype_string = 'i4'
        elif self.type == 'INT64'  : dtype_string = 'i8'
        else                       : dtype_string = 'u1'    # unsigned integer 1Byte
        self.buff = buffers_fb[self.buffer].DataAsNumpy()
        self.quantization = tensor_fb.Quantization()
        assert type(self.quantization) == tflite.QuantizationParameters.QuantizationParameters
        self.scale = self.max = self.min = self.zero_point = None

        self.scale      = self.quantization.ScaleAsNumpy()     if self.quantization.ScaleLength()    > 0 else None
        self.max        = self.quantization.MaxAsNumpy()       if self.quantization.MaxLength()      > 0 else None
        self.min        = self.quantization.MinAsNumpy()       if self.quantization.MinLength()      > 0 else None
        self.zero_point = self.quantization.ZeroPointAsNumpy() if self.quantization.ZeroPointLength()> 0 else None

        # Initialize data, dati
        if buffers_fb[self.buffer].DataLength()>0:
            self.data = self.buff.view(dtype=dtype_string).reshape(self.shape)     # Ultra fast!
            self.dati = self.data.astype(dati_dtype).copy()
        #    if self.zero_point is not None: self.dati = self.data.astype(np.int32) - np.int32(self.zero_point)
        else:
            self.data = np.zeros(tuple(self.shape),dtype=self.type2np(self.type))
            self.dati = np.zeros(tuple(self.shape),dtype=dati_dtype)

        # Initialize scale, max, min, zero_point
        if self.scale is not None:
            assert len(self.scale) == 1,"Json format error len(scale)="+str(len(self.scale))
            self.scale = self.scale[0]
        elif self.max is not None or self.min is not None or self.zero_point is not None:
            self.scale = 1.0

        if self.max is not None:
            assert len(self.max) == 1,"Json format error len(max)="+str(len(self.max))
            self.max = self.max[0]

        if self.zero_point is not None:
            assert len(self.zero_point) == 1,"Json format error len(zero_point)="+str(len(self.zero_point))
            self.zero_point = self.zero_point[0]

        # Convert to float
        if self.min is not None:
            self.data  = (self.scale * self.data + self.min).astype(np.float32)
            self.min   = self.min[0]
            sys.stdout.write("convert tensor-{:<3d} {} to float by self.min {:8.3f}".format(self.idx,self.type,self.min))

        elif self.zero_point is not None:
            self.min   =  self.scale * self.zero_point
            self.data  = (self.scale * (self.data.astype(np.int32) - self.zero_point)).astype(np.float32)
            sys.stdout.write("convert tensor-{:<3d} {} to float by self.zero_point {}".format(self.idx,self.type,self.zero_point))

        # Offseting dati
        if self.zero_point is not None:
            sys.stdout.write(" dati offset by self.zero_point {:4d}".format(self.zero_point))
            self.dati  = self.dati.astype(dati_dtype) - dati_dtype(self.zero_point)
        if self.min is not None or self.zero_point is not None :sys.stdout.write('\n')
 
        # Targetting
        if not _floating_infer:
            self.data=self.dati.copy()

    def TensorType2String(self, TensorType):
        if   TensorType == tflite.TensorType.TensorType.FLOAT32: return "FLOAT32"
        elif TensorType == tflite.TensorType.TensorType.FLOAT16: return "FLOAT16"
        elif TensorType == tflite.TensorType.TensorType.INT32:   return "INT32"
        elif TensorType == tflite.TensorType.TensorType.UINT8:   return "UINT8"
        elif TensorType == tflite.TensorType.TensorType.INT64:   return "INT64"
        elif TensorType == tflite.TensorType.TensorType.STRING:  return "STRING"
        else: assert False,"Unknown:TensorType2String(TensorType)"+str(TensorType)

    def Quantization_Options(self, verbose=False):
        return ( self.scale, self.max, self.min, self.zero_point )

    def type2np(self,type_string):
        if type_string == 'FLOAT32': return np.float32
        if type_string == 'FLOAT16': return np.float16
        if type_string == 'INT32': return np.int32
        if type_string == 'INT64': return np.int64
        if type_string == 'UINT8': return np.uint8
        return np.float

    # Notice! when inference-type FLOAT,
    #   input_data = (np.float32(input_data) - args.input_mean) / args.input_std
    #   reference : https://github.com/raymond-li/tflite_tensor_outputter/blob/master/tflite_tensor_outputter.py
    #
    def set(self, img, verbose=False):
        global _floating_infer
        # If input-type QUINT8 and inference-typ FLOAT then converter generates DEQUANT operator
        assert type(img) == np.ndarray,"Input image type must be numpy.ndarray but got "+str(type(img))
        #assert img.dtype == self.type2np(self.type),"Cannot set tensor: expect {} but {}".format(self.type,img.dtype)
        self.buff = img
        if verbose: print("set buff tensor range max/min/mean ={}/{}/{:.3f} type {}".format(img.max(), img.min(), img.mean(), img.dtype))
        if self.type == 'UINT8':
            # 0 - 255 : range of dati
            self.dati = img.astype(dati_dtype).copy() # Don't care zero_point of a input tensor!
        else:
            self.dati = img.copy()
        if verbose: print("set dati tensor range max/min/mean ={}/{}/{:.3f} type {}".format(self.dati.max(),self.dati.min(),self.dati.mean(),self.dati.dtype))
    #    if (self.max < img.max() or self.min > img.min()):
    #        if self.warn_convert>0:
    #            print("Warning: Suppots float32 only so converting input {} to float32".format(img.dtype))
    #            self.warn_convert = 0
    #        img = ( self.scale * img + self.min ).astype(np.float32)
        if _floating_infer:
            self.data = img.copy()
        else:
            self.data = self.dati
        if verbose: print("set data tensor range max/min/mean ={:.3f}/{:.3f}/{:.3f} type {}".format(self.data.max(),self.data.min(),self.data.mean(),self.data.dtype))
        return self.data

    def view(self, msg=None, cont=True):
        if msg is not None: print("\n***\n*** "+msg+"\n***")
        print("tensors[{}]({}) buffer:{}".format(self.idx, self.name, self.buffer))
        print("  type@tflite :{} type@run :{}".format(self.type,self.data.dtype))
        print("  shape@tflite:{} shape@run:{}".format(self.shape, self.data.shape))
        print("  quantization:min/max/scale/zerop {} {} {} {}".format(self.min, self.max, self.scale,self.zero_point))
        print("  dati         min/max/mean        {} {} {:.3f}".format(self.dati.min(),self.dati.max(),self.dati.mean()))
        print("  data         min/max/mean        {:.3f} {:.3f} {:.3f}".format(self.data.min(),self.data.max(),self.data.mean()))
        assert cont,"Fatal Error occurrence at tensor"

class graph:
    def __init__(self, tflite='mnist.tflite', floating=False, verbose=False):
        if floating: floating_on(verbose=verbose)
        self.model    = read_tflite_model(tflite)
        self.subgraph = self.model.Subgraphs(0)
        self.inputs   = list(self.subgraph.InputsAsNumpy())
        self.outputs  = list(self.subgraph.OutputsAsNumpy())
        buffers_fb    = [ self.model.Buffers(b) for b in range(self.model.BuffersLength()) ]

        if verbose: print("Creating tensors structure ..")
        self.tensors  = []
        for idx in range(self.subgraph.TensorsLength()):
            tensor_fb = self.subgraph.Tensors(idx)
            gtnsr = tensor(idx, tensor_fb, buffers_fb)
            self.tensors.append(gtnsr)

        self.operators = []
        if verbose: print("Creating operators structure ..")
        for idx in range(self.subgraph.OperatorsLength()):
            operator_fb = self.subgraph.Operators(idx)
            oprtr = operator(idx, operator_fb, self.model.OperatorCodes, self.tensors)
            self.operators.append(oprtr)

        self.reset_refs()
        self.operate_order_list     = []
        if verbose: print("Creating Graph done.")

    def ialpha(self, y, a, b, c):
        ( scale_y, max_y, min_y, zero_point_y ) = self.tensors[y].Quantization_Options()
        ( scale_a, max_a, min_a, zero_point_a ) = self.tensors[a].Quantization_Options()
        ( scale_b, max_b, min_b, zero_point_b ) = self.tensors[b].Quantization_Options()
        ( scale_c, max_c, min_c, zero_point_c ) = self.tensors[c].Quantization_Options()
        return scale_a*scale_b/scale_y, scale_c/scale_y, scale_y, scale_a, scale_b, scale_c

    def reset_refs(self): self.operator_refs = [0]*len(self.operators)

    def refs(self, operator_idx):
        refs = self.operator_refs[operator_idx]
        self.operator_refs[operator_idx] += 1
        return refs

    def generators(self, tensors):
        # find subgraph input operator index
        output_operators_idxes = []
        source_tensor_idxes    = []
        assert type(tensors) == list,"tensors(index) must be list"
        for ope_idx, ope in enumerate(self.operators):
            ope_outs = ope.outputs
            assert type(ope_outs) == list,"ope_outs(index) must be list"
            if len(set(ope_outs+tensors)) != len(ope_outs+tensors):
                output_operators_idxes.append(ope_idx)
                source_tensor_idxes.append(ope.inputs)
        return output_operators_idxes, source_tensor_idxes

    def consumers(self, tensors):
        # find subgraph output operator index
        input_operators_idxes = []
        destin_tensor_idxes   = []
        for ope_idx, ope in enumerate(self.operators):
            ope_ins = ope.inputs
            if len(set(ope_ins+tensors)) != len(ope_ins+tensors):
                input_operators_idxes.append(ope_idx)
                destin_tensor_idxes.append(ope.outputs)
        return input_operators_idxes, destin_tensor_idxes

    def print_operator(self, operator_idx):
        opcode = self.operators[operator_idx].opcode_index
        o_obj  = self.operators[operator_idx]
        o_nick = self.operators[operator_idx].nick
        print("dest_tensor {} {} {:16s} <= operator {} {:3d}(code {:2d}) = src {}".format(
                o_obj.outputs,
                self.tensors[o_obj.outputs[0]].type,
                self.tensors[o_obj.outputs[0]].name,
                o_nick,
                operator_idx,
                opcode,
                o_obj.inputs
            )
        )

    #     Generators      Focus        Consumers
    #Tensor ---- ope ---+ Tensor +---- ope --- Tensor
    #List               | List   |             List
    #                   |        |
    #Tensor ____ ope ___|        |____ ope --- Tensor
    #List                                      List
    def walk_from(self, tensor_idx, verbose=True):
        operators, tensors = self.generators(tensor_idx)
        for o, t in zip(operators, tensors):
            if self.refs(o)>0:continue
            self.walk_from(t, verbose)
            self.operate_order_list.append(o)
            if verbose: self.print_operator(o)

    def allocate_graph(self, verbose=True):
        if verbose: print("Allocatng Graph ..")
        self.reset_refs()
        self.operate_order_list     = []
        self.walk_from(self.outputs, verbose=verbose)
        for order, operator_idx in enumerate(self.operate_order_list):
            pass
        if verbose: print("Allocatng Graph done.")

    def invoke(self, verbose=False):
        global _floating_infer
        if verbose: print("----- INVOKING      -----")
        for order, operator_idx in enumerate(self.operate_order_list):
            operator = self.operators[operator_idx]
            #for i in self.inputs:   # Check only
            #    input_ = self.tensors[i]
            #    assert tuple(input_.shape)==input_.data.shape,"Input shape mismatch {} {}".format(
            #            self.tensors[i].shape, self.tensors[i].data.shape)
            ans = operator.eval()
            tensor_output = self.tensors[operator.outputs[0]]
            if not _floating_infer and operator.denomi is not None:
                if verbose: print(tensor_output.data.max(), operator.denomi)
                tensor_output.data = dati_dtype( tensor_output.data / operator.denomi )   # To avoid dati_dtype overflow
            if verbose: operator.view()
        if verbose: print("----- DONE --------------")
        return ans

if __name__=='__main__':
    import argparse
    args = argparse.ArgumentParser()
    def chF(f): return f if os.path.exists(f) else sys.exit(-1)
    args.add_argument('-t',"--tflite",       type=chF, default='mnist.tflite')
    args.add_argument('-i',"--images",       type=int, default=1)
    args.add_argument('-q',"--quantization", action='store_true')
    args.add_argument('-i16',"--int16",      action='store_true')
    args.add_argument('-v',"--verbose",      action='store_true')
    args = args.parse_args()
    if args.quantization:
        print("Inference with UINT8 Quantization")
    else:
        print("Inference with Default type")
        _floating_infer = True
    if args.int16:
        dati_dtype = np.int16

    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    g = graph(tflite=args.tflite, verbose=args.verbose)
    g.allocate_graph(verbose=True)

    corrects = 0
    for i in range(args.images):
        
        number_img = mnist.test.images[i]
        number_gt  = mnist.test.labels[i]
        # input-type inference-type
        # uint8      uint8           no-convert
        # uint8      float           convert
        # float      uint8           NG
        # float      float           no-convert
        if args.quantization:
            assert g.tensors[g.inputs[0]].type == 'UINT8',"-q {} but input {}".format(args.quantization, g.tensors[g.inputs[0]].type)
            g.tensors[g.inputs[0]].set((255*number_img[np.newaxis,:]).astype(np.uint8))
        else:
            g.tensors[g.inputs[0]].set(number_img[np.newaxis,:].astype(np.float32))
        y = g.invoke(verbose=False)
        gt = np.argmax(number_gt)
        pr = np.argmax(y)
        if gt!=pr:
            print("{:5d} incorrenct:gt-{} pr-{}".format(i,gt,pr))
        else:
            corrects+=1

    print("accurracy %.3f %d/%d"%(1.0*corrects/args.images,corrects,args.images))

