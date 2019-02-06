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

class_names = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

def save_as_txt(data,outfile):
    a1 = data.reshape(-1)
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
    embed_image(resized, boxed, int((w-new_w)//2), int((h-new_h)//2))
    return boxed

def softmax(a):
    exp_a = np.exp( a )
    sum_exp_a = np.sum(exp_a)
    y= exp_a / sum_exp_a
    return y

class DetectionObject:
    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale) 
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence

def EntryIndex(side_w, side_h, lcoords, lclasses, location, entry):
    n   = int(location / (side_w * side_h))
    loc = int(location % (side_w * side_h))
    return n * side_w * side_h * (lcoords + lclasses + 1) + entry * side_w * side_h + loc

def ParseYOLOV2Output(
    output_blob,
    resized_im_h,
    resized_im_w,
    original_im_h,
    original_im_w,
    threshold
):
    num = 5
    coords = 4
    classes=20
    side_h = 13
    side_w = 13

    anchors = [ 
        0.572730, 0.677385, 
        1.874460, 2.062530, 
        3.338430, 5.474340,
        7.882820, 3.527780,
        9.770520, 9.168280,
    ]
    #side = side_h
    side_square = side_h * side_w;
    output_blob = output_blob.astype(dtype=np.float32)

    objects = []
    for i in range(side_square):
        row = int(i/side_w)
        col = int(i%side_w)
        for n in range(num):
            obj_index = EntryIndex(side_h,side_w,coords,classes,n*side_h*side_w+i,coords)
            box_index = EntryIndex(side_h,side_w,coords,classes,n*side_h*side_w+i,0)
            scale = output_blob[obj_index]
            if scale < threshold: continue;

            x = (col + output_blob[box_index + 0 * side_square]) / side_w * original_im_w;
            y = (row + output_blob[box_index + 1 * side_square]) / side_w * original_im_h;
            height = np.exp(
                    output_blob[box_index + 3 * side_square]
                ) * anchors[2 * n + 1] / side_h * original_im_h
            width  = np.exp(
                    output_blob[box_index + 2 * side_square]
                ) * anchors[2 * n + 0] / side_h * original_im_w

            for j in range(classes):
                class_index = EntryIndex(side_h,side_w,coords,classes,n*side_square+i,coords+1+j)
                prob = scale * output_blob[class_index]
                if prob < threshold: continue;
                obj  = DetectionObject(x, y, height, width, j, prob,
                        float(original_im_h) / float(resized_im_h),
                        float(original_im_w) / float(resized_im_w)
                )
                objects.append(obj);
    return objects

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
# VOC YOLOv2 region layer memory layout
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
    original_im_h, original_im_w = frame.shape[:2]

    frame =frame/255.0
    save_as_txt(frame.transpose(2,0,1),"dog_im.txt")

    frame =letterbox_image(frame, model_w, model_h)
    flat_ = frame.transpose(2,0,1).reshape(-1)
    save_as_txt(flat_,"dog_sized.txt")

    #STEP-6
    in_frame = cv2.resize(frame, (model_w, model_h))
    in_frame = in_frame[np.newaxis,:,:,:]
    in_frame = in_frame.transpose((0, 3, 1, 2))  # Change data layout from HWC to CHW
    print("in-frame",in_frame.shape)

    start = time()
    #STEP-7
    ASYNC = False
    if ASYNC:
        exec_net.start_async(request_id=0, inputs={input_blob: in_frame})
    else:
        res = exec_net.infer(inputs={input_blob: in_frame})

    if exec_net.requests[0].wait(-1) == 0:
        if ASYNC:
            res = exec_net.requests[0].outputs[out_blob]
            print("res.shape",res.shape)
            result = res
        else:
            for outkey in res.keys(): print("outkey=",outkey)
            result = res[outkey][0]
            print("result.shape",result.shape)
        save_as_txt(result,"dog_region.txt")
        sec = time()-start
        objects = ParseYOLOV2Output(
            result,
            model_w,
            model_h,
            original_im_h,
            original_im_w,
            0.5
        )
        for i in objects:
            print("%.3f%% %s (%d %d) - (%d %d)"%(
                i.confidence,
                class_names[i.class_id],
                i.xmin, i.ymin,
                i.xmax, i.ymax
            ))
        print("objects len=",len(objects))
        print("elapse = %.3fmsec %.3fFPS"%(1000.0*sec,1.0/sec))
    else:
        print("error")

#STEP-10
del exec_net
del plugin
