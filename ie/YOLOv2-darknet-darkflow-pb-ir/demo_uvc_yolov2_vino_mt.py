#!/usr/bin/env python3
from pdb import *
import sys,os
assert sys.version_info.major >= 3,'Use python3'
import numpy as np
from time import time,sleep
import cv2
import argparse
import itertools as itt
#from postscript import *
from openvino.inference_engine import IENetwork, IEPlugin
import multiprocessing as mp
import queue
import threading
import heapq
import signal

num = 5
coords = 4
downsampling_rate = 32

coco_class_names = [
    "person", "bicycle", "car", "motorbike", "aeroplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
coco_anchors = [ 
    0.572730, 0.677385, 
    1.874460, 2.062530, 
    3.338430, 5.474340,
    7.882820, 3.527780,
    9.770520, 9.168280,
]
c1_class_names = [
    "person",
]
c1_anchors = [ 
    1.3221, 1.73145,
    3.19275, 4.00944,
    5.05587, 8.09892,
    9.47112, 4.84053,
    11.2364, 10.0071
]
c2_class_names = [
    "chair", "person",
]
c2_anchors = [ 
    1.3221, 1.73145,
    3.19275, 4.00944,
    5.05587, 8.09892,
    9.47112, 4.84053,
    11.2364, 10.0071
]
voc_class_names = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]
voc_anchors = [ 
    1.3221, 1.73145,
    3.19275, 4.00944,
    5.05587, 8.09892,
    9.47112, 4.84053,
    11.2364, 10.0071
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

def IntersectionOverUnion(box_1, box_2):
    width_of_overlap_area  = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin);
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    return area_of_overlap / area_of_union

class DetectionObject:
    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale) 
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence

def softmax(result, n, temp, stride, output):
    largest = -10000.0
    sums = 0.0
    for i in range(n):
        if result[i*stride] > largest:largest = result[i*stride]
    for i in range(n):
        e = np.exp(result[i*stride]/temp - largest/temp)
        sums += e
        output[i*stride] = e
    for i in range(n):
        output[i*stride] /= sums

def softmax_cpu(result,n,batch,batch_offset,groups,group_offset,stride,temp,output):
    for b in range(batch):
        for g in range(groups):
            res_out = softmax(
                result[(b*batch_offset+g*group_offset):],
                n,
                temp,
                stride,
                output[(b*batch_offset+g*group_offset):]
            )

def EntryIndex(side_w, side_h, lcoords, lclasses, location, entry):
    n   = int(location / (side_w * side_h))
    loc = int(location % (side_w * side_h))
    return n * side_w * side_h * (lcoords + lclasses + 1) + entry * side_w * side_h + loc

def parse_result(
    output_blob,
    resized_im_h,
    resized_im_w,
    original_im_h,
    original_im_w,
    threshold
):
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

            inferred_x = output_blob[ box_index + 0 * side_square ]
            inferred_y = output_blob[ box_index + 1 * side_square ]
            inferred_w = output_blob[ box_index + 2 * side_square ]
            inferred_h = output_blob[ box_index + 3 * side_square ]
            x = ( col + inferred_x ) / side_w * original_im_w;
            y = ( row + inferred_y ) / side_h * original_im_h;
            height = np.exp( inferred_h ) * anchors[2*n+1] / side_h * original_im_h
            width  = np.exp( inferred_w ) * anchors[2*n+0] / side_h * original_im_w

            for j in range(classes):
                class_index = EntryIndex(side_h,side_w,coords,classes,n*side_square+i,coords+1+j)
                prob = scale * output_blob[class_index]
                if prob < threshold: continue;
                obj  = DetectionObject(x, y, height, width, j, prob,
                        float(resized_im_h) / float(original_im_h),
                        float(resized_im_w) / float(original_im_w)
                )
                objects.append(obj);
    return objects

def overlay_objects(image, objects):
    for obj in objects:
        confidence = obj.confidence
        if obj.confidence <= 0.0: continue  # skip object rejected by NMS
        label = obj.class_id
        p1 = (obj.xmin, obj.ymin)
        p2 = (obj.xmax, obj.ymax)
        cv2.rectangle(image, p1, p2, (0, 0, 255),min(int(image.shape[2]//100),5))
        cv2.putText(image,class_names[label],p1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

def keep_aspect(image, new_h, new_w):
    org_h, org_w = image.shape[:2]
    if org_h > org_w:
        w = int(new_w * org_h/org_w)
        h = new_h
    else:
        w = new_w
        h = int(new_h * org_w/org_w)
    return cv2.resize(image,(w,h))

def check_classes(classes_set):
    classes_mode = 20
    if classes_set != 'voc':
        model_bin=data_type+'/yolov2-'+classes_set+'.bin'
        model_xml=data_type+'/yolov2-'+classes_set+'.xml'
        if not os.path.exists(model_bin) or not os.path.exists(model_xml):
            print("not found model", os.path.splitext(model_bin)[0])
            sys.exit(-1)
        if classes_set == 'coco': classes_mode=80
        if classes_set == '2c'  : classes_mode=2
        if classes_set == '1c'  : classes_mode=1
        return True, model_bin, model_xml, classes_mode
    return False, '', '', classes_mode

class Myriad():

    def __init__(self, model_xml, model_bin, device, requests, in_frameQ, resultQ):
        print(model_bin,"on",device,"acceptable requests",requests)
        self.bq = 0
        self.model_xml  = model_xml
        self.model_bin  = model_bin
        self.requests   = requests
        self.device     = device
        self.in_frameQ  = in_frameQ
        self.resultQ    = resultQ
        self.net        = IENetwork(model=model_xml, weights=model_bin)
        self.requests   = requests
        self.plugin     = IEPlugin(device=device, plugin_dirs=None)
        self.exec_net   = self.plugin.load(network=self.net, num_requests=requests)
        self.input_blob = next(iter(self.net.inputs))  #input_blob = 'data'
        self.out_blob   = next(iter(self.net.outputs)) #out_blob   = 'detection_out'
        self.model_form = self.net.inputs[self.input_blob].shape # NCHW
        self.reqlist    = [0]*requests
        self.reqhistory = []

    def predict(self):
        if self.in_frameQ.empty():return
        in_frame  = self.in_frameQ.get()
        try:
            empty_idx = self.reqlist.index(0)
        except:
            empty_idx = -1
        if empty_idx != -1:
            #print("empty_idx",empty_idx)
            self.exec_net.start_async(request_id=empty_idx, inputs={self.input_blob: in_frame})
            self.reqlist[empty_idx] = 1
            heapq.heappush(self.reqhistory,(int(1000*time()),empty_idx))

        (t, idx)= heapq.heappop(self.reqhistory)
        if self.exec_net.requests[idx].wait(0) == 0:
            self.exec_net.requests[idx].wait(-1)
            res = self.exec_net.requests[idx].outputs[self.out_blob]
            result = res.reshape(-1)
            self.resultQ.put(result)
            self.reqlist[idx] = 0
        else:
            heapq.heappush(self.reqhistory,(t,idx))

    def close(self):
        del self.exec_net
        del self.plugin

def infer_thread(myriad):
    while True:
        myriad.predict()

def infer(model_xml, model_bin, ncs_devices, requests, in_frameQ, resultQ):
    infer_threads=[]
    for i in range(ncs_devices):
        myriad   = Myriad(model_xml,model_bin,"MYRIAD",requests,in_frameQ,resultQ)
        infer_th = threading.Thread(target=infer_thread,args=(myriad,))
        infer_th.start()
        infer_threads.append(infer_th)
    for thr in infer_threads: thr.join()

def camera(frameQ):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():sys.exit(-1)
#    print("Opened UVC-Camera via /dev/video0",model_w,model_h,"camera-in")
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    print("camera start")
    while True:
        if frameQ.full(): frameQ.get()
        r,frame = cap.read()
        frameQ.put(frame)
    cap.close()

args = argparse.ArgumentParser()
args.add_argument("-d", "--device"   , type=str, default="MYRIAD", help="MYRIAD/CPU")
args.add_argument("-c", "--classes",   type=str, default='1c',     help="dataset voc/2c/1c/coco")
args.add_argument("-n", "--ncs",       type=int, default=1,        help="using device num")
args.add_argument("-r", "--req",       type=int, default=2,        help="using request num")
args.add_argument("-s", "--softmax",   action="store_true",        help="aplly softmax")
args = args.parse_args()

if args.softmax:print("Aplly softmax")

data_type="FP16"
if args.device == "CPU": data_type="FP32"

model_xml=data_type+'/yolov2.xml'
model_bin=data_type+'/yolov2.bin'
ret, _bin, _xml, classes_mode = check_classes(args.classes)
if ret: model_xml, model_bin = _xml, _bin
if classes_mode == 80:class_names = coco_class_names
if classes_mode == 20:class_names = voc_class_names
if classes_mode ==  2:class_names = c2_class_names
if classes_mode ==  1:class_names = c1_class_names
if classes_mode == 80:anchors     = coco_anchors
if classes_mode == 20:anchors     = voc_anchors
if classes_mode ==  2:anchors     = c2_anchors
if classes_mode ==  1:anchors     = c1_anchors

# net = IENetwork(model=model_xml, weights=model_bin)

classes=len(class_names)

print("num/coods/classes/downsampling",num,coords,classes,downsampling_rate)

thresh_conf=0.69 # if 0.69 then can detect motorbike but 0.60 then detect person instead of motorbike
thresh_conf=0.60 # But in YOLO-OpenVINO/YOLOv2/main.cpp thresh_conf is 0.5
#thresh_conf=0.50
thresh_iou =0.45

#print(model_bin, "on", args.device)
#plugin = IEPlugin(device=args.device, plugin_dirs=None)
#if args.device == "CPU":
#    HOME = os.environ['HOME']
#    PATHLIBEXTENSION = os.getenv(
#        "PATHLIBEXTENSION",
#        HOME+"/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so"
#    )
#    plugin.add_cpu_extension(PATHLIBEXTENSION)

#buffsize=args.requests
#exec_net = plugin.load(network=net, num_requests=buffsize)

#input_blob = next(iter(net.inputs))  #input_blob = 'data'
#out_blob   = next(iter(net.outputs)) #out_blob   = 'detection_out'
#model_n, model_c, model_h, model_w = net.inputs[input_blob].shape #Tool kit R4

model_n, model_c, model_h, model_w = (1, 3, 480, 640)

#print("n/c/h/w (from xml)= %d %d %d %d"%(model_n, model_c, model_h, model_w))
#print("input_blob : out_blob =[",input_blob,":",out_blob,"] ",net.outputs[out_blob].shape)

#del net

side_h = int(model_h//downsampling_rate)
side_w = int(model_w//downsampling_rate)
files=[]

# CAMERA Process
frameQ = mp.Queue(10)
camproc= mp.Process(
    target=camera,
    args=(
        frameQ,
    ),
    daemon=True
) 
camproc.start()

# DEVICEs Process
in_frameQ = mp.Queue(10)
resultQ   = mp.Queue(10)
myrproc = mp.Process(
    target=infer,
    args=(
        model_xml, model_bin, args.ncs, args.req, in_frameQ, resultQ
    ),
    daemon=True
)
myrproc.start()

sec=count_cam=count_inf=1
exit_code=False
start = time()
latest_result = np.zeros((9000),dtype=np.float32)
while True:
    frame = frameQ.get() # HWC
    draw_img = keep_aspect(frame, model_h, model_w)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    original_im_h, original_im_w = frame.shape[:2]
    frame =frame/255.0
#    frame =letterbox_image(frame, model_w, model_h)    # Too slow

    in_frame = frame[np.newaxis,:,:,:]        # Add new axis as a batch dimension HWC NHWC
    in_frame = in_frame.transpose((0, 3, 1, 2))  # Change data layout from NHWC to NCHW

    if in_frameQ.full():in_frameQ.get()
    in_frameQ.put(in_frame.copy())

    try:
        result = resultQ.get_nowait()
        latest_result = result
        count_inf+=1
    except queue.Empty:
        result = latest_result

    sec =(time()-start)
    count_cam+=1

    # Apply softmax instead of Region layer
    if args.softmax:
        index = EntryIndex(side_h, side_w, 4, 0, 0, coords + 1)
        softmax_cpu(
            result[index:],
            classes,
            num,
            int((side_h*side_w*num*(coords+1)+side_h*side_w*num*classes)/num),
            side_w*side_h,
            1,
            side_w*side_h,
            1,
            result[index:]
        )

    # Pull objects from result
    objects = parse_result(
        result,
        model_h,
        model_w,
        original_im_h,
        original_im_w,
        thresh_conf
    )
    condidates = len(objects)

    # NMS
    for i, obj1 in enumerate(objects):
        if obj1.confidence <= 0: continue
        for j, obj2 in enumerate(objects):
            if j<=i:continue
            if IntersectionOverUnion(obj1, obj2) >= thresh_iou:
                objects[i].confidence = 0.0
    # Draw
    overlay_objects(draw_img , objects)

    #show result image
    cv2.imshow('YOLOv2_demo',draw_img)
    key=cv2.waitKey(1)
    if key!=-1:
        if key==27: exit_code=True
    if exit_code:break

    sys.stdout.write('\b'*40)
    sys.stdout.write('%9.5fFPS(%9.5f Playback)'%(count_inf/sec,count_cam/sec))
    sys.stdout.flush()

print("\nfinalizing")
cv2.destroyAllWindows()
camproc.terminate()
myrproc.terminate()
#del exec_net
#del plugin

