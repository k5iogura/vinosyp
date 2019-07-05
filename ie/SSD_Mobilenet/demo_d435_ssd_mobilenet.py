#!/usr/bin/env python3
import argparse
from pdb import *
import sys,os
import pyrealsense2 as rs
import cv2
import numpy as np
from random import randrange, seed
from time import time, sleep
from openvino.inference_engine import IENetwork, IEPlugin

class D435:
      def __init__(self, color=True, depth=False, w=640, h=480, fps=30):
          self.color    = color
          self.depth    = depth
          self.pipeline = rs.pipeline()
          config        = rs.config()
          self.align_to = rs.stream.color
          self.align    = rs.align(self.align_to)
          if color: config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
          if depth: config.enable_stream(rs.stream.depth, w, h, rs.format.z16,  fps)
          profile = self.pipeline.start(config)
          self.scale    = profile.get_device().first_depth_sensor().get_depth_scale()
          print("depth_sensor_scale:",self.scale)

      def read(self):
          color_frame = depth_frame = depth_array = None
          frame = self.pipeline.wait_for_frames()
          align_frame = self.align.process(frame)
          if self.color:
              color_frame = frame.get_color_frame()   # no needs align.process for color stream
              color_frame = np.asanyarray(color_frame.get_data())
          if self.depth:
              depth_frame = align_frame.get_depth_frame()
              depth_frame1= depth_frame.as_depth_frame()
              depth_array = np.asanyarray(depth_frame.get_data())
          return color_frame, depth_frame, depth_array

      def release(self):
          self.pipeline.stop()

def make_meters(dth_np, box, color, cam_scale, seg):
    ratio_w = dth_np.shape[1]
    ratio_h = dth_np.shape[0]
    rect_lt = (int( ratio_w * box[0] ), int( ratio_h * box[1] ))    # coord Camera-out
    rect_rb = (int( ratio_w * box[2] ), int( ratio_h * box[3] ))
    rect_xy = (int(rect_lt[0]+(rect_rb[0] - rect_lt[0])/2),int(rect_lt[1]+(rect_rb[1] - rect_lt[1])/2))
    dth_obj_m = dth_np[rect_lt[1]:rect_rb[1], rect_lt[0]:rect_rb[0]]*cam_scale

    # Search distance until nearest object
    dth_obj_m = np.clip(dth_obj_m, 0.001, 15.000) # histogram of meter wise
    bins, range_m = np.histogram(dth_obj_m, bins=15)
    index_floor = np.argmax(bins)                 # range which occupy most area in bbox
    range_floor = range_m[index_floor]
    indexYX = np.where((dth_obj_m>range_floor))
    if len(indexYX[0]) == 0 and len(indexYX[1]) == 0:return 0.
    meters  = dth_obj_m[indexYX].min()

    # Create meter wise histogram from nearest object to limit of D435
    dth_obj_m = np.clip(dth_obj_m, meters, meters+15.000)
    bins, range_m = np.histogram(dth_obj_m, bins=15)
    index_floor = np.argmax(bins)                 # range which occupy most area in bbox
    range_floor = range_m[index_floor]
    range_ceil  = range_m[index_floor+1]
    indexYX = np.where((dth_obj_m>range_floor) & (dth_obj_m<range_ceil))
    if len(indexYX[0]) == 0 and len(indexYX[1]) == 0:return 0.
    meters  = dth_obj_m[indexYX].min()

    indexYX = (indexYX[0]+rect_lt[1], indexYX[1]+rect_lt[0])
    seg[indexYX[0], indexYX[1], :] = color
    return meters

seed(2222)
LABELS = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')
COLORS = [[ randrange(0,255), 64, randrange(0,255)] for i in range(0,len(LABELS))]

def overlay_on_image(display_image, object_info, dth_np, cam_scale):

    # the minimal score for a box to be shown
    min_score_percent = 25

    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent): return

    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)


    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    if dth_np is None:
        label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"
        label_text = "%s(%d%%)"%(LABELS[int(class_id)],percentage)
    else:
        box_color = COLORS[int(class_id)]  # segment color
        meters = make_meters(dth_np, object_info[3:7], box_color, cam_scale, display_image)
        label_text = "%s(%.2fm)"%(LABELS[int(class_id)], meters)

    label_background_color = (125, 175, 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

args = argparse.ArgumentParser()
args.add_argument("-d","--device", type=str, default='MYRIAD',help='on Device')
args.add_argument("-c","--color",  action="store_true")
args = args.parse_args()

input_image_size=(300,300)

precision='FP32'
if args.device=='MYRIAD':precision='FP16'
model_xml='vinosyp/models/SSD_Mobilenet/'+precision+'/MobileNetSSD_deploy.xml'
model_bin='vinosyp/models/SSD_Mobilenet/'+precision+'/MobileNetSSD_deploy.bin'
model_xml = os.environ['HOME'] + "/" + model_xml
model_bin = os.environ['HOME'] + "/" + model_bin
net = IENetwork(model=model_xml, weights=model_bin)	#R5

plugin = IEPlugin(device=args.device, plugin_dirs=None)
if args.device=='CPU':
    HOME=os.environ['HOME']
    LIBCPU_EXTENSION = HOME+"/inference_engine_samples/intel64/Release/lib/libcpu_extension.so"
    if not os.path.exists(LIBCPU_EXTENSION):
        print('run sample such as demo_squeezenet_download_convert_run.sh')
        sys.exit(-1)
    PATHLIBEXTENSION = os.getenv(
        "PATHLIBEXTENSION",
        LIBCPU_EXTENSION
    )
    plugin.add_cpu_extension(PATHLIBEXTENSION)
exec_net = plugin.load(network=net, num_requests=1)

input_blob = next(iter(net.inputs))  #input_blob = 'data'
out_blob   = next(iter(net.outputs)) #out_blob   = 'detection_out'
model_n, model_c, model_h, model_w = net.inputs[input_blob].shape #Tool kit R4
print("n/c/h/w (from xml)= %d %d %d %d"%(model_n, model_c, model_h, model_w))
print("input_blob : out_blob =",input_blob,":",out_blob)

del net

cameraNo=0
cap = D435(color=True, depth=not args.color)
print("Opened D435-Camera via /dev/video0")
print("LABELES:",len(LABELS))

start = time()
done_frame=0
mask = None
while True:
    frame_org, _, dth_np = cap.read()
    frame = cv2.resize(frame_org,input_image_size).astype(dtype=np.float)
    frame-= 127.5       # means
    frame*= 0.007853    # scale

    if mask is None: mask = np.zeros(frame_org.shape,dtype=np.uint8)

    in_frame = cv2.resize(frame, (model_w, model_h))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((model_n, model_c, model_h, model_w))

    exec_net.start_async(request_id=0, inputs={input_blob: in_frame})

    mask = (mask/2).astype(np.uint8)
    if exec_net.requests[0].wait(-1) == 0:
        res = exec_net.requests[0].outputs[out_blob]
        for j in range(res.shape[2]):
            if res[0][0][j][0] < 0:break
            overlay_on_image(mask, res[0][0][j], dth_np, cap.scale)
        cv2.imshow('D435-Camera',frame_org | mask)
        key=cv2.waitKey(1)
        if key != -1:break
    else:
        print("error")
    # FPS
    done_frame+=1
    end = time()+1e-10
    sys.stdout.write('\b'*20)
    sys.stdout.write("%10.2f FPS"%(done_frame/(end-start)))
    sys.stdout.flush()

print("\nfinalizing")
cv2.destroyAllWindows()
del exec_net
del plugin
