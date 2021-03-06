#!/usr/bin/env python3
import argparse
from pdb import *
import sys,os
import cv2
import numpy as np
from time import time
from openvino.inference_engine import IENetwork, IEPlugin

LABELS = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')

def overlay_on_image(display_image, object_info):

    # the minimal score for a box to be shown
    min_score_percent = 25

    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent):
        # ignore boxes less than the minimum score
        return

    label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    # draw the classification label string just above and to the left of the rectangle
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

    # label text above the box
    cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

args = argparse.ArgumentParser()
args.add_argument("-d","--device", type=str, default='MYRIAD',help='on Device')
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
cap = cv2.VideoCapture(cameraNo)
if not cap.isOpened():sys.exit(-1)
print("Opened UVC-Camera via /dev/video0")
cap.set(cv2.CAP_PROP_FPS,30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

start = time()
done_frame=0
while True:
    ret, frame_org = cap.read()
    if not ret:break
    frame = cv2.resize(frame_org,input_image_size).astype(dtype=np.float)
    frame-= 127.5       # means
    frame*= 0.007853    # scale

    in_frame = cv2.resize(frame, (model_w, model_h))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((model_n, model_c, model_h, model_w))

    exec_net.start_async(request_id=0, inputs={input_blob: in_frame})

    if exec_net.requests[0].wait(-1) == 0:
        res = exec_net.requests[0].outputs[out_blob]
        for j in range(res.shape[2]):
            if res[0][0][j][0] < 0:break
            overlay_on_image(frame_org, res[0][0][j])
        cv2.imshow('USB-Camera',frame_org)
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
