from pdb import *
import argparse
import sys,os
import cv2
import numpy as np
from time import time
from openvino.inference_engine import IENetwork, IEPlugin
from picamera.array import PiRGBArray
from picamera import PiCamera

LABELS = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')

def preproc(frame_org, input_image_sizeNCHW):
    model_n, model_c, model_w, model_h = input_image_sizeNCHW
    frame = cv2.resize(frame_org,(model_w, model_h)).astype(dtype=np.float)
    frame-= 127.5       # means
    frame*= 0.007853    # scale
    in_frame = cv2.resize(frame, (model_w, model_h))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((model_n, model_c, model_h, model_w))
    return in_frame

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
args.add_argument("-d",   "--device",   type=str, default="MYRIAD", help="MYRIAD/CPU")
args.add_argument("-r",   "--requests", type=int, default=3   , help="Maximum requests for NCS")
args.add_argument("-cfr", "--camera_framerate",   type=int,     default= 120,help="Maximum Framerate for CSI")
args.add_argument("-crw", "--camera_resolution_w",type=int,     default= 320,help="Camera Width")
args.add_argument("-crh", "--camera_resolution_h",type=int,     default= 240,help="Camera Height")
args.add_argument("-demo","--demo",action='store_true',         help="Demonstration Mode")
args = args.parse_args()

WindowName ='CSI-Camera'

model_xml='vinosyp/models/SSD_Mobilenet/FP16/MobileNetSSD_deploy.xml'
model_bin='vinosyp/models/SSD_Mobilenet/FP16/MobileNetSSD_deploy.bin'
model_xml = os.environ['HOME'] + "/" + model_xml
model_bin = os.environ['HOME'] + "/" + model_bin
net = IENetwork(model=model_xml, weights=model_bin)	#R5

max_req=args.requests
plugin   = IEPlugin(device='MYRIAD', plugin_dirs=None)
exec_net = plugin.load(network=net, num_requests=max_req)

input_blob = next(iter(net.inputs))  #input_blob = 'data'
out_blob   = next(iter(net.outputs)) #out_blob   = 'detection_out'
model_n, model_c, model_h, model_w = net.inputs[input_blob].shape #Tool kit R4
input_image_size=(model_w,model_h)   # for cv2
print("max requests for NCS:",max_req)
print("n/c/h/w (from xml)= %d %d %d %d"%(model_n, model_c, model_h, model_w))
print("input_blob : out_blob =",input_blob,":",out_blob)
print("CSI Reso W/H",args.camera_resolution_w,args.camera_resolution_h)

del net

camera = PiCamera()
camera.framerate = 33   # slow
camera.framerate = 60   # fast
camera.framerate = 90   # more fast
camera.framerate = 120  # more more fast Total(camera + prediction) 18FPS over
camera.framerate = args.camera_framerate
camera.resolution = (args.camera_resolution_w, args.camera_resolution_h)
rawCapture = PiRGBArray(camera, size=(args.camera_resolution_w, args.camera_resolution_h))

results = []
latest  = None
for reqNo,csi_cam in enumerate(camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)):
    frame_org = csi_cam.array
    in_frame  = preproc(frame_org, (model_n, model_c, model_h, model_w))
    rawCapture.truncate(0)
    exec_net.start_async(request_id=reqNo, inputs={input_blob: in_frame})
    if exec_net.requests[reqNo].wait(-1)==0:
        latest = res = exec_net.requests[reqNo].outputs[out_blob]
        results.append(res)
    if reqNo == max_req-1:break

cv2.namedWindow(WindowName)
if args.demo:cv2.moveWindow(WindowName,200,0)
start = time()
done_frame=0
view_frame=0
reqNo     =0
for csi_cam in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    # CSI Camera-In
    frame_org= cv2.flip(csi_cam.array,0)
    in_frame = preproc(frame_org, (model_n, model_c, model_h, model_w))
    rawCapture.truncate(0)

    # Prediction
    pred_wait = time()
    if exec_net.requests[reqNo].wait(0) == 0:
        exec_net.requests[reqNo].wait(-1)
        latest = res = exec_net.requests[reqNo].outputs[out_blob]
        exec_net.start_async(request_id=reqNo, inputs={input_blob: in_frame})
        done_frame+=1
    else:
        res = latest
    pred_elapsed = time() - pred_wait

    # Drawing Result
    for j in range(res.shape[2]):
        if res[0][0][j][0] < 0:break
        overlay_on_image(frame_org, res[0][0][j])
    cv2.imshow(WindowName,frame_org)
    view_frame+=1
    key=cv2.waitKey(1)
    if key != -1:break

    # FPS
    end = time()+1e-10
    sys.stdout.write('\b'*60)
    FPS_str  = "%7.2f FPS"%(done_frame/(end-start+pred_elapsed))
    PBack_str= "%7.2f FPS"%(view_frame/(end-start))
    sys.stdout.write("Playback %s (Prediction %s)"%(PBack_str, FPS_str))
    sys.stdout.flush()
    reqNo+=1
    if reqNo>=max_req:reqNo=0

    # CSI Camera-In

print("\nfinalizing")
cv2.destroyAllWindows()
del exec_net
del plugin
