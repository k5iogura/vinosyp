#!/usr/bin/env python3
from pdb import *
import sys,os
from math import sqrt
from time import time
import cv2
import numpy as np
import argparse
from openvino.inference_engine import IENetwork, IEPlugin
from picamera.array import PiRGBArray
from picamera import PiCamera

#from test import preprocessing, postprocessing

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def iou(boxA,boxB):
  # boxA = boxB = [x1,y1,x2,y2]

  # Determine the coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  # Compute the area of intersection
  intersection_area = (xB - xA + 1) * (yB - yA + 1)
 
  # Compute the area of both rectangles
  boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
  # Compute the IOU
  iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

  return iou



def non_maximal_suppression(thresholded_predictions,iou_threshold):

  nms_predictions = []

  # Add the best B-Box because it will never be deleted
  if len(thresholded_predictions)<=0: return nms_predictions
  nms_predictions.append(thresholded_predictions[0])

  # For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
  # thresholded_predictions[i][0] = [x1,y1,x2,y2]
  i = 1
  while i < len(thresholded_predictions):
    n_boxes_to_check = len(nms_predictions)
    #print('N boxes to check = {}'.format(n_boxes_to_check))
    to_delete = False

    j = 0
    while j < n_boxes_to_check:
        curr_iou = iou(thresholded_predictions[i][0],nms_predictions[j][0])
        if(curr_iou > iou_threshold ):
            to_delete = True
        #print('Checking box {} vs {}: IOU = {} , To delete = {}'.format(thresholded_predictions[i][0],nms_predictions[j][0],curr_iou,to_delete))
        j = j+1

    if to_delete == False:
        nms_predictions.append(thresholded_predictions[i])
    i = i+1

  return nms_predictions

def preprocessing(input_image,ph_height,ph_width,ph_form='WHC'):

  #input_image = cv2.imread(input_img_path)        # HWC BGR

  # Resize the image and convert to array of float32
  resized_image = cv2.resize(input_image,(ph_width, ph_height), interpolation = cv2.INTER_CUBIC)
  #print(resized_image.shape)

  if ph_form == 'WHC':
    input_image = input_image.transpose((1,0,2))  # WHC
  else: pass                                      # HWC
  image_data = np.array(resized_image, dtype='f')

  # Normalization [0,255] -> [0,1]
  image_data /= 255.

  # BGR -> RGB? The results do not change much
  # copied_image = image_data
  #image_data[:,:,2] = copied_image[:,:,0]
  #image_data[:,:,0] = copied_image[:,:,2]

  # Add the dimension relative to the batch size needed for the input placeholder "x"
  image_array = np.expand_dims(image_data, 0)  # NWHC or NHWC

  return image_array

def postprocessing(predictions,input_image,score_threshold,iou_threshold,ph_height,ph_width):

  #input_image = cv2.imread(input_img_path)  # HWC
  input_image = cv2.resize(input_image,(ph_width, ph_height), interpolation = cv2.INTER_CUBIC)

  n_classes = 20
  n_grid_cells = 13
  n_b_boxes = 5
  n_b_box_coord = 4

  # Names and colors for each class
  classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
  colors = [(254.0, 254.0, 254), (239.8, 211.6, 127), 
              (225.7, 169.3, 0), (211.6, 127.0, 254),
              (197.5, 84.6, 127), (183.4, 42.3, 0),
              (169.3, 0.0, 254), (155.2, -42.3, 127),
              (141.1, -84.6, 0), (127.0, 254.0, 254), 
              (112.8, 211.6, 127), (98.7, 169.3, 0),
              (84.6, 127.0, 254), (70.5, 84.6, 127),
              (56.4, 42.3, 0), (42.3, 0.0, 254), 
              (28.2, -42.3, 127), (14.1, -84.6, 0),
              (0.0, 254.0, 254), (-14.1, 211.6, 127)]

  # Pre-computed YOLOv2 shapes of the k=5 B-Boxes
  anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]

  thresholded_predictions = []
  #print('Thresholding on (Objectness score)*(Best class score) with threshold = {}'.format(score_threshold))

  # IMPORTANT: reshape to have shape = [ 13 x 13 x (5 B-Boxes) x (4 Coords + 1 Obj score + 20 Class scores ) ]
  # From now on the predictions are ORDERED and can be extracted in a simple way!
  # We have 13x13 grid cells, each cell has 5 B-Boxes, each B-Box have 25 channels with 4 coords, 1 Obj score , 20 Class scores
  # E.g. predictions[row, col, b, :4] will return the 4 coords of the "b" B-Box which is in the [row,col] grid cell
  predictions = np.reshape(predictions,(9,11,5,25))

  # IMPORTANT: Compute the coordinates and score of the B-Boxes by considering the parametrization of YOLOv2
  for row in range(9):
    for col in range(11):
      for b in range(n_b_boxes):

        tx, ty, tw, th, tc = predictions[row, col, b, :5]

        # IMPORTANT: (416 img size) / (13 grid cells) = 32!
        # YOLOv2 predicts parametrized coordinates that must be converted to full size
        # final_coordinates = parametrized_coordinates * 32.0 ( You can see other EQUIVALENT ways to do this...)
        center_x = (float(col) + sigmoid(tx)) * 32.0
        center_y = (float(row) + sigmoid(ty)) * 32.0

        roi_w = np.exp(tw) * anchors[2*b + 0] * 32.0
        roi_h = np.exp(th) * anchors[2*b + 1] * 32.0

        final_confidence = sigmoid(tc)

        # Find best class
        class_predictions = predictions[row, col, b, 5:]
        class_predictions = softmax(class_predictions)

        class_predictions = tuple(class_predictions)
        best_class = class_predictions.index(max(class_predictions))
        best_class_score = class_predictions[best_class]

        # Compute the final coordinates on both axes
        left   = int(center_x - (roi_w/2.))
        right  = int(center_x + (roi_w/2.))
        top    = int(center_y - (roi_h/2.))
        bottom = int(center_y + (roi_h/2.))
        
        if( (final_confidence * best_class_score) > score_threshold):
          thresholded_predictions.append([[left,top,right,bottom],final_confidence * best_class_score,classes[best_class]])

  # Sort the B-boxes by their final score
  thresholded_predictions.sort(key=lambda tup: tup[1],reverse=True)

  #print('Printing {} B-boxes survived after score thresholding:'.format(len(thresholded_predictions)))
  #for i in range(len(thresholded_predictions)):
  #  print('B-Box {} : {}'.format(i+1,thresholded_predictions[i]))

  # Non maximal suppression
  #print('Non maximal suppression with iou threshold = {}'.format(iou_threshold))
  nms_predictions = non_maximal_suppression(thresholded_predictions,iou_threshold)

  # Print survived b-boxes
  #print('Printing the {} B-Boxes survived after non maximal suppression:'.format(len(nms_predictions)))
  #for i in range(len(nms_predictions)):
  #  print('B-Box {} : {}'.format(i+1,nms_predictions[i]))

  # Draw final B-Boxes and label on input image
  for i in range(len(nms_predictions)):

      color = colors[classes.index(nms_predictions[i][2])]
      best_class_name = nms_predictions[i][2]

      # Put a class rectangle with B-Box coordinates and a class label on the image
      input_image = cv2.rectangle(input_image,(nms_predictions[i][0][0],nms_predictions[i][0][1]),(nms_predictions[i][0][2],nms_predictions[i][0][3]),color)
      cv2.putText(input_image,best_class_name,(int((nms_predictions[i][0][0]+nms_predictions[i][0][2])/2),int((nms_predictions[i][0][1]+nms_predictions[i][0][3])/2)),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
  
  return input_image

args = argparse.ArgumentParser()
args.add_argument("images", nargs='*', type=str)
args.add_argument("-d", "--device"   , type=str, default="MYRIAD", help="Default MYRIAD or CPU")
args.add_argument("-cfr", "--camera_framerate"   , type=int, default=33, help="")
args.add_argument("-crw", "--camera_resolution_w", type=int, default=320, help="")
args.add_argument("-crh", "--camera_resolution_h", type=int, default=240, help="")
args = args.parse_args()

data_type="FP16"
if args.device == "CPU": data_type="FP32"

model_xml='./'+data_type+'/y.xml'
model_bin='./'+data_type+'/y.bin'
plugin = IEPlugin(device=args.device, plugin_dirs=None)
extension = "/inference_engine_samples/intel64/Release/lib/libcpu_extension.so"
extension = "/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so" # since 2019R1
extension = os.environ['HOME']+extension
if args.device == "CPU":plugin.add_cpu_extension(extension)
net = IENetwork(model=model_xml, weights=model_bin)	# R5

print(model_bin, "on", args.device)
exec_net = plugin.load(network=net, num_requests=1)

input_blob = next(iter(net.inputs))  #input_blob = 'data'
out_blob   = next(iter(net.outputs)) #out_blob   = 'detection_out'
print(net.inputs[input_blob].shape)
model_n, model_c, model_h, model_w = net.inputs[input_blob].shape #Tool kit R4
print("input_blob shape(from xml)", net.inputs[input_blob].shape)
print("name (input_blob : out_blob) =",input_blob,":",out_blob)

camera = PiCamera()
camera.framerate = args.camera_framerate
camera.resolution = (args.camera_resolution_w, args.camera_resolution_h)
rawCapture = PiRGBArray(camera, size=(args.camera_resolution_w, args.camera_resolution_h))

results = []
latest  = None
score_threshold = 0.3
iou_threshold = 0.3

start = time()
done_image = 0

for reqNo,csi_cam in enumerate(camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)):
    in_frame_org = csi_cam.array
    rawCapture.truncate(0)
    in_frame = preprocessing(in_frame_org,model_h,model_w,ph_form='HWC') # NHWC
    in_frame = in_frame.transpose((0,3,1,2))

    exec_net.start_async(request_id=0, inputs={input_blob: in_frame})

    if exec_net.requests[0].wait(-1) == 0:
        elapsed = time() - start
        done_image+=1
        res = exec_net.requests[0].outputs[out_blob]
        res = res.transpose((0,2,3,1))
        res_image = postprocessing(res,in_frame_org,score_threshold,iou_threshold,model_h,model_w)
        cv2.imshow('yolov2-tiny_352x288',res_image)
        key = cv2.waitKey(1)
        if key!=-1:break
        sys.stdout.write('\b'*20)
        sys.stdout.write("%.3fFPS"%(done_image/elapsed))
        sys.stdout.flush()
    else:
        print("error")

print("\nfinalize")
cv2.destroyAllWindows()
del net
del exec_net
del plugin

sys.exit(-1)
