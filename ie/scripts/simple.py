#!/usr/bin/env python3
#STEP-1
import sys,os
import cv2
from openvino.inference_engine import IENetwork, IEPlugin

#STEP-2
model_xml='openvino_models/ir/OpenPose/graph_freeze.xml'
model_bin='openvino_models/ir/OpenPose/graph_freeze.bin'
model_xml = os.environ['HOME'] + "/" + model_xml
model_bin = os.environ['HOME'] + "/" + model_bin
net = IENetwork.from_ir(model=model_xml, weights=model_bin)

#STEP-3
plugin = IEPlugin(device='MYRIAD', plugin_dirs=None)
exec_net = plugin.load(network=net, num_requests=1)

#STEP-4
input_blob = next(iter(net.inputs))  #input_blob = 'data'
out_blob   = next(iter(net.outputs)) #out_blob   = 'detection_out'
model_n, model_c, model_h, model_w = net.inputs[input_blob].shape #Tool kit R4
print("n/c/h/w (from xml)= %d %d %d %d"%(model_n, model_c, model_h, model_w))
print("input_blob : out_blob =",input_blob,":",out_blob)

del net

del exec_net
del plugin
sys.exit(1)
#STEP-5
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #STEP-6
    cap_w = cap.get(3)
    cap_h = cap.get(4)
    in_frame = cv2.resize(frame, (model_w, model_h))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((model_n, model_c, model_h, model_w))

    #STEP-7
    exec_net.start_async(request_id=0, inputs={input_blob: in_frame})

    if exec_net.requests[0].wait(-1) == 0:
        res = exec_net.requests[0].outputs[out_blob]

        #STEP-8
        for obj in res[0][0]:
            if obj[2] > 0.5:
                xmin = int(obj[3] * cap_w)
                ymin = int(obj[4] * cap_h)
                xmax = int(obj[5] * cap_w)
                ymax = int(obj[6] * cap_h)
                class_id = int(obj[1])
                # Draw box and label\class_id
                color = (255, 0, 0)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, class_id + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    #STEP-9
    cv2.imshow("Detection Results", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

#STEP-10
cv2.destroyAllWindows()
del exec_net
del plugin
