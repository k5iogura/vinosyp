#!/bin/bash
source /opt/intel/openvino/bin/setupvars.sh
for i in `seq 1000`;do sleep 5;sudo sh -c "echo 0 > /sys/class/graphics/fb0/blank";done&
sudo sh -c "echo 0 > /sys/class/graphics/fb0/blank"
export DISPLAY=:0
xhost +
cd /home/pi/vinosyp/ie/YOLOv2tiny352x288-darknet-pb-ir
xterm -geometry +10+10 -e "python3 tinyyolov2_demo_csi.py -crw 320 -crh 240"
