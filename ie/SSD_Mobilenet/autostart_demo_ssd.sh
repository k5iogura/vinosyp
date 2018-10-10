#! /bin/bash
sudo sh -c "echo 0 > /sys/class/graphics/fb0/blank"
export DISPLAY=:0
xhost +
cd /home/pi/vinosyp/ie/SSD_Mobilenet
python3 demo_csi_ssd_mobilenet.py -r 3 -crw 640 -crh 400 -demo

for i in `seq 1000`;
do
sleep 90
sudo sh -c "echo 0 > /sys/class/graphics/fb0/blank"
done
