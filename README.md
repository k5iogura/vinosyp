# OpenVINO for Movidius NCS-1 and NCS-2(2nd generation stick)

![](files/logo.png)

[github for NCS(1st generation)](https://github.com/movidius/ncsdk) is completed. And NCS-2(2nd generation) is supported by OpenVINO. OpenVINO support not only NCS 1 and 2 but also CPU, GPU and FPGA(Altera Arria10 GX).  

[OpenVINO do not support RaspnerryPi](https://ncsforum.movidius.com/discussion/1302/intel-neural-compute-stick-2-information#latest)  

This is story of estimation of combination btn movidius NCS and OpenVINO.  

## Requirement

- Ubuntu16.04 on intelPC
- Nueral Compute Stick(1st generation), maybe ok with NCS-2(2nd generation)
- **OpenVINO 2018.5.445(releases_openvino-2018-r5)**  
  If you work with OpenVINO-R4 see this repo. branch **"OpenVINOR4"**.
## Download and installation

After registration we can get OpenVINO from [here](https://software.intel.com/en-us/openvino-toolkit).  

According to [here(Install the Intel® Distribution of OpenVINO™ toolkit for Linux)](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux), we can do from install to check by running DEMO(inference car.png) on CPU processing(We employed caffe optimizer installation only).  

**After Download l_openvino_toolkit_p_<version>.tgz,**
```
$ tar xzf l_openvino_toolkit_p_<version>.tgz
$ cd l_openvino_toolkit_p_<version>
# ./install_cv_sdk_dependencies.sh
# ./install_GUI.sh
$ . /opt/intel/computer_vision_sdk/bin/setupvars.sh ## add your .bashrc
```
**Setup Caffe and Tensorflow Model Optimizer**  
```
$ cd /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/install_prerequistes
$ ./install_prerequisites_caffe.sh
$ ./install_prerequisites_tf.sh
```
If you need more Framework, run bellow,  
```
install_prerequisites_<FW>.sh, FW is such as mxnet, onnx, kaldi.  
```

**1st DEMO result is bellow,**  

**Notice!:** Demo sample script create directory ~/openvino and download .prototxt and .caffemodel in it. So notice permission of directory.  

```
$ cd /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/install_prerequisites/
# ./install_prerequisites.sh
$ cd /opt/intel/computer_vision_sdk/deployment_tools/demo
$ ./demo_squeezenet_download_convert_run.sh
```

![](files/squeezenet_demo.png)

In my first impression, download and installation of OpenVINO is very easy!, great.  

**2nd DEMO result is bellow,**

1pipe : recognize car region box  
2pipe : recognize licence plate region box  
3pipe : recognize charactors on licence plate  

```
$ cd ~/intel/computer_vision_sdk/deployment_tools/demo
$ ./demo_security_barrier_camera.sh
```

![](files/pipeline.png)

Our DEMO-environment is on VirtualBox Ubuntu16.04 on Intel Celeron CPU.  
But it shows performance as 6.39fps for 1st pipe, 19.69fps for 2nd pipe, 9.55fps for 3rd pipe.  

## Using Movidius NCS-1

To run demo scripts on NCS-1, **add usbboot rule** and **add your user id into "users group"**, finally run demo scripts with **-d MYRIAD option**.

Setup udev rules for MYRIAD.  
Place [97-myriad-usbboot.rules](./etc/udev/rules.d/97-myriad-usbboot.rules) on /etc/udev/rules.d/

```
# usermod -a -G users "$(whoami)"
# cp 97-usbboot.rules /etc/udev/rules.d/
# udevadm control --reload-rules
# udevadm trigger
# ldconfig
# reboot
```

After Rebooting plugin NCS-1 on USB port.  

On the our way to install, we selected option for Movidius NCS-1 and NCS-2 support, so that we are ready to run NCS via OpenVINO as inference engin(called IE) **by adding -d MYRIAD** with sample script.  

```
$ cd /opt/intel/computer_vision_sdk/deployment_tools/demo
$ ./demo_squeezenet_download_convert_run.sh -d MYRIAD
$ ./demo_security_barrier_camera.sh -d MYRIAD
```
Check "[INFO] Loading LPR model to **the MYRIAD plugin**" log messages.  

## Additional demo for other models on NCS-1  

### SSD_MobileNet
- Download ncappzoo from [here](https://github.com/k5iogura/ncappzoo)  
- Run model optimizer with .caffemodel and .prototxt for SSD_MobileNet  
- Run DEMO script

#### model conversion caffe to ir(OpenVINO intermidiate representation)  
For OpenVINO generate intermidiate representation as .bin and .xml.  
Here .bin file includes weights of model and .xml file includes network structure.  
OpenVINO Model Optimizer help is bellow,  
```
$ /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_caffe.py --help
usage: mo_caffe.py [-h] [--input_model INPUT_MODEL] [--model_name MODEL_NAME]
                   [--output_dir OUTPUT_DIR] [--input_shape INPUT_SHAPE]
                   [--scale SCALE] [--reverse_input_channels]
                   [--log_level {CRITICAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}]
                   [--input INPUT] [--output OUTPUT]
                   [--mean_values MEAN_VALUES] [--scale_values SCALE_VALUES]
                   [--data_type {FP16,FP32,half,float}] [--disable_fusing]
                   [--disable_resnet_optimization]
                   [--finegrain_fusing FINEGRAIN_FUSING] [--disable_gfusing]
                   [--move_to_preprocess] [--extensions EXTENSIONS]
                   [--batch BATCH] [--version] [--silent]
                   [--freeze_placeholder_with_value FREEZE_PLACEHOLDER_WITH_VALUE]
                   [--generate_deprecated_IR_V2] [--input_proto INPUT_PROTO]
                   [-k K] [--mean_file MEAN_FILE]
                   [--mean_file_offsets MEAN_FILE_OFFSETS]
                   [--disable_omitting_optional]
                   [--enable_flattening_nested_params]
```

For Movidius use data_type with FP16 only.  
command line may be bellow,
```
$ export MO=/opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/
$ cd ~/openvino_fs/models/SSD_Mobilenet/caffe
$ python $MO/mo_caffe.py --input_model MobileNetSSD_deploy.caffemodel --output_dir ../FP16/ --data_type FP16
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	~/openvino_fs/models/SSD_Mobilenet/caffe/MobileNetSSD_deploy.caffemodel
	- Path for generated IR: 	~/openvino_fs/models/SSD_Mobilenet/caffe/../FP16/
	- IR output name: 	MobileNetSSD_deploy
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values:  	Not specified	
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	False
Caffe specific parameters:
	- Enable resnet optimization: 	True
	- Path to the Input prototxt: 	~/openvino_fs/models/SSD_Mobilenet/caffe/MobileNetSSD_deploy.prototxt
	- Path to CustomLayersMapping.xml: 	Default
	- Path to a mean file:  	Not specified
	- Offsets for a mean file: 	Not specified
Model Optimizer version: 	        1.4.292.6ef7232d

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: ~/openvino_fs/models/SSD_Mobilenet/caffe/../FP16/MobileNetSSD_deploy.xml
[ SUCCESS ] BIN file: ~/openvino_fs/models/SSD_Mobilenet/caffe/../FP16/MobileNetSSD_deploy.bin
[ SUCCESS ] Total execution time: 2.85 seconds. 

$ ls ../FP16
MobileNetSSD_deploy.bin  MobileNetSSD_deploy.mapping  MobileNetSSD_deploy.xml  
```

#### DetectionOutput Layer of OpenVINO

*DetectionOutput* layer in **models/SSD_Mobilenet/caffe/MobileNetSSD_deploy.prototxt** consists of bellow,

```
layer {
  name: "detection_out"
  type: "DetectionOutput"
  bottom: "mbox_loc"
  bottom: "mbox_conf_flatten"
  bottom: "mbox_priorbox"
  top: "detection_out"
  include {
    phase: TEST
  }
  detection_output_param {
    num_classes: 21
    share_location: true
    background_label_id: 0
    nms_param {
      nms_threshold: 0.45
      top_k: 100
    }
    code_type: CENTER_SIZE
    keep_top_k: 100
    confidence_threshold: 0.25
  }
}
```

In OpenVINO Framework, above *DetectionOutput* layer outputs bellow numpy structure.  

**DetectionOutput layer output structure, numpy shape is (1,1,100,7)**

|Valid|class-id|conf|x1|y1|x2|y2|
|   -:|      -:|  -:|-:|-:|-:|-:|
|    0|class-id|conf|x1|y1|x2|y2|
|    0|     ...| ...|..|..|..|..|
|    0|class-id|conf|x1|y1|x2|y2|
|   -1|       0|   0| 0| 0| 0| 0|
|    0|     ...| ...|..|..|..|..|
|    0|       0|   0| 0| 0| 0| 0|

100 lines and 7 items a line are outputed.  
Valid is 0 or -1 here 0 is valid and -1 is invalid beyond lines.  
Analyze above structure and use result of inference with your custom way.  

#### Try to infer sample images with SSD_Mobilenet model as text output  
Simple script named "ssd_mobilenet.py" infer 3 images and output results as text.  

```
$ cd ~/openvino_fs/ie/SSD_Mobilenet/
$ ls
images/  ssd_mobilenet.py demo_ssd_mobilenet.py demo_uvc_ssd_mobilenet.py

// result as text printout
$ python3 ssd_mobilenet.py images/*
n/c/h/w (from xml)= 1 3 300 300
input_blob : out_blob = data : detection_out
input image = images/DAR_Facts_17.jpg
in-frame (1, 3, 300, 300)
fin (1, 1, 100, 7)
top
[0.          6.           1.          0.5126953   0.20935059  0.9482422  0.70410156]
[0.          7.           0.29614258  0.43115234  0.39086914  0.49072266 0.49487305]
[ 0.         15.          0.4152832   0.34228516  0.41918945  0.38720703 0.625     ]
[ 0.         15.          0.26489258  0.25732422  0.41308594  0.3178711  0.79785156]
input image = images/Gene-Murtagh-Kingspan-670x310.jpg
in-frame (1, 3, 300, 300)
fin (1, 1, 100, 7)
top
[ 0.         15.          0.9980469   0.10913086  0.0234375   0.86035156  0.9892578 ]
input image = images/car.jpg
in-frame (1, 3, 300, 300)
fin (1, 1, 100, 7)
top
[0.         7.         1.         0.10473633 0.38916016 0.8925781 0.91552734]
input image = images/pedestiran-bridge.jpg
in-frame (1, 3, 300, 300)
fin (1, 1, 100, 7)
top
[ 0.         15.          0.7158203   0.32128906  0.08862305  0.43945312  0.87402344]
[ 0.         15.          0.5390625   0.4086914   0.11035156  0.52197266  0.8515625 ]
[ 0.         15.          0.45532227  0.6220703   0.12280273  0.7216797   0.75      ]
```
each line means that "N/A  class  x1  y1  x2  y2" and here classes are as VOC bellow,  
0: background  
1: aeroplane  
2: bicycle  
3: bird  
4: boat  
5: bottle  
6: bus  
7: car  
8: cat  
9: chair  
10: cow  
11: diningtable  
12: dog  
13: horse  
14: motorbike  
15: person  
16: pottedplant  
17: sheep  
18: sofa  
19: train  
20: tvmonitor  

#### Drawing result of inference on inferred image
Next script named "demo_ssd_mobilenet.py" shows results of inferenced region boxes on image.  

```
$ python3 demo_ssd_mobilenet.py images/pedestiran-bridge.jpg
```

![](./files/pedestiran-bridge_result.jpg)  

#### Using USB Camera as input of demo
Next scripts named "demo_uvc_ssd_mobilenet.py" provides real-time inference demonstration.  
Ubuntu16.04 supports UVC Camera by default kernel **via /dev/video0**.  

```
// check uvc camera device
$ ls /dev/video*
/dev/video0

// result in video window
$ python3 demo_uvc_ssd_mobilenet.py
```

![](./files/uvc_camera.png)

My morning coffee!

### YOLOv1/v2/v3  
Famous object detection DNN model called "YOLO" works on Darknet Framework.  
Darknet code is based on pure C language code and very simple function.  
Some github supports conversion darknet to other framework such as caffe, tensorflow, chainer.  
To impliment YOLO onto other Framework, we can use these convert tools.  
Refer [Using the Model Optimizer to Convert TensorFlow* Models]
(https://software.intel.com/en-us/articles/OpenVINO-Using-TensorFlow#yolov1-v2-to-tf)

#### [darkflow](https://github.com/thtrieu/darkflow)
darkflow includes tool called **"flow"** that convert darknet .cfg and .weights to tensorflow .pb file.  
Look at [How to install darkflow](https://github.com/thtrieu/darkflow)

#### install darkflow
clone and check  

```
//clone
$ git clone https://github.com/thtrieu/darkflow
$ cd darkflow

//requirement
$ pip Cython  
# apt python-numpy  
$ pip tensorflow==1.12.0  
# apt python-opencv  

//install global
$ pip install .
Processing ~/darkflow
Installing collected packages: darkflow
  Running setup.py install for darkflow ... done
Successfully installed darkflow-1.0.0

//check by showing help
$ cd
$ flow --h
```

**Patch ~/.local/lib/python2.7/site-packages/darkflow/net/build.py line 171.**

#### convertion to tensorflow .pb files

```
//check convertion Yolov3
$ git clone https://github.com/pjreddi/darknet
$ cd darknet
$ wget https://pjreddie.com/media/files/yolov3.weights
$ flow --model cfg/yolov3.cfg --load yolov3.weights --savepb
Parsing ./cfg/yolov3.cfg
Layer [shortcut] not implemented
```

Error occurence, refer Intel information about YoloV3 tensorflow convertion problems.  

```
//check convertion with Yolov2
$ wget https://pjreddie.com/media/files/yolov2.weights
$ cp data/coco.names labels.txt
$ flow --model cfg/yolov2.cfg --load yolov2.weights --savepb

Parsing ./cfg/yolov2.cfg
Parsing cfg/yolov2.cfg
Loading yolov2.weights ...
Successfully identified 203934260 bytes
Finished in 0.041063785553s

Building net ...
Source | Train? | Layer description                | Output size
-------+--------+----------------------------------+---------------
       |        | input                            | (?, 608, 608, 3)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 608, 608, 32)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 304, 304, 32)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 304, 304, 64)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 152, 152, 64)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 152, 152, 128)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 152, 152, 64)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 152, 152, 128)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 76, 76, 128)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 76, 76, 256)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 76, 76, 128)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 76, 76, 256)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 38, 38, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 38, 38, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 38, 38, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 38, 38, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 38, 38, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 38, 38, 512)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 19, 19, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 19, 19, 1024)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 19, 19, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 19, 19, 1024)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 19, 19, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 19, 19, 1024)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 19, 19, 1024)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 19, 19, 1024)
 Load  |  Yep!  | concat [16]                      | (?, 38, 38, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 38, 38, 64)
 Load  |  Yep!  | local flatten 2x2                | (?, 19, 19, 256)
 Load  |  Yep!  | concat [27, 24]                  | (?, 19, 19, 1280)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 19, 19, 1024)
 Load  |  Yep!  | conv 1x1p0_1    linear           | (?, 19, 19, 425)
-------+--------+----------------------------------+---------------
Running entirely on CPU
2019-01-25 17:48:08.137460: I tensorflow/core/platform/cpu_feature_guard.cc:141]
Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
Finished in 7.60918092728s

Rebuild a constant version ...
Done

$ ls built_graph/
yolov2.meta  yolov2.pb
```
results tensorflow .pb file was placed in **built_graph directory**.  
Empty bin/ ckpt/ sample_img/ directories ware created but i dont know why.  

```
//check convertion with Yolov1
$ wget https://pjreddie.com/media/files/yolov1.weights
$ cp data voc.names labels.txt
$ flow --model cfg/yolov1.cfg --load yolov1.weights --savepb

Parsing ./cfg/yolov1.cfg
Parsing cfg/yolov1.cfg
Loading yolov1.weights ...
Successfully identified 789312988 bytes
Finished in 0.0449588298798s
Model has a VOC model name, loading VOC labels.

Building net ...
Source | Train? | Layer description                | Output size
-------+--------+----------------------------------+---------------
       |        | input                            | (?, 448, 448, 3)
 Load  |  Yep!  | conv 7x7p3_2  +bnorm  leaky      | (?, 224, 224, 64)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 112, 112, 64)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 112, 112, 192)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 56, 56, 192)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 56, 56, 128)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 56, 56, 256)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 56, 56, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 56, 56, 512)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 28, 28, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 28, 28, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 28, 28, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 28, 28, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 28, 28, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 28, 28, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 28, 28, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 28, 28, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 28, 28, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 28, 28, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 28, 28, 1024)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 14, 14, 1024)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 14, 14, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 14, 14, 1024)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 14, 14, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 14, 14, 1024)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 14, 14, 1024)
 Load  |  Yep!  | conv 3x3p1_2  +bnorm  leaky      | (?, 7, 7, 1024)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 7, 7, 1024)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 7, 7, 1024)
 Load  |  Yep!  | loca 3x3p1_1  leaky              | (?, 7, 7, 256)
 Load  |  Yep!  | drop                             | (?, 7, 7, 256)
 Load  |  Yep!  | flat                             | (?, 12544)
 Load  |  Yep!  | full 12544 x 1715  linear        | (?, 1715)
-------+--------+----------------------------------+---------------
Running entirely on CPU
2019-01-25 18:17:37.315643: I tensorflow/core/platform/cpu_feature_guard.cc:141]
Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-01-25 18:17:37.601772: W tensorflow/core/framework/allocator.cc:122] Allocation of 462422016 exceeds 10% of system memory.
2019-01-25 18:17:38.788114: W tensorflow/core/framework/allocator.cc:122] Allocation of 462422016 exceeds 10% of system memory.
2019-01-25 18:17:39.266294: W tensorflow/core/framework/allocator.cc:122] Allocation of 462422016 exceeds 10% of system memory.
2019-01-25 18:17:39.742040: W tensorflow/core/framework/allocator.cc:122] Allocation of 462422016 exceeds 10% of system memory.
2019-01-25 18:17:40.789931: W tensorflow/core/framework/allocator.cc:122] Allocation of 462422016 exceeds 10% of system memory.
Finished in 18.2570509911s

Rebuild a constant version ...
Done

$ ls built_graph
yolov1.meta  yolov1.pb
```
#### check downloaded weight and cfg on Darknet
Check that weight and cfg files work fine on Darknet framework.  

```
$ make
$ ./darknet
usage: ./darknet <function>

$ ./darknet detect cfg/yolov2.cfg yolov2.weights data/dog.jpg
...
Loading weights from yolov2.weights...Done!
data/dog.jpg: Predicted in 27.621588 seconds.
dog: 82%
truck: 64%
bicycle: 85%

$ ./darknet detect cfg/yolov1.cfg yolov1.weights data/dog.jpg
...
Loading weights from yolov1.weights...Done!
data/dog.jpg: Predicted in 16.605034 seconds.
train: 55%
```
Yolov2 work fine.  
Yolov1 seems not work correctly but on coco.names train==7 and on voc.names car==7, so maybe right.   

### convertion tensorflow .pb file to OpenVINO IR files
Refer intel information [here](https://software.intel.com/en-us/articles/OpenVINO-Using-TensorFlow#yolov1-v2-to-ir)  

```
//convert .pb to .bin and xml
/opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py \
--input_model yolov2.pb  \
--output_dir FP32  \
--data_type FP32   \
--batch 1          \
--tensorflow_use_custom_operations_config  \
        /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/tf/yolo_v1_v2.json 
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/home/ogura/darknet/built_graphV2/yolov2.pb
	- Path for generated IR: 	/home/ogura/darknet/built_graphV2/FP32
	- IR output name: 	yolov2
	- Log level: 	ERROR
	- Batch: 	1
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP32
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	False
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Offload unsupported operations: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	None
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/tf/yolo_v1_v2.json
Model Optimizer version: 	1.5.12.49d067a0

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: ~/darknet/built_graphV2/FP32/yolov2.xml
[ SUCCESS ] BIN file: ~/darknet/built_graphV2/FP32/yolov2.bin
[ SUCCESS ] Total execution time: 18.28 seconds. 

$ ls FP32/
yolov2.bin  yolov2.mapping  yolov2.xml
```

sudo apt install libomp-dev  @ YOLO-OpenVINO

### convertion flow
Main workflow to implement Yolo on OpenVINO is bellow,  
- convert .cfg and .weights files to tensorflow .pb file via darkflow tool
- convert .pb file to OpenVINO .bin and .xml files for NCS
- run script to check

## Also refer below web site,  
None
