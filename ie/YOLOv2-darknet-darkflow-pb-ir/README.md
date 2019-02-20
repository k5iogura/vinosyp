# YOLOv1/v2/v3 to OpenVINO 

Famous object detection DNN model called "YOLO" works on Darknet Framework.  
Darknet code is based on pure C language and has very simple function.  
Some github supports conversion darknet weights and cfg to the other frameworks such as caffe, tensorflow, chainer.  
To impliment YOLO onto OpenVINO Framework, we can refer these conversion methods.  
Refer [Using the Model Optimizer to Convert TensorFlow Models]
(https://software.intel.com/en-us/articles/OpenVINO-Using-TensorFlow#yolov1-v2-to-tf)

### convertion flow with darkflow
**Recommended workflow** to implement Yolo on OpenVINO is bellow,  
- convert **.cfg and .weights** files to tensorflow **.pb** file **via darkflow tool called "flow"**
- convert .pb file to OpenVINO **.bin and .xml** files for NCS **via model_optimizer called "mo_tf.py"**
- run script to check

**tool in/out**

|no|in          |tool    |out       |process                             |
|-:|   -        |   -    |   -      |   -                                |
|1 |.cfg,weights|    flow|       .pb|darknet to tensorflow   built_graph/|
|2 |.pb         |mo_tf.py|.bin, .xml|tensorflow to IRmodel   FP16/       |
|3 |.bin, .xml  |yolov2_vino.py|.png|Execution IRmodel with NCS          |

#### check downloaded weight and cfg on Darknet
Check that .weights and .cfg files work fine on Darknet framework.  

```
$ git clone https://github.com/pjreddi/darknet
$ cd darknet
$ make
$ ./darknet
usage: ./darknet <function>

$ wget https://pjreddie.com/media/files/yolov3.weights
$ wget https://pjreddie.com/media/files/yolov2-voc.weights
$ wget https://pjreddie.com/media/files/yolov1.weights

$ ./darknet detect cfg/yolov2-voc.cfg yolov2-voc.weights data/dog.jpg
...
Loading weights from yolov2-voc.weights...Done!
data/dog.jpg: Predicted in 27.621588 seconds.
dog: 82%
truck: 64%
bicycle: 85%

$ ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
...
Loading weights from yolov3.weights...Done!
data/dog.jpg
dog: 100%
truck: 92%
bicycle: 99%

$ ./darknet detect cfg/yolov1.cfg yolov1.weights data/dog.jpg
...
Loading weights from yolov1.weights...Done!
data/dog.jpg: Predicted in 16.605034 seconds.
train: 55%
```
Yolov2 and Yolov3 work fine.  
Yolov1 seems not work correctly but on coco.names train==7 and on voc.names car==7, so maybe right.   

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

**Patch ~/.local/lib/python2.7/site-packages/darkflow/net/build.py line 171. =20/16?**

#### convert .cfg, .weights to tensorflow .pb files

I assume bellow,
```
$ cd **this repo**/ie/YOLOv2-darknet-darkflow-pb-ir/
$ cp **darknet repo**/yolov*.weights .
$ cp **darknet repo**/cfg/yolov*.cfg .
```

**convertion with YOLOv3**

```
//check convertion Yolov3
$ flow --model cfg/yolov3.cfg --load yolov3.weights --savepb
Parsing ./cfg/yolov3.cfg
Layer [shortcut] not implemented
```

Error occurence, refer Intel information about YoloV3 tensorflow convertion problems.  

**convertion with YOLOv2**

```
//check convertion with Yolov2
$ cp data/voc.names labels.txt
$ flow --model cfg/yolov2-voc.cfg --load yolov2-voc.weights --savepb

Parsing ./cfg/yolov2-voc.cfg
Parsing cfg/yolov2-voc.cfg
Loading yolov2-voc.weights ...
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
 Load  |  Yep!  | conv 1x1p0_1    linear           | (?, 19, 19, 125)
-------+--------+----------------------------------+---------------
Running entirely on CPU
2019-01-25 17:48:08.137460: I tensorflow/core/platform/cpu_feature_guard.cc:141]
Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
Finished in 7.60918092728s

Rebuild a constant version ...
Done

$ ls built_graph/
yolov2-voc.meta  yolov2-voc.pb
```
results tensorflow .pb file was placed in **built_graph directory**.  
Empty bin/ ckpt/ sample_img/ directories ware created but i dont know why.  

**convertion with YOLOv1**

```
//check convertion with Yolov1
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

### convert tensorflow .pb file to OpenVINO IRmodel

Refer intel information [here](https://software.intel.com/en-us/articles/OpenVINO-Using-TensorFlow#yolov1-v2-to-ir)  

```
//convert .pb to .bin and xml
$ cp /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/tf/yolo_v1_v2.json .
$ /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py \
--input_model yolov2-voc.pb  \
--output_dir FP32  \
--data_type FP32   \
--batch 1          \
--tensorflow_use_custom_operations_config yolo_v1_v2.json 

Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	~/darknet/built_graphV2/yolov2-voc.pb
	- Path for generated IR: 	~/darknet/built_graphV2/FP32
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
	- Use the config file: 	yolo_v1_v2.json
Model Optimizer version: 	1.5.12.49d067a0

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: ~/darknet/built_graphV2/FP32/yolov2-voc.xml
[ SUCCESS ] BIN file: ~/darknet/built_graphV2/FP32/yolov2-voc.bin
[ SUCCESS ] Total execution time: 18.28 seconds. 

$ ls FP32/
yolov2-voc.bin  yolov2-voc.mapping  yolov2-voc.xml

```

```
// maybe bellow also,
git clone https://github.com/gflags/gflags
cd gflags
mkdir build;cd build
cmake .. && make
# make install
// needs bellow,
sudo apt install libomp-dev  @ YOLO-OpenVINO

cd ..  // Go to darknet directory
```

### Execute script to check

To run yolov2 demo,
```
$ python3 demo_img_yolov2_vino.py -d MYRIAD sample_image/dog.jpg
```

#### About prescript to IE input
- read image via opencv
- convert color placement BGR to RGB
- div 255.0
- transform to letterbox

#### About postscript from IE output
- Understand output layout of yolo
- nms
- confidence
- classification

### About difference btn darkflow and OpenVINO

Notice implementation differences of two framework, see bellow,
- darkflow execute Region layer of yolov2.cfg on python(Cython) without tensorflow.  
- OpenVINO execute Region layer on NCS.  

### About difference btn C++ and Python
- OpenVINO uses python-opencv on the other hand darkflow uses C++ opencv. So image resizing result is difference.  

Therefore postscript codes are difference. Attempt bellow,  

```
// result of darkflow
// result of python_vino.py
```

