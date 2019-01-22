# OpenVINO for Movidius NCS-1 and NCS-2(2nd generation stick)

![](files/logo.png)

[github for NCS(1st generation)](https://github.com/movidius/ncsdk) is completed. And NCS-2(2nd generation) is supported by OpenVINO. OpenVINO support not only NCS 1 and 2 but also CPU, GPU and FPGA(Altera Arria10 GX).  

[OpenVINO do not support RaspnerryPi](https://ncsforum.movidius.com/discussion/1302/intel-neural-compute-stick-2-information#latest)  

This is story of estimation of combination btn movidius NCS and OpenVINO.  

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
```

On the our way to install, we selected option for Movidius NCS-1 and NCS-2 support, so that we are ready to run NCS via OpenVINO as inference engin(called IE) **by adding -d MYRIAD** with sample script.  

```
$ cd /opt/intel/computer_vision_sdk/deployment_tools/demo
$ ./demo_squeezenet_download_convert_run.sh -d MYRIAD
$ ./demo_security_barrier_camera.sh -d MYRIAD
```
Check "[INFO] Loading LPR model to **the MYRIAD plugin**" log messages.  

## Additional demo for other models  

### SSD_MobileNet
- Download ncappzoo from [here](https://github.com/k5iogura/ncappzoo)  
- Run model optimizer with .caffemodel and .prototxt for SSD_MobileNet  
- Run DEMO script

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
$ python mo_caffe.py --input_model MobileNetSSD_deploy.caffemodel --output_dir ../FP16/ --mean_values 127 --data_type FP16
```

## Also refer below web site,  
[Intel Neural Compute Stick Getting start](https://software.intel.com/en-us/neural-compute-stick/get-started)  
[AIを始めよう！PythonでOpenVINOの仕組みを理解する](https://qiita.com/ammo0613/items/ff7452f2c7fab36b2efc)  
