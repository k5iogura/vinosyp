# OpenVINO for Movidius NCS-1 and NCS-2(2nd generation stick)

![](files/logo.png)

[github for NCS(1st generation)](https://github.com/movidius/ncsdk) is completed. And NCS-2(2nd generation) is supported by OpenVINO. OpenVINO support not only NCS 1 and 2 but also CPU, GPU and FPGA(Altera Arria10 GX).  

[OpenVINO do not support RaspnerryPi](https://ncsforum.movidius.com/discussion/1302/intel-neural-compute-stick-2-information#latest)  

This is story of estimation of combination btn movidius NCS and OpenVINO.  

## Download and installation

After registration we can get OpenVINO from [here](https://software.intel.com/en-us/openvino-toolkit).  

According to [here(Install the Intel® Distribution of OpenVINO™ toolkit for Linux)](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux), we can do from install to check by running DEMO(inference car.png) on CPU processing(We employed caffe optimizer installation only).  

**1st DEMO result is bellow,**  

![](files/squeezenet_demo.png)

In my first impression, download and installation of OpenVINO is very easy!, great.  

**2nd DEMO result is bellow,**

1pipe : recognize car region box  
2pipe : recognize licence plate region box  
3pipe : recognize charactors on licence plate  

![](files/pipeline.png)

Our DEMO-environment is on VirtualBox Ubuntu16.04 on Intel Celeron CPU.  
But it shows performance as 6.39fps for 1st pipe, 19.69fps for 2nd pipe, 9.55fps for 3rd pipe.  

## Using Movidius NCS-1

On the our way to install, we selected option for Movidius NCS-1 and NCS-2 support, so that we are ready to run NCS via OpenVINO as inference engin(called IE).  


