# Installation of tensorflow-gpu

**tensorflow-gpu** is provided as compiled wheel because to use available Intel instruction set such as AVX, SSE, FMA  
and Graphic processor as GPGPU.  

Other hand, tensorflow-gpu wheel constraints your using CPU, GPU.  
Unfortunaly building tensorflow from source code is very tuff job.  

If want to use tensorflow-gpu from pip, should install CUDA and CuDNN package correspoding to the tensorflow-gpu wheel binary.  

#### CUDA and CuDNN in local place
Install CUDA-9.0 like bellow,  
```
$ cd; make local
$ sh cuda_9.0.176_384.81_linux.run
```
Specify ~/local/ as installation directory when recieved question such as local/cuda-9.0.  

And install CuDNN like bellow,  
```
$ cd local
$ ln -s cuda-9.0 cuda
$ tar xf cudnn-9.0-linux-x64-v7.5.0.56.tgz
$ rm cuda
```

And setup enviromental variable like bellow,  
```
$ export PATH=~/local/cuda-9.0/bin:$PATH
$ export LD_LIBRARY_PATH=~/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
```
As of this point, can use GPU as GPGPU.  
**In other words, LD_LIBRARY_PATH and PATH decide which version of CUDA and CuDNN will be used**.  

#### Install tensorflow
```
$ pip install tensorflow-gpu==1.11.0
$ python -c "import tensorflow"
$ python
  import tensorflow as tf
  sess=tf.Session()
```
### Why tensorflow-gpu version is 1.11.0 and why CUDA-9.0 and CuDNN-7.5 

First of all, when i installed tensorflow-gpu, i got "not found libcublas.so.9" error message.  
So that install CUDA-9.0 and CuDNN corresponding to CUDA-9.0.  
If you want to use tensorflow-gpu=1.13.1, you have to install CUDA-10.0 and CuDNN corresponding to it.  

#### Check it  
If you got bellow, this is it.  
```
2019-05-07 16:53:07.936004: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-05-07 16:53:08.408057: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1411] Found device 0 with properties:
name: Tesla V100-PCIE-16GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
pciBusID: 0000:20:00.0
totalMemory: 15.78GiB freeMemory: 116.62MiB
2019-05-07 16:53:08.408264: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1490] Adding visible gpu devices: 0
2019-05-07 16:53:09.479217: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-05-07 16:53:09.479298: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977]      0
2019-05-07 16:53:09.479317: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990] 0:   N
2019-05-07 16:53:09.479758: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1103] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 30 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:20:00.0, compute capability: 7.0)
```

**07.May,2019**
