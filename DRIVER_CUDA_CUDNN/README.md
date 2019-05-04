# Installation of NVIDIA as GPGPU  

### Prepare  
Centos7.5  
cuda_10.0.130_410.48_linux.run  
cudnn-10.0-linux-x64-v7.5.0.56.tgz  

### Check NVIDIA GPU on machine.  
```
$ lspci | grep -i nvidia
01:00.0 VGA compatible controller: NVIDIA Corporation GP107 [GeForce GTX 1050 Ti] (rev a1)
01:00.1 Audio device: NVIDIA Corporation GP107GL High Definition Audio Controller (rev a1)
```
**GTX1050Ti** here!  

### Disable noubeau  

On graphical mode,  
```
  # vi /etc/modprobe.d/blacklist-nouveau.conf
  # ln blacklist-nouveau.conf nouveau-blacklist.conf

    blacklist nouveau
    options nouveau modeset=0

  # dracut --force

  # systemctl set-default multi-user.target
  # reboot
```
Into CLI login mode.  

### Runfile Installer  
```
  # sh cuda_10.0.130_410.48_linux.run --silent
```
After a few minites, return to prompting.  

### Setup xconfig
```
  # nvidia-xconfig
  $ ls /etc/X11/
    Xmodmap  Xresources  applnk  fontpath.d  xinit  xorg.conf  xorg.conf.backup  xorg.conf.d  xorg.conf.nvidia-xconfig-original

  # systemctl set-default graphical.target
  # reboot
```
Into graphical login mode.  

### Install cuDNN  

```
  # cd /usr/local
  # tar xzf cudnn-10.0-linux-x64-v7.5.0.56.tgz
  $ ls /usr/local/cuda/lib64/*cudnn*
    /usr/local/cuda/lib64/libcudnn.so    /usr/local/cuda/lib64/libcudnn.so.7.5.0
    /usr/local/cuda/lib64/libcudnn.so.7  /usr/local/cuda/lib64/libcudnn_static.a

```

### Check installations of CUDA and cuDNN using darknet
```
  $ git clone https://github.com/pjreddie/darknet
  $ cd darknet
  $ vi Makefile
    GPU   = 1
    CUDNN = 1
  $ make -j4
  $ objdump -p darknet | grep lib
  NEEDED               libm.so.6
  NEEDED               libcuda.so.1
  NEEDED               libcudart.so.10.0
  NEEDED               libcublas.so.10.0
  NEEDED               libcurand.so.10.0
  NEEDED               libcudnn.so.7
  NEEDED               libstdc++.so.6
  NEEDED               libpthread.so.0
  NEEDED               libc.so.6
```

```
  $ vi example/darknet
    cfg/voc.data
  $ make -j4
  $ wget https://pjreddie.com/media/files/yolov2-voc.weights
  $ ./darknet detect cfg/yolov2-voc.cfg yolov2-voc.weights data/dog.jpg
```
Check GPU usage with nvidia-smi command.

Predict yolov2 on GPU with UVC Camera.  
```
  $ pip install opencv-devel
  $ ./darknet detector demo cfg/voc.data cfg/yolov2-voc.cfg yolov2-voc.weights
  FPS:23.7
```

**May.04, 2019**  
