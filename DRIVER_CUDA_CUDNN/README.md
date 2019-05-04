# Installation of NVIDIA  

### Disable noubeau  
```
  # vi /etc/modprobe.d/blacklist-nouveau.conf
  # ln blacklist-nouveau.conf nouveau-blacklist.conf

    blacklist nouveau
    options nouveau modeset=0

  # dracut --force

  # systemctl set-default multi-user.target
  # reboot
```
Into no graphical login mode.  

### Runfile Installer  
```
  $ sudo sh cuda_<version>_linux.run --silent
```

### Setup xconfig
```
  # nvidia-xconfig
  # systemctl set-default graphical.target
  # reboot
```
Into graphical login mode.  

### Install cuDNN  

```
  # cd /usr/local
  # tar xzf cudnn-10.0-linux-x64-v7.5.0.56.tgz
```

### Check CUDA and cuDNN with darknet
```
  $ git clone https://github.com/pjreddie/darknet
  $ cd darknet
  $ vi Makefile
    GPU   = 1
    CUDNN = 1
  $ make -j4
  $ objdump -p ~/darknet/darknet | grep lib
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
