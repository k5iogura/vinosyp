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

