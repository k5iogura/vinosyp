# OpenVINO on RaspberryPI stretch

Download OpenVINO according to [OpenVINO R5 supports RaspberryPI stretch](https://software.intel.com/en-us/articles/OpenVINO-Install-RaspberryPI).

## ARM Single Board Computer

RaspberryPi Model B+  
7inch LCD  
MIPI Camera  

## First move for RaspberryPi
**Before booting,**  

- Install RaspberryPi OS stretch image into SDCard.

- Change LCD upside down  
  To modify upside down of LCD, add line  
  'lcd_rotate=2'  
  in file boot/config.txt, and reboot!  

- For ssh connection  
  On Windows with USB adapter, insert SDCard of Raspi stretch image.  
  touch boot/ssh  
  boot/ssh file contains nothing or free words.  

**After booting,**  

- Change keyboard layout to Japanese

```
# raspi-config  
```
> International Options  
>> Change Keyboard Layout  
>>> Generic 105-key(Intel) PC  
>>>> Other  
>>>>>  Japanese  
>>>>>> Japanese - (OADG 109A)  
>>>>>>> The default for the keyboard layout  
>>>>>>>> No compose key  
>>>>>>>>> Finish  

too deep!  

```
# apt update
# apt install -y uim uim-anthy  
# reboot  
```

- Expand filesystem

```
# raspi-config --expand-rootfs
# reboot
```

- Upgrade  

```
# apt update && apt upgrade && reboot
```

- sshd  

Generate ssh keys if needs on such as jessi,,  
```
# cd /etc/ssh
# ssh-keygen -f /etc/ssh/ssh_host_rsa_key -N '' -t rsa
# ssh-keygen -f /etc/ssh/ssh_host_dsa_key -N '' -t dsa
# ssh-keygen -f /etc/ssh/ssh_host_ecdsa_key -N '' -t ecdsa
# ssh-keygen -f /etc/ssh/ssh_host_ed25519_key -N '' -t ed25519
```

```
# systemctl enable ssh
# systemctl restart ssh
# reboot
```

- Install VBoxLinuxAdditions

mount cdrom from Pulldown Menu and bellow,
```
# sh /media/cdrom/VBoxLinuxAdditions.run
# reboot
```

- Resize Display if need,

```
# xrandr
   800x600       60.00*+  60.32  
   2560x1600     59.99  
   1920x1440     60.00  
   1856x1392     60.00  
   1792x1344     60.00  
   1920x1200     59.88  
   1600x1200     60.00  
   1680x1050     59.95  
   1400x1050     59.98  
   1280x1024     60.02  
   1440x900      59.89  
   1280x960      60.00  
   1360x768      60.02  
   1280x800      59.81  
   1152x864      75.00  
   1280x768      59.87  
   1024x768      60.00  
   640x480       59.94  

# xrandr -s 1024x768
```

## Install OpenVINO R5
[Download l_openvino_toolkit_ie_p_2018.5.445.tgz](https://download.01.org/openvinotoolkit/2018_R5/packages/l_openvino_toolkit_ie_p_2018.5.445.tgz) or latest version from here  

```
$ wget https://download.01.org/openvinotoolkit/2018_R5/packages/l_openvino_toolkit_ie_p_2018.5.445.tgz
$ tar xzf l_openvino_toolkit_ie_p_2018.5.445.tgz
$ sed -i "s|<INSTALLDIR>|$(pwd)/inference_engine_vpu_arm|"  inference_engine_vpu_arm/bin/setupvars.sh
$
```

- setup variables

```
$ . inference_engine_vpu_arm/bin/setupvars.sh
[setupvars.sh] OpenVINO environment initialized
$ sudo usermod -a -G users "$(whoami)"
$
```
Let you think return value of "$(whoami)" is ether pi or root!

- setup USB rule

```
$ sh inference_engine_vpu_arm/install_dependencies/install_NCS_udev_rules.sh
Update udev rules so that the toolkit can communicate with your neural compute stick
[install_NCS_udev_rules.sh] udev rules installed
$
```
Flush!! but don't warry,,  

- build sample

```
# apt install -y cmake make docker docker.io
# which docker
/usr/bin/docker
```
```
$ cd inference_engine_vpu_arm/deployment_tools/inference_engine/samples
$ mkdir build && cd build
$ 
```