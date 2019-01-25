#!/bin/bash
sudo usermod -a -G users "$(whoami)"
sudo cp 97-usbboot.rules /etc/udev/rules.d/
ls /etc/udev/rules.d
udevadm control --reload-rules
udevadm trigger
ldconfig

echo "reboot to update!"

