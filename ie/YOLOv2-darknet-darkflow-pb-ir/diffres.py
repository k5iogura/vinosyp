#!/usr/bin/env python3
from pdb import *

import sys,os
import numpy as np
from time import time
import cv2
import argparse

img_sizeWH = [
    [640,480],
    [640,480],
]

def read_box(filename):
    with open(filename) as fp:
        _list = fp.read().split()
        _list = np.asarray(_list,dtype=np.float32).reshape(-1,5)
    return _list

def pixscale(_list, scale):
    scaled_list = []
    for l in _list:
        scaled_list.append(l[0])
        scaled_list.extend(l[1:3]*scale)
        scaled_list.extend(l[3:5]*scale)
    return np.asarray(scaled_list,dtype=np.float32).reshape(-1,5)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def check_acc(file0, file1):
    list0 = read_box(file0)
    list1 = read_box(file1)
    list0 = pixscale(list0, img_sizeWH[0])
    list1 = pixscale(list1, img_sizeWH[1])
    print(list0)
    print(list1)
    iou_max=0.
    nearest_boxes = []
    nearest_box   = []
    for l1 in list1:
        for l0 in list0:#GT
            iou_box = iou(l0[1:], l1[1:])
            if iou_box > iou_max:
                iou_max = iou_box
                nearest_box=l1
                print(iou_max)
        nearest_boxes.append(nearest_box)
    print(nearest_boxes)

def main(args):
    check_acc(args.gt, args.inferred)
    pass

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("gt", type=str)
    args.add_argument("inferred", type=str)
    args = args.parse_args()

    sys.exit(main(args))
