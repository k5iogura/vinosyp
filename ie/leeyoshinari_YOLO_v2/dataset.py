#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:leeyoshinari
#------------------------------------------------------------------------------------
import os
import cv2
import numpy as np
import yolo.config as cfg
import xml.etree.ElementTree as ET

class Dataset(object):
    def __init__(self):
        self.image_size = cfg.IMAGE_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.box_per_cell = cfg.BOX_PRE_CELL
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

        self.count = 0
        self.epoch = 1
        self.count_t = 0

        if False:   # Default Paths
            self.data_path       = os.path.join('data','Pascal_voc','VOCdevkit','VOC2007')
            self.train_txt_path  = os.path.join(self.data_path, 'ImageSets','Main', 'trainval.txt')
            self.test_txt_path   = os.path.join(self.data_path, 'ImageSets','Main', 'test.txt')
            self.image_path      = os.path.join(self.data_path, 'JPEGImages')
            self.annotation_path = os.path.join(self.data_path, 'Annotations')

    def preLoad(self):
        self.train_label = self.load_labels('train')
        self.test_label  = self.load_labels('test')
        print('preLoad:',self.train_txt_path,len(self.train_label))
        print('preLoad:',self.test_txt_path, len(self.test_label))

    def takeIn(self, Ds_object):
        self.train_label.extend(Ds_object.train_label)
        self.test_label.extend(Ds_object.test_label)
        print('takeIn:',Ds_object.train_txt_path,len(self.train_label))
        print('takeIn:',Ds_object.test_txt_path, len(self.test_label))
        return self

    def load_labels(self, model):
        txtname = self.train_txt_path if model == 'train' else self.test_txt_path

        with open(txtname, 'r') as f:
            image_ind = [x.strip() for x in f.readlines()]

        labels = []
        for ind in image_ind:
            label, num = self.load_data(ind)
            if num == 0:
                continue
            imagename = os.path.join(self.image_path, ind + '.jpg')
            labels.append({'imagename': imagename, 'labels': label})
        np.random.shuffle(labels)
        return labels


    def load_data(self, index):
        label    = np.zeros([self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes])
        filename = os.path.join(self.annotation_path, index + '.xml')
        tree = ET.parse(filename)
        image_size   = tree.find('size')
        image_width  = float(image_size.find('width').text)
        image_height = float(image_size.find('height').text)
        h_ratio = 1.0 * self.image_size / image_height
        w_ratio = 1.0 * self.image_size / image_width

        objects = tree.findall('object')
        for obj in objects:
            box = obj.find('bndbox')
            x1 = max(min((float(box.find('xmin').text)) * w_ratio, self.image_size), 0)
            y1 = max(min((float(box.find('ymin').text)) * h_ratio, self.image_size), 0)
            x2 = max(min((float(box.find('xmax').text)) * w_ratio, self.image_size), 0)
            y2 = max(min((float(box.find('ymax').text)) * h_ratio, self.image_size), 0)
            class_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            boxes = [0.5 * (x1 + x2) / self.image_size, 0.5 * (y1 + y2) / self.image_size, np.sqrt((x2 - x1) / self.image_size), np.sqrt((y2 - y1) / self.image_size)]
            cx = 1.0 * boxes[0] * self.cell_size
            cy = 1.0 * boxes[1] * self.cell_size
            xind = int(np.floor(cx))
            yind = int(np.floor(cy))
            
            label[yind, xind, :, 0] = 1
            label[yind, xind, :, 1:5] = boxes
            label[yind, xind, :, 5 + class_ind] = 1

        return label, len(objects)


    def next_batches(self, label):
        images = np.zeros([self.batch_size, self.image_size, self.image_size, 3])
        labels = np.zeros([self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes])
        num = 0
        while num < self.batch_size:
            imagename = self.train_label[self.count]['imagename']
            images[num, :, :, :] = self.image_read(imagename)
            labels[num, :, :, :, :] = self.train_label[self.count]['labels']
            num += 1
            self.count += 1
            if self.count >= len(self.train_label):
                np.random.shuffle(self.train_label)
                self.count = 0
                self.epoch += 1
        return images, labels


    def next_batches_test(self, label):
        images = np.zeros([self.batch_size, self.image_size, self.image_size, 3])
        labels = np.zeros([self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes])
        num = 0
        while num < self.batch_size:
            imagename = self.test_label[self.count_t]['imagename']
            images[num, :, :, :] = self.image_read(imagename)
            labels[num, :, :, :, :] = self.test_label[self.count_t]['labels']
            num += 1
            self.count_t += 1
            if self.count_t >= len(self.test_label):
                self.count_t = 0
        return images, labels


    def image_read(self, imagename):
        image = cv2.imread(imagename)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0 * 2.0 - 1.0
        return image

class Pascal_voc_VOC2007(Dataset):
    def __init__(self):
        super(Pascal_voc_VOC2007,self).__init__()
        self.data_path       = os.path.join('data','Pascal_voc','VOCdevkit','VOC2007')
        self.train_txt_path  = os.path.join(self.data_path, 'ImageSets','Main', 'trainval.txt')
        self.test_txt_path   = os.path.join(self.data_path, 'ImageSets','Main', 'test.txt')
        self.image_path      = os.path.join(self.data_path, 'JPEGImages')
        self.annotation_path = os.path.join(self.data_path, 'Annotations')
        self.preLoad()

class Pascal_voc_VOC2012(Dataset):
    def __init__(self):
        super(Pascal_voc_VOC2012,self).__init__()
        self.data_path       = os.path.join('data','Pascal_voc','VOCdevkit','VOC2012')
        self.train_txt_path  = os.path.join(self.data_path, 'ImageSets','Main', 'trainval.txt')
        self.test_txt_path   = os.path.join(self.data_path, 'ImageSets','Main', 'val.txt')
        self.image_path      = os.path.join(self.data_path, 'JPEGImages')
        self.annotation_path = os.path.join(self.data_path, 'Annotations')
        self.preLoad()