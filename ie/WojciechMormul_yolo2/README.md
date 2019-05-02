# You Only Look Once

[original site](https://github.com/WojciechMormul/yolo2)  

1.Train YOLOv2 object detector from scratch with Tensorflow.

## Usage
Prepare two files: 

data.csv (three columns: filenames, rois, classes - each row contains image filepath, list of rois (each [x,y,w,h]), list of classes) and anchors.txt (each row contains width and height of one anchor).

How to make data.csv  
```
  $ wget {somewhere}/VOCtest_06-Nov-2007.tar
  $ tar xf VOCtest_06-Nov-2007.tar
  $ python voc_labels.py
  $ python find.py > data.csv
```

data.csv_example  
```
filename,rois,classes
/home/ogura/VOC/VOCdevkit/VOC2007/JPEGImages/006514.jpg,"[[0.513,0.728,0.392,0.540],[0.519,0.657,0.333,0.494]]","[14,8]"
/home/ogura/VOC/VOCdevkit/VOC2007/JPEGImages/001850.jpg,"[[0.606,0.592,0.784,0.438],[0.531,0.754,0.054,0.258],[0.482,0.769,0.060,0.216],[0.481,0.727,0.058,0.348]]","[0,14,14,14]"
/home/ogura/VOC/VOCdevkit/VOC2007/JPEGImages/008003.jpg,"[[0.506,0.621,0.940,0.691],[0.527,0.456,0.462,0.866]]","[13,14]"
/home/ogura/VOC/VOCdevkit/VOC2007/JPEGImages/000658.jpg,"[[0.721,0.827,0.554,0.341]]","[17]"
/home/ogura/VOC/VOCdevkit/VOC2007/JPEGImages/004596.jpg,"[[0.497,0.502,0.995,0.988]]","[6]"
/home/ogura/VOC/VOCdevkit/VOC2007/JPEGImages/002746.jpg,"[[0.370,0.658,0.688,0.415]]","[6]"
/home/ogura/VOC/VOCdevkit/VOC2007/JPEGImages/009521.jpg,"[[0.970,0.605,0.056,0.208]]","[19]"
/home/ogura/VOC/VOCdevkit/VOC2007/JPEGImages/001162.jpg,"[[0.683,0.600,0.586,0.539]]","[17]"
/home/ogura/VOC/VOCdevkit/VOC2007/JPEGImages/006571.jpg,"[[0.301,0.443,0.506,0.453],[0.611,0.447,0.586,0.893]]","[7,11]"
...
```

```
python make_tfrecord.py
python train.py
python test.py
```

<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/neta.png" width="400">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/netb.png" width="400">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/netc.png" width="400">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/loss2.png" width="300">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/merge.png" width="700">

2.Evaluate YOLOv2 model trained with COCO dataset using Tensorflow. Conversion from Darknet to Tensorflow framework done with darkflow project.

<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/a2.png" width="400">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/a4.png" width="290">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/r1.png" width="700">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/r2.png" width="700">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/r3.png" width="700">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/r4.png" width="700">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/r5.png" width="700">

