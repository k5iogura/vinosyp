# TinyYOLOv2_352x288 for AI-Chips via Tensorflow

Can run yolov2-tiny with HW = 288x352 featuremap on some device such 
 as CPU, Myriad2.  
yolov2-tiny-352x288 model is faster than 416x416 model.  

![bases on github](https://github.com/simo23/tinyYOLOv2)  

## flow  

- Install required packages  
  Tensorflow, tfdebugger, OpenVINO  
  ![Reference about OpenVINO installation](https://github.com/k5iogura/vinosyp/blob/master/README.md)  
  Use pip3 for installation of tensorflow  
  Use apt for eog or feh  
  
- Prepare data  
  ![yolov2-tiny-voc_352_288_final.weights](https://github.com/k5iogura/darknet_a10/tree/master/model)  
  
```
  $ cat yolov2-tiny-voc_352_288_final.weights.* > yolov2-tiny-voc_352_288_final.weights
```

- Create y.pb by test.py  

```
  $ python3 test.py
```

- Using UVC Camera  
Connect UVC Camera via USB port.  
```
  $ python3 test_webcam.py
```

- Optimize y.pb for OpenVINO FP32 and FP16  

```
  $ export mo=/opt/intel/openvino/deployment_tools/model_optimizer/
  $ $mo/mo_tf.py --input_model y.pb \
    --output xoutput \
    --data_type FP16 \
    --output_dir FP16
```
  Issue mo_tf.py with both FP16 and FP32.  
  
- Run Demo with FP32 or FP16  

```
  $ python3 tinyyolov2_predict.py person.jpg -d CPU
  $ python3 tinyyolov2_predict.py person.jog -d MYRIAD
```

- Check result  

```
  $ eog person_result.png
```

![](person_result.png)
Why two dogs!:smile:  

