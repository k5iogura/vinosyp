# TinyYOLOv2_352x288 for AI-Chips via Tensorflow

[original readme here](./README_original.md)  

## flow  

- Install required packages  
  Tensorflow, tfdebugger, OpenVINO  
  ![Reference about OpenVINO installation](https://github.com/k5iogura/vinosyp/blob/master/README.md)  
  Use pip3 for installation of tensorflow
  
- Prepare data  
  yolov2-tiny-voc_352_288_final.weights  
  
- Create y.pb by test.py  

```
  $ python3 test.py
```

- Optimize y.pb for OpenVINO by test.pb  

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
  $ python3 tinyyolov2_predict.py person.jpg
```

- Check result  

```
  $ eog person_result.png
```

![](person_result.png)
