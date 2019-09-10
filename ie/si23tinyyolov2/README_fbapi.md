# Inference with tflite by numpy only python  

### Requirements  

### Test  
Create y.pb(include YOLOv2-Tiny network and weights)  
Convert to frozen.pb  
Prepare tflite flatbuffers python modules  
Run inference with y.tflite.  

```
 $ wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights
 $ python test_pb.py
 $ tflite_convert --output_file=y.tflite --graph_def_file=y.pb --inference_type=FLOAT
                  --inference_input_type=FLOAT --input_arrays=input/Placeholder --output_arrays=xoutput
 $ ./make_tflite.sh
 $ python fbapi.py
```
