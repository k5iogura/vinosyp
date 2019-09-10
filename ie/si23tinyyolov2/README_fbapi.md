# Inference with tflite by numpy only python  

### Requirements  
tensorflow  v1.11 or later  
feh or eog imagemagick some image viewer    
flatbuffers   

### How to test
- **Create y.pb(include YOLOv2-Tiny network and weights)**  
- **Convert to frozen.pb**  
- **Prepare tflite flatbuffers python modules**  
- **Run inference with y.tflite.**  

```
 $ wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights
 $ ls yolov2-tiny-voc.weights
   yolov2-tiny-voc.weitghts
   
 $ python test_pb.py
 $ ls y.pb
   y.pb
   
 $ tflite_convert --output_file=y.tflite --graph_def_file=y.pb --inference_type=FLOAT
                  --inference_input_type=FLOAT --input_arrays=input/Placeholder --output_arrays=xoutput
 $ ls y.tflite
   y.tflite
 
 $ ./make_tflite.sh
 $ ls -d tflite
   tflite/
   
 $ python fbapi.py
 $ feh result.jpg
```

![](./result.jpb)

**Sep.10, 2019**
