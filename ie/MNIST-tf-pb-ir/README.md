# With MNIST Training Tensorflow and Prediction OpenVINO

### Abstract

This is feasibility study of execution Tensorflow training network via OpenVINO with CPU or NCS devices using simple example '**MNIST**'.  
Tensorflow can discript low level operation such as x, +, - versus tensor.  
By transforming Tensorflow saving model file to OpenVINO IR model file and executes network on CPU or NCS, we **ensure its deployment flow**.  

### On This Repository

For easy testing,,, 
```
$ cd
$ git clone https://github.com/k5iogura/vinosyp
$ cd vinosyp/ie/MNIST-tf-pb-ir
$ ./first_mode.sh
```

- Downloads MNIST Dataset via Internet
- Setups OpenVINO Inference_engine
- Runs Inference_engine with python3
- Estimates of result prediction

```
$ ./first_move.sh 
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting MNIST_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
./FP16/mnist.bin on MYRIAD
[1, 784]
input_blob shape(from xml) [1, 784]
name (input_blob : out_blob) = input : output
predict/label = ( 7 / 7 ) Pass
predict/label = ( 2 / 2 ) Pass
predict/label = ( 1 / 1 ) Pass
predict/label = ( 0 / 0 ) Pass
predict/label = ( 4 / 4 ) Pass
predict/label = ( 1 / 1 ) Pass
predict/label = ( 4 / 4 ) Pass
predict/label = ( 9 / 9 ) Pass
predict/label = ( 6 / 5 ) NG
predict/label = ( 9 / 9 ) Pass
...
```

### MNIST Networks and Training via Tensorflow

In mnist_train.py,,

```
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])
```

Fullconnection network only used.  

For training via Tensorflow with CPU,,
```
$ python3 mnist_train.py
0.9164
```
### If Wanna Checking 28x28 testing images via OpenCV...
```
$ python3 load.py
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
label 7
...
```
![](7.png)  

**Apr 15, 2019**
