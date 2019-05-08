from pdb import set_trace
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

import load
mnist = load.mnist
fashion_mnist = keras.datasets.fashion_mnist

def ys_x10(test_ys):
    Wlist = []
    for i in test_ys:
        zeros10=[0.0]*10
        zeros10[int(i)]=1.0
        Wlist.append(zeros10)
    return np.asarray(Wlist,dtype=np.float)

def modelorg(x):
    W1 = tf.Variable(tf.zeros([784, 10]))
    b1 = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W1) + b1
    return y

def model2ly(x):
    W1 = tf.Variable(tf.zeros([784, 128]),name='f1')
    b1 = tf.Variable(tf.zeros([128]))
    W2 = tf.Variable(tf.zeros([128, 10]),name='f2')
    b2 = tf.Variable(tf.zeros([10]))
    y1 = tf.matmul(x, W1) + b1
    y  = tf.matmul(tf.nn.relu(y1),W2) + b2
    return y

x = tf.placeholder(tf.float32, [None, 784])
with tf.name_scope('net'):
    #y = modelorg(x)
    y = model2ly(x)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
(Zbatch_xs, Zbatch_ys),(Ztest_xs,Ztest_ys) = fashion_mnist.load_data()

Zbatch_xs = Zbatch_xs.reshape(-1,28*28)/255.
Ztest_xs  = Zbatch_xs.reshape(-1,28*28)/255.

Wlist = []
for i in Zbatch_ys:
    zeros10=[0.0]*10
    zeros10[int(i)]=1.0
    Wlist.append(zeros10)
Zbatch_ys = np.asarray(Wlist,dtype=np.float)

Wlist = []
for i in Ztest_ys:
    zeros10=[0.0]*10
    zeros10[int(i)]=1.0
    Wlist.append(zeros10)
Ztest_ys = np.asarray(Wlist,dtype=np.float)

batch_size = 100
for i in range(1000):
  Xbatch_xs, Xbatch_ys = mnist.train.next_batch(batch_size)
  idx = ( np.random.rand(batch_size)*Zbatch_xs.shape[0] ).astype(np.int)
  batch_xs = Zbatch_xs[idx]
  batch_ys = Zbatch_ys[idx]
  #set_trace()
  sess.run(train_step, feed_dict={x: Xbatch_xs, y_: Xbatch_ys})
  #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print(y.shape)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

