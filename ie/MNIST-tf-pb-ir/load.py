import tensorflow as tf
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def main():
    for idx, img in enumerate(mnist.test.images):
        img = img[np.newaxis,:].transpose((1,0)).reshape((28,-1,1))
        int_label=np.argmax(mnist.test.labels[idx])
        print("label",int_label)
        cv2.imshow('digits',img)
        if cv2.waitKey(0)==27:break
if __name__=='__main__':
    main()
