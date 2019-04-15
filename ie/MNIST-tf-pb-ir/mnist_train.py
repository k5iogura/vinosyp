import load
mnist = load.mnist

import tensorflow as tf
import cv2

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# Save
graph=tf.Graph()

with graph.as_default():
    x_2 = tf.placeholder(tf.float32, [1, 784],name="input")
    W_2 = tf.constant(sess.run(W),name="Weight")
    b_2 = tf.constant(sess.run(b),name="bias")
    y_2 = tf.nn.softmax(tf.matmul(x_2, W_2) + b_2,name="output")
    sess2 = tf.Session()

    sess2.run(tf.global_variables_initializer())

    tf.train.write_graph(graph,".","mnist.pb",as_text=False)
