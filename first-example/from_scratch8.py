import tensorflow as tf
import numpy as np

x_zeros = np.random.multivariate_normal(mean=np.array((1, 1)), cov=.1*np.eye(2), size=(50,))
x_ones = np.random.multivariate_normal(mean=np.array((-1, -1)), cov=.1*np.eye(2), size=(50,))
y_zeros = np.zeros((50,))
y_ones = np.ones((50,))
all_x = np.vstack([x_zeros, x_ones])
all_y = np.concatenate([y_zeros, y_ones])

x = tf.placeholder(dtype=tf.float32, shape=(100,2))
y = tf.placeholder(dtype=tf.float32, shape=(100,))
w = tf.Variable(tf.random_normal((2, 1)), dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)
y_logit = tf.squeeze(tf.matmul(x, w) + b)
entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
l = tf.reduce_sum(entropy)
train_op = tf.train.AdamOptimizer(.01).minimize(l)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        _, loss = sess.run([train_op, l], feed_dict={x: all_x, y: all_y})
        print("loss: %f" % loss)