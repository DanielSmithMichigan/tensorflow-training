import tensorflow as tf
import numpy as np

x_0 = np.random.multivariate_normal(mean=np.array((1, 1)), cov=.1*np.eye(2), size=(50,))
x_1 = np.random.multivariate_normal(mean=np.array((-1, -1)), cov=.1*np.eye(2), size=(50,))
y_0 = np.zeros((50,))
y_1 = np.ones((50,))
all_x = np.vstack([x_0, x_1])
all_y = np.concatenate([y_0, y_1])

x = tf.placeholder(dtype=tf.float32, shape=np.array((100, 2)), name="x")
y = tf.placeholder(dtype=tf.float32, shape=np.array((100,)), name="y")
w = tf.Variable(tf.random_normal((2, 1)), dtype=tf.float32)
b = tf.Variable(tf.zeros((1,)), dtype=tf.float32)
y_logit = tf.squeeze(tf.matmul(x, w) + b)
y_pred = tf.round(tf.sigmoid(y_logit))
entropy = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=y,
    logits=y_logit
)
l = tf.reduce_sum(entropy)
train_op = tf.train.AdamOptimizer(.01).minimize(l)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        _, loss = sess.run([train_op, l], feed_dict={x: all_x, y: all_y})
        print("loss: %f" % loss)