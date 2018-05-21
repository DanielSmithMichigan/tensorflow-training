import tensorflow as tf
import numpy as np

x_0 = np.random.multivariate_normal(mean=(-1, -1), cov=.1*np.eye(2), size=(50,))
x_1 = np.random.multivariate_normal(mean=(1, 1), cov=.1*np.eye(2), size=(50,))
x_set = np.vstack([x_0, x_1])
y_0 = np.zeros(50)
y_1 = np.ones(50)
y_set = np.concatenate([y_0, y_1])

x = tf.placeholder(tf.float32, (50, 2))
y = tf.placeholder(tf.float32, (50,))
w = tf.Variable(tf.random_normal((2, 1)))
b = tf.Variable(tf.zeros((1,)))
y_logits = tf.squeeze(tf.matmul(x, w) + b)
y_norm = tf.norm(y_logits)
y_pred = tf.round(y_norm)
entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=y_logits,
    labels=y
)
l = tf.reduce_sum(entropy)
training_op = tf.train.AdamOptimizer(.001).minimize(l)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        feedDict={x: x_set, y: y_set}
        sess.run([l, training_op])