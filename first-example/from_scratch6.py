import tensorflow as tf
import numpy as np
x_zeros = np.random.multivariate_normal(mean=np.array((-1, -1)), cov=.1*np.eye(2), size=(50,))
y_zeros = np.zeros((50,))
x_ones = np.random.multivariate_normal(mean=np.array((1, 1)), cov=.1*np.eye(2), size=(50,))
y_ones = np.ones((50,))
x_set = np.vstack([x_zeros, x_ones])
y_set = np.concatenate([y_zeros, y_ones])

x = tf.placeholder(tf.float32, (100, 2))
y = tf.placeholder(tf.float32, (100, ))
w = tf.Variable(tf.random_normal((2, 1)))
b = tf.Variable(tf.random_normal((1, )))
y_logit = tf.squeeze(tf.matmul(x, w) + b)
y_norm = tf.sigmoid(y_logit)
y_pred = tf.round(y_norm)
entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_logit)
l = tf.reduce_sum(entropy)
train_op = tf.train.AdamOptimizer(.01).minimize(l)
train_writer = tf.summary.FileWriter('/tmp/logistic-train', tf.get_default_graph())
tf.summary.scalar("loss", l)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
          _, summary, loss = sess.run([train_op, merged, l], feed_dict={x: x_set, y: y_set})
          print("loss: %f" % loss)
          train_writer.add_summary(summary, i)