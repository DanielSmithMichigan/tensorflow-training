import tensorflow as tf
import numpy as np

x_zeros = np.random.multivariate_normal(mean=np.array((-1, -1)), cov=.1*np.eye(2), size=(50,))
y_zeros = np.zeros(50)
x_ones = np.random.multivariate_normal(mean=np.array((1,1)), cov=.1*np.eye(2), size=(50,))
y_ones = np.ones(50)
x_data = np.vstack([x_zeros, x_ones])
y_data = np.concatenate([y_zeros, y_ones])

x_var = tf.placeholder(tf.float32, (100,2))
y_var = tf.placeholder(tf.float32, (100,))
w = tf.Variable(tf.random_normal((2, 1)))
b = tf.Variable(tf.zeros((1,)))
y_logits = tf.squeeze(tf.matmul(x_var, w) + b)
y_norm = tf.norm(y_logits)
y_prediction = tf.round(y_norm)
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_var)
l = tf.reduce_sum(entropy)
train_op = tf.train.AdamOptimizer(.01).minimize(l)
train_writer = tf.summary.FileWriter('/tmp/logistic-train', tf.get_default_graph())

tf.summary.scalar("loss", l)
merged = tf.summary.merge_all()

n_steps = 10000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_steps):
        feed_dict = {x: x_data, y: y_data}
        _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
        print("loss: %f" % loss)

