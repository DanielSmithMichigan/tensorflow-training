import tensorflow as tf
import numpy as np

x_0 = np.random.multivariate_normal(
    mean=np.array((-1, -1)),
    cov=.1*np.eye(2),
    size=(50,)
)
y_0 = np.zeros(50)
x_1 = np.random.multivariate_normal(
    mean=np.array((1, 1)),
    cov=.1*np.eye(2),
    size=(50,)
)
y_1 = np.ones(50)
x = np.concatenate([x_0, x_1])
y = np.concatenate([y_0, y_1])

x_placeholder = tf.placeholder(tf.float32, (50, 2))
y_placeholder = tf.placeholder(tf.float32, (50,))
w = tf.Variable(tf.random_normal(0, 1))
b = tf.Variable(0)
y_logit = tf.squeeze(tf.matmul(x, w) + b)
y_norm = tf.normal(y_logit)
y_prediction = tf.round(y_norm)
entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=y_logit,
    labels=y
)
l=tf.reduce_sum(entropy)
train_op = tf.train.AdamOptimizer(.001).minimize(l)
train_writer = tf.summary.FileWriter('/tmp/logistic-train', tf.get_default_graph())
tf.summary.scalar("loss", l)
merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        _, summary, loss = sess.run([train_op, merged, l], feed_dict={x:x, y:y})
        print("loss: %f" % loss)


