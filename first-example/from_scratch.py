import tensorflow as tf
import numpy as np

N = 100
# x_zeros = np.random.multivariate_normal(
#        mean=np.array((-1, -1)),
#        cov=.1*np.eye(2),
#        size=(int(N/2),)
# )
x0 = np.random.multivariate_normal(mean=np.array((-1, -1)), cov=.1 * np.eye(2), size=(50,))
x1 = np.random.multivariate_normal(mean=np.array((1, 1)), cov=.1 * np.eye(2), size=(50,))
y0 = np.zeros(50)
y1 = np.ones(50)
x_set = np.vstack([x0, x1])
y_set = np.vstack([y0, y1])

x = tf.placeholder(tf.float32, (N, 2))
y = tf.placeholder(tf.float32, (N,))
w = tf.Variable(tf.random_normal((2, 1)))
b = tf.Variable(tf.zeros((1,)))
y_logit = tf.squeeze(tf.matmul(x, w) + b)
y_one_prob = tf.norm(y_logit)
y_pred = tf.round(y_one_prob)
entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit,labels=y)
l = tf.reduce_sum(entropy)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(l)
tf.summary.scalar("loss", l)
merged = tf.summary.merge_all()

n_steps=1000
with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       for i in range(n_steps):
              _, summary, loss = sess.run([train_op, merged, l], feed_dict={x: x_set, y: y_set})
              print("loss: %f" % loss)


