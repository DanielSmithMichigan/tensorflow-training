import deepchem as dc
import tensorflow as tf

_, (train, valid, test), _ = dc.molnet.load_tox21()
train_x, train_y, train_w = train.X, train.y, train.w
valid_x, valid_y, valid_w = valid.X, valid.y, valid.w
test_x, test_y, test_w = test.X, test.y, test.w

train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

d = 1024
n_hidden = 1
keep_prob = .5
dropout_prob = .5
learning_rate = 0.001
n_epochs = 50
N = 100
batch_size = 50


x = tf.placeholder(tf.float32, (None, d))
y = tf.placeholder(tf.float32, (None,))

w = tf.Variable(tf.random_normal((d, n_hidden)))
b = tf.Variable(tf.random_normal((n_hidden,)))
x_hidden = tf.nn.relu(tf.matmul(x, w) + b)
x_hidden = tf.nn.dropout(x_hidden, keep_prob)
w_output = tf.Variable(tf.random_normal((n_hidden, 1)))
b_output = tf.Variable(tf.random_normal((1,)))
y_logit = tf.matmul(x_hidden, w_output) + b_output
y_one_prob = tf.sigmoid(y_logit)
y_pred = tf.round(y_one_prob)

y_expand = tf.expand_dims(y, 1)
entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
l = tf.reduce_sum(entropy)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)
tf.summary.scalar("loss", l)
merged = tf.summary.merge_all()

step = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        pos = 0
        while pos < N:
            batch_x = train_x[pos:pos+batch_size]
            batch_y = train_y[pos:pos+batch_size]
            feed_dict = {x: batch_x, y: batch_y, keep_prob: dropout_prob}
            _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
            print("epoch %d, step %d, loss %f" % (epoch, step, loss))
            train_writer.add_summary(summary, step)
            step += 1
            pos += batch_size