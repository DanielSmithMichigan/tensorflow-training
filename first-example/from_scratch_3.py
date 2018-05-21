import tensorflow as tf
import numpy as np

x_zero = np.random.multivariate_normal(mean=(-1, -1), cov=.1*np.eye(2), size=(50,))
y_zero = np.zeros(50)
x_one = np.random.multivariate_normal(mean=(1, 1), cov=.1*np.eye(2), size=(50,))
y_one = np.ones(50)
x = np.concatenate([x_zero, x_one])
y = np.concatenate([y_zero, y_one])

x_input = tf.placeholder(dtype=float32, shape=(100, 2))
y_input = tf.placeholder(dtype=float32, shape=(100, ))
w = tf.Variable(np.random_normal(size=(2, 1)))
b = tf.Variable(tf.zeros((1,)))