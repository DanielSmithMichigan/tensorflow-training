import tensorflow as tf
import numpy as np
import gym
import math

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)
tf.set_random_seed(1)

numStateVariables = 4
numActions = 2

learningRate = .01
gamma = 0.95
layerDepth = [16, 16, 16]
__MAX_TRAINING_EPISODE = 2
__TRAINING_STEPS_PER_EPISODE = 2
__RENDERING = True

l_inputStateAction = tf.placeholder(tf.float32, shape=(None, None, numStateVariables + 1))
cell = tf.nn.rnn_cell.BasicLSTMCell(layerDepth[0], state_is_tuple=True)
batch_size = tf.shape(l_inputStateAction)[1]
initial_state = cell.zero_state(batch_size, tf.float32)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, l_inputStateAction, initial_state=initial_state, time_major=True)



l_dense_1 = tf.layers.dense(inputs=rnn_outputs, units=layerDepth[2], activation=tf.nn.relu)
l_predictions = tf.layers.dense(inputs=l_dense_1, units=numStateVariables + 1)
l_stateRewards = tf.placeholder(tf.float32, shape=(None, None, numStateVariables + 1))
l_loss = tf.losses.mean_squared_error(labels=l_stateRewards,predictions=l_predictions)
trainingOperation = tf.train.AdamOptimizer(learningRate).minimize(l_loss)

def randomAction():
	if np.random.rand(1) < 0.5:
		return 1
	return 0

stateActions = []
stateRewards = []

steps = 0

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for episode in range(__MAX_TRAINING_EPISODE):
		state = env.reset()
		if __RENDERING:
			env.render()
		done = False
		for step in range(__TRAINING_STEPS_PER_EPISODE):
			currentAction = randomAction()
			# stateActions.append(np.concatenate(state, np.array([currentAction])))
			newState, reward, done, _ = env.step(currentAction)
			# stateRewards.append(np.concatenate(newState, np.array([reward])))
			if __RENDERING:
				env.render()
			if (done):
				break
			state = newState
	lossValue, _, _ = sess.run([l_loss, trainingOperation], feed_dict={
		l_stateRewards: stateRewards,
		l_inputStateAction: stateActions
	})


