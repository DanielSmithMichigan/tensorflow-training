import tensorflow as tf
import numpy as np
import gym
import math

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)
tf.set_random_seed(1)

numStates = 4
numActions = 2
numResults = 20

currentLearningRate = .01
learningRateSteps = [400, 800]
gamma = 0.95
layers = [10, 10]
maxPoleAngle = 3.0 # max is 12
maxCartPosition = 1.2 #max is 2.4
maxVelocityCart = 1.0
maxVelocityPole = 1.0
episodesPerResult = 1200


def velocityMatchesAngle(state):
	cartVelocity = state[1]
	poleVelocity = state[3]
	percentCart = min(abs(cartVelocity) / maxVelocityCart, 1)
	percentPole = min(abs(poleVelocity) / maxVelocityPole, 1)
	signCart = cartVelocity / abs(cartVelocity)
	signPole = poleVelocity / abs(poleVelocity)
	return min(max(percentCart / percentPole * signCart * signPole, -1), 1)

def getReward(state):
	cartPosition = state[0]
	cartVelocity = state[1]
	poleAngle = state[2]
	poleVelocity = state[3]
	rewardArr = [
		min(abs(cartPosition) / maxCartPosition, 1),
		min(abs(cartVelocity) / maxVelocityCart, 1),
		min(abs(poleAngle) / maxPoleAngle, 1),
		min(abs(poleVelocity) / maxVelocityPole, 1),
		# velocityMatchesAngle(state)
	]
	return np.mean(rewardArr) + 1

def getRewards(_states):
	_rewards = []
	cumulative = 0
	for i in reversed(range(len(_states))):
		reward = getReward(_states[i])
		cumulative = cumulative * gamma + reward
		_rewards = [cumulative] + _rewards
	mean = np.mean(_rewards)
	stddev = np.std(_rewards)
	_rewards = (_rewards - mean) / (stddev + .0000001)
	return _rewards

def getOneHot(size, pos):
	x = np.zeros(size)
	np.put(x, pos, 1)
	return x

learningRate = tf.placeholder(tf.float32, shape=[])
with tf.name_scope("input-layer"):
	states = tf.placeholder(tf.float32, [None, numStates])

with tf.name_scope("hidden-layers"):
	dense1 = tf.layers.dense(inputs=states, units=layers[0], activation=tf.nn.relu)
	# dense1_summary = tf.summary.image("Dense1", dense1)
	# dense2 = tf.layers.dense(inputs=dense1, units=layers[1], activation=tf.nn.relu)
	# dense2_summary = tf.summary.image("Dense2", dense2)

with tf.name_scope("output"):
	actionLogits = tf.layers.dense(inputs=dense1, units=numActions, activation=tf.nn.relu)
	suggestedAction = tf.nn.softmax(actionLogits)

with tf.name_scope("training"):
	actions = tf.placeholder(tf.int32, [None, numActions])
	entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=actionLogits,labels=actions)
	rewards = tf.placeholder(tf.float32, [None, ])
	loss = tf.reduce_mean(entropy * rewards)
	train_op = tf.train.AdamOptimizer(learningRate).minimize(loss)
	tf.summary.scalar("Loss", loss)
	tf.summary.tensor_summary("Entropy", entropy)
	write_op = tf.summary.merge_all()
	
train_writer = tf.summary.FileWriter('/tensorboard/pg/1', tf.get_default_graph())

means = []

for epoch in range(numResults):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		episode = 1
		maxSteps = 0
		lastTenSteps = []
		while(episode < episodesPerResult):
			print("Episode: ",episode)
			state = env.reset()
			env.render()
			done = False
			while(not done):
				accumulatedStates = []
				accumulatedActions = []
				episodeSteps = 0
				while(True):
					episodeSteps = episodeSteps + 1
					actionProbabilities = sess.run(suggestedAction, feed_dict={states:state.reshape([1, 4])})
					actionChoice = np.random.choice(range(actionProbabilities.shape[1]), p=actionProbabilities.ravel())
					accumulatedStates.append(state)
					accumulatedActions.append(getOneHot((2,), actionChoice))
					newState, stepReward, done, _ = env.step(actionChoice)
					if (done):
						break
					env.render()
					state = newState
				rewardValues = getRewards(accumulatedStates)
				episodeReward = np.sum(rewardValues)
				maxSteps = max(episodeSteps, maxSteps)
				lastTenSteps.append(episodeSteps)
				if (len(lastTenSteps) > 10):
					lastTenSteps.pop(0)
				if (episode == learningRateSteps[0]):
					currentLearningRate = .001
				if (episode == learningRateSteps[1]):
					currentLearningRate = .0001
				lossValue, _, _ = sess.run([loss, train_op, write_op], feed_dict={
					states: np.vstack(np.array(accumulatedStates)),
					actions: np.vstack(np.array(accumulatedActions)),
					rewards: rewardValues,
					learningRate: currentLearningRate
				})
				episode = episode + 1
		means.append(np.mean(lastTenSteps))
		print("COUNT: ",len(means))
		print("MEAN: ",np.mean(means))
		print("STD: ",np.std(means))



