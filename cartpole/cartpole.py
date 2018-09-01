import tensorflow as tf
import numpy as np
import gym

env = gym.make('CartPole-v0')
env = env.unwrapped
# Policy gradient has high variance, seed for reproducability
env.seed(1)
tf.set_random_seed(1)

## ENVIRONMENT Hyperparameters
state_size = 4
action_size = env.action_space.n

## TRAINING Hyperparameters
max_episodes = 10000
learning_rate = 0.01
gamma = 0.90 # Discount rate

def normalizeRewards(rewards):
	normalizedRewards = np.zeros_like(rewards)
	total = 0.0
	for i in reversed(range(len(rewards))):
		total = total * gamma + rewards[i]
		normalizedRewards[i] = total
	mean = np.mean(normalizedRewards)
	stddev = np.std(normalizedRewards)
	normalizedRewards = (normalizedRewards - mean) / stddev
	return normalizedRewards

with tf.name_scope("inputs"):
	inputData = tf.placeholder(tf.float32, [None, state_size])
	actions = tf.placeholder(tf.int32, [None, action_size])
	rewards = tf.placeholder(tf.float32, [None, ], name="rewards")
	meanReward = tf.placeholder(tf.float32, name="meanReward")
with tf.name_scope("hidden_layers"):
	dense1 = tf.layers.dense(inputs=inputData, units=10, activation=tf.nn.relu)
	dense2 = tf.layers.dense(inputs=dense1, units=action_size, activation=tf.nn.relu)
with tf.name_scope("output"):
	outputRaw = tf.layers.dense(inputs=dense2, units=action_size)
	outputSoftmax = tf.nn.softmax(outputRaw)
with tf.name_scope("loss"):
	entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = outputRaw, labels = actions)
	loss = tf.reduce_mean(entropy * rewards)
with tf.name_scope("train"):
	train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	writer = tf.summary.FileWriter("/tensorboard/pg/1")
	tf.summary.scalar("Loss", loss)
	tf.summary.scalar("Reward_mean", meanReward)
	write_op = tf.summary.merge_all()

rewardHistory = []
rewardSum = 0
rewardMax = 0
episode = 0
episode_states, episode_actions, episode_rewards = [],[],[]
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for episode in range(max_episodes):
		sumRewardEpisode = 0
		state = env.reset()
		while True:
			env.render()
			actionProbabilities = sess.run(outputSoftmax, feed_dict={inputData:state.reshape([1,4])})
			print(actionProbabilities)
			action = np.random.choice(range(actionProbabilities.shape[1]), p=actionProbabilities.ravel())
			newState, reward, done, info = env.step(action)
			episode_states.append(state)
			action_ = np.zeros(action_size)
			action_[action] = 1
			episode_actions.append(action_)
			episode_rewards.append(reward)
			if done:
				episode_rewards_sum = np.sum(episode_rewards)
				rewardHistory.append(episode_rewards_sum)
				rewardSum = rewardSum + episode_rewards_sum
				meanRewardData = np.divide(rewardSum, episode+1)
				maximumRewardRecorded = np.amax(rewardHistory)
				print("==========================================")
				print("Episode: ", episode)
				print("Reward: ", episode_rewards_sum)
				print("Mean Reward", meanRewardData)
				print("Max reward so far: ", maximumRewardRecorded)
				normalizedRewards = normalizeRewards(episode_rewards)
				loss_, _ = sess.run([loss, train_op],
					feed_dict={
						inputData: np.vstack(np.array(episode_states)),
						actions: np.vstack(np.array(episode_actions)),
						rewards: normalizedRewards 
					}
				)
				print("loss: ",loss)
				summary = sess.run(write_op, feed_dict={
					inputData: np.vstack(np.array(episode_states)),
					actions: np.vstack(np.array(episode_actions)),
					rewards: normalizedRewards,
					meanReward: meanRewardData
				})
				# writer.add_summary(summary, episode)
				# writer.flush()
				episode_states, episode_actions, episode_rewards, episode_prob = [],[],[],[]
				break
			state = newState