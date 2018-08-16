import tensorflow as tf
import numpy as np
from random import shuffle
import gym

numStateVariables = 4
QA_INDICE = 0
SR_INDICE = 1
LOSS_WEIGHT_INDICE = 2

__CART_POSITION = 0
__CART_VELOCITY = 1
__TIP_ANGLE = 2
__TIP_ANGLE_VELOCITY = 3

__RANDOM = 0
__NOT_RANDOM = 1

__MIN_TEST_SIZE = 128
__MIN_TRAIN_SIZE = 512
__NUM_LAYERS = 4
__LAYER_DEPTH = 128
__LEARNING_RATE = 10e-5
__GAMMA = .95
__SEARCH_DEPTH = 4
__NEGATIVE_REWARD_LOSS = 3
__MAX_EPOCHS = 100
__MIN_EPISODES_100_TRAINING = 20
__MIN_FIT_QUALITY = .8
__MAX_EPISODES = 20000
__EPISODE_RENDERING_MIN = 30

env = gym.make('CartPole-v0')

def follow(sess, state, action, depth):
    modelInput = np.concatenate((state, [action]))
    expectedResult = sess.run([predictions], feed_dict={
        t_qaInput: [modelInput]
    })
    expectedState = expectedResult[0][0][:numStateVariables]
    expectedReward = expectedResult[0][0][numStateVariables]
    adjustment = 0
    if depth > 0:
        adjustment = __GAMMA * np.average([
            follow(sess, expectedState, 0, depth - 1),
            follow(sess, expectedState, 1, depth - 1)
        ])
    expectedReward = expectedReward + adjustment
    return expectedReward

def getModelAction(sess, state):
    resultZero = follow(sess, state, 0, __SEARCH_DEPTH)
    resultOne = follow(sess, state, 1, __SEARCH_DEPTH)
    if resultOne > resultZero:
        return 1
    return 0

t_qaInput = tf.placeholder(tf.float32, [None, numStateVariables + 1])
prevLayer = t_qaInput
denseLayers = []
for i in range(__NUM_LAYERS):
    currentLayer = tf.layers.dense(inputs=prevLayer, units=__LAYER_DEPTH, activation=tf.nn.relu)
    denseLayers.append(currentLayer)
    prevLayer = currentLayer
predictions = tf.layers.dense(inputs=denseLayers[-1], units=numStateVariables + 1)
t_srInput = tf.placeholder(tf.float32, [None, numStateVariables + 1])
losses = tf.losses.mean_squared_error(predictions=predictions, labels=t_srInput)
t_lossWeights = tf.placeholder(tf.float32, [None, numStateVariables + 1])
weightedLoss = tf.multiply(losses, t_lossWeights)
trainingOperation = tf.train.AdamOptimizer(__LEARNING_RATE).minimize(weightedLoss)

testSr = []
testQa = []
testLossWeights = []
trainSr = []
trainQa = []
trainLossWeights = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for episode in range(__MAX_EPISODES):
        print("Episode: ", episode)
        state = env.reset()
        done = False
        stepNum = 0
        qa = []
        sr = []
        lossWeights = []
        while not done:
            if episode > __EPISODE_RENDERING_MIN:
                env.render()
            if len(trainQa) < __MIN_TRAIN_SIZE:
                actionChosen = np.random.randint(0, 2)
            else:
                actionChosen = getModelAction(sess, state)
            newState, reward, done, _ = env.step(actionChosen)
            currentLossWeights = [1, 1, 1, 1, 1]
            if done:
                reward = -1
                currentLossWeights[numStateVariables] = __NEGATIVE_REWARD_LOSS
            qa.append(np.concatenate((state, [actionChosen])))
            sr.append(np.concatenate((newState, [reward])))
            lossWeights.append(currentLossWeights)
            stepNum = stepNum + 1
            state = newState
        print("Episode Length: ",stepNum)
        if len(trainSr) - len(sr) > __MIN_TRAIN_SIZE:
            trainSr = trainSr[len(sr):]
            trainQa = trainQa[len(qa):]
            trainLossWeights = trainLossWeights[len(lossWeights):]
        if len(testSr) == 0:
            testSr = sr
            testQa = qa
            testLossWeights = lossWeights
        elif len(testSr) < __MIN_TEST_SIZE:
            testSr = np.vstack((testSr, sr))
            testQa = np.vstack((testQa, qa))
            testLossWeights = np.vstack((testLossWeights, lossWeights))
        elif len(trainSr) == 0:
            trainSr = sr
            trainQa = qa
            trainLossWeights = lossWeights
        else:
            trainSr = np.vstack((trainSr, sr))
            trainQa = np.vstack((trainQa, qa))
            trainLossWeights = np.vstack((trainLossWeights, lossWeights))
        print("Training size: ",len(trainSr))
        print("Test size: ",len(testSr))
        if len(trainSr) >= __MIN_TRAIN_SIZE:
            for epoch in range(__MAX_EPOCHS):
                trainLoss, _ = sess.run([losses, trainingOperation],feed_dict={
                    t_qaInput: trainQa,
                    t_srInput: trainSr,
                    t_lossWeights: trainLossWeights
                })
                testLoss = sess.run([losses], feed_dict={
                    t_qaInput: testQa,
                    t_srInput: testSr,
                    t_lossWeights: testLossWeights
                })
                fitQuality = trainLoss / testLoss
                print("Epoch: ",epoch," || Fit quality: ", fitQuality[0], " loss: ",testLoss[0])
                if (fitQuality <  __MIN_FIT_QUALITY) and episode > __MIN_EPISODES_100_TRAINING:
                    break




