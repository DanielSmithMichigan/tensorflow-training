import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import gym

numStateVariables = 4
QA_INDICE = 0
SR_INDICE = 1
LOSS_WEIGHT_INDICE = 2

batchSize = 128
numTrainObservations = 1024
numTestObservations = 256
numEpochs = 850
layerWidth = 4
layerDepth = 64
learningRate = 10e-5
gamma = .95
searchDepth = 6
negativeRewardLoss = 200

__CART_POSITION = 0
__CART_VELOCITY = 1
__TIP_ANGLE = 2
__TIP_ANGLE_VELOCITY = 3

env = gym.make('CartPole-v0')
def gatherObservations(necessaryObservations):
    all_observations = []
    all_qa = []
    all_sr = []
    all_loss_weights = []
    numObservations = 0
    while numObservations < necessaryObservations:
        state = env.reset()
        done = False
        current_sr = []
        numSteps = 0
        while done == False:
            actionChosen = np.random.randint(0, 2)
            newState, reward, done, _ = env.step(actionChosen)
            qa = np.concatenate((state, [actionChosen]))
            all_qa.append(qa)
            loss_weights = np.ones((numStateVariables + 1,))
            if (done):
                reward = -1
                loss_weights[numStateVariables] = negativeRewardLoss
            all_loss_weights.append(loss_weights)
            sr = np.concatenate((newState, [reward]))
            all_sr.append(sr)
            state = newState
            numObservations = numObservations + 1
            numSteps = numSteps + 1
            if numObservations >= necessaryObservations:
                break
    all_observations.append(all_qa)
    all_observations.append(all_sr)
    all_observations.append(all_loss_weights)
    return all_observations

def follow(sess, state, action, depth):
    modelInput = np.concatenate((state, [action]))
    expectedResult = sess.run([predictions], feed_dict={
        qaInput: [modelInput]
    })
    expectedState = expectedResult[0][0][:numStateVariables]
    expectedReward = expectedResult[0][0][numStateVariables]
    adjustment = 0
    if depth > 0:
        adjustment = gamma * np.average([
            follow(sess, expectedState, 0, depth - 1),
            follow(sess, expectedState, 1, depth - 1)
        ])
    expectedReward = expectedReward + adjustment
    return expectedReward

def printState(state):
    print("Cart position: ",state[0])
    print("Cart velocity: ",state[1])
    print("Tip angle: ",state[2])
    print("Tip angle velocity: ",state[3])

print("Getting train data")
trainObservations = gatherObservations(numTrainObservations)

def observeAction(action):
    cartPositions = []
    cartVelocities = []
    tipAngles = []
    tipAngleVelocities = []
    rewards = []
    state = env.reset()
    for i in range(40):
        modelInput = np.concatenate((state, [action]))
        expectedResult = sess.run([predictions], feed_dict={
            qaInput: [modelInput]
        })
        state = expectedResult[0][0][:numStateVariables]
        cartPositions.append(state[__CART_POSITION])
        cartVelocities.append(state[__CART_VELOCITY])
        tipAngles.append(state[__TIP_ANGLE])
        tipAngleVelocities.append(state[__TIP_ANGLE_VELOCITY])
    print("POSITION")
    for i in range(len(cartPositions)):
        print(cartPositions[i])
    print("VELOCITIES")
    for i in range(len(cartVelocities)):
        print(cartVelocities[i])
    print("ANGLES")
    for i in range(len(tipAngles)):
        print(tipAngles[i])
    print("ANGLE VELOCITY")
    for i in range(len(tipAngleVelocities)):
        print(tipAngleVelocities[i])

print("Getting test data")
testObservations = gatherObservations(numTrainObservations)

observationStd = np.std(testObservations[SR_INDICE], axis=0)
observedMean = np.average(testObservations[SR_INDICE], axis=0)
observedMax = np.array(testObservations[SR_INDICE]).max(axis=0)
observedMin = np.array(testObservations[SR_INDICE]).min(axis=0)

qaInput = tf.placeholder(tf.float32, [None, numStateVariables + 1])
layerOne = tf.layers.dense(inputs=qaInput, units=layerDepth, activation=tf.nn.relu)
layerTwo = tf.layers.dense(inputs=layerOne, units=layerDepth, activation=tf.nn.relu)
layerThree = tf.layers.dense(inputs=layerTwo, units=layerDepth, activation=tf.nn.relu)
predictions = tf.layers.dense(inputs=layerThree, units=numStateVariables + 1)
srInput = tf.placeholder(tf.float32, [None, numStateVariables + 1])
lossWeights = tf.placeholder(tf.float32, [None, numStateVariables + 1])
losses = tf.losses.mean_squared_error(predictions=predictions, labels=srInput)
weightedLoss = tf.multiply(losses, lossWeights)
trainingOperation = tf.train.AdamOptimizer(learningRate).minimize(weightedLoss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(numEpochs):
        train_losses = []
        print("Processing epoch: ",epoch)
        for observationId in range(0, len(trainObservations[QA_INDICE]), batchSize):
            l, _ = sess.run([losses, trainingOperation],feed_dict={
                qaInput: trainObservations[QA_INDICE][observationId:observationId+batchSize],
                srInput: trainObservations[SR_INDICE][observationId:observationId+batchSize],
                lossWeights: trainObservations[LOSS_WEIGHT_INDICE][observationId:observationId+batchSize]
            })
            train_losses.append(l)
        testLoss = sess.run([losses], feed_dict={
            qaInput: testObservations[QA_INDICE][:numTestObservations],
            srInput: testObservations[SR_INDICE][:numTestObservations],
            lossWeights: testObservations[LOSS_WEIGHT_INDICE][observationId:observationId+batchSize]
        })
        trainLossVal = np.average(train_losses[-10:])
        testLossVal = testLoss[0]
        print({
            "trainingLoss": trainLossVal,
            "testLoss": testLossVal,
            "fitQuality": trainLossVal / testLossVal
        })
    print("Beginning simulation")
    print("Stddev", observationStd)
    print("Mean", observedMean)
    print("Max", observedMax)
    print("Min", observedMin)
    stepNum = 0
    done = False
    state = env.reset()
    diffs = []
    while not done:
        print("STEP: ",stepNum)
        resultZero = follow(sess, state, 0, searchDepth)
        resultOne = follow(sess, state, 1, searchDepth)
        actionChosen = 0
        if resultOne > resultZero:
            actionChosen = 1
        expectedSr = sess.run([predictions], feed_dict={
            qaInput: [np.concatenate((state, [actionChosen]))]
        })[0][0]
        expectedState = expectedSr[:numStateVariables]
        newState, reward, done, _ = env.step(actionChosen)
        print("Action: ",actionChosen)
        print(newState)
        stepNum = stepNum + 1
        state = newState