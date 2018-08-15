import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import gym

numStateVariables = 4
QA_INDICE = 0
SR_INDICE = 1

batch_size = 16
maxBackpropLength = 16
numTrainObservations = 20000
numTestObservations = 2000
numEpochs = 15
layers = [16, 16, 16]
learningRate = 0.0001
gamma = .90
searchDepth = 6
noisePct = 0
numNoisyStates = 1
observedStd = [0.09774672,0.57341062,0.10342966,0.86594986]
observedMean = [8.15114988e-04,2.61279860e-02,2.21402959e-03,-1.85102279e-02]

env = gym.make('CartPole-v0')
def gatherObservations(necessaryObservations):
    all_observations = []
    all_qa = []
    all_sr = []
    numObservations = 0
    while numObservations < necessaryObservations:
        state = env.reset()
        done = False
        current_sr = []
        while done == False:
            actionChosen = np.random.randint(0, 2)
            newState, reward, done, _ = env.step(actionChosen)
            qa = np.concatenate((state, [actionChosen]))
            all_qa.append(qa)
            current_sr.append(newState) 
            sr = np.concatenate((newState, [reward]))
            all_sr.append(sr)
            state = newState
            numObservations = numObservations + 1
            if numObservations >= necessaryObservations:
                break
    all_observations.append(all_qa)
    all_observations.append(all_sr)
    return all_observations

def getBatches(observations, _batch_size):
    batches = []
    qa_backprop_slices = []
    sr_backprop_slices = []
    for currentSequencePosition in range(len(observations[QA_INDICE])):
        if currentSequencePosition + maxBackpropLength > len(observations[QA_INDICE]):
            break
        qa_backprop_slices.append(observations[QA_INDICE][currentSequencePosition:currentSequencePosition + maxBackpropLength])
        sr_backprop_slices.append(observations[SR_INDICE][currentSequencePosition:currentSequencePosition + maxBackpropLength])
    qa_batches = []
    sr_batches = []
    for currentSequencePosition in range(0, len(qa_backprop_slices), _batch_size):
        if currentSequencePosition + _batch_size > len(qa_backprop_slices):
            break
        qa_batches.append(qa_backprop_slices[currentSequencePosition:currentSequencePosition + _batch_size])
        sr_batches.append(sr_backprop_slices[currentSequencePosition:currentSequencePosition + _batch_size])
    batches.append(qa_batches)
    batches.append(sr_batches)
    return batches

def getNoisyState(state):
    noise = np.random.normal(loc=0.0, scale=noisePct * observationStd[:numStateVariables], size=(numStateVariables,))
    return state + noise

def followWithNoise(sess, state, action, depth):
    noisyStates = [getNoisyState(state) for i in range(numNoisyStates)]
    return np.average([follow(sess, noisyState, action, depth) for noisyState in noisyStates])

def rewardFromState(state):
    state = state.copy()
    for i in range(len(state)):
        state[i] = abs(state[i] - observedMean[i]) / observationStd[i]
    return -np.sum(state[:numStateVariables])

def follow_new(sess, state, action, depth):
    modelInput = np.concatenate((state, [action]))
    expectedResult = sess.run([predictions], feed_dict={
        qa_input: [modelInput]
    })
    expectedState = expectedResult[0][0][:numStateVariables]
    expectedReward = rewardFromState(expectedState)
    if depth > 0:
        expectedReward = expectedReward + gamma * follow(sess, expectedState, action, depth - 1)
    return expectedReward

def follow(sess, state, action, depth):
    modelInput = np.concatenate((state, [action]))
    expectedResult = sess.run([predictions], feed_dict={
        qa_input: [modelInput]
    })
    expectedState = expectedResult[0][0][:numStateVariables]
    expectedReward = rewardFromState(expectedState)
    if depth > 0:
        expectedReward = expectedReward + gamma * np.average([
            follow(sess, expectedState, 0, depth - 1),
            follow(sess, expectedState, 1, depth - 1)
        ])
    return expectedReward


print("Getting train data")
trainObservations = gatherObservations(numTrainObservations)
trainBatches = getBatches(trainObservations, batch_size)

print("Getting test data")
testObservations = gatherObservations(numTrainObservations)
testBatches = getBatches(testObservations, numTestObservations)

observationStd = np.std(testObservations[SR_INDICE], axis=0)
observedMean = np.average(testObservations[SR_INDICE], axis=0)
observedMax = np.array(testObservations[SR_INDICE]).max(axis=0)
observedMin = np.array(testObservations[SR_INDICE]).min(axis=0)

print("Data retrieved")
qa_input = tf.placeholder(tf.float32, [None, numStateVariables + 1])
layerOne = tf.layers.dense(inputs=qa_input, units=layers[0])
layerTwo = tf.layers.dense(inputs=layerOne, units=layers[1])
rnn_layer = tf.layers.dense(inputs=layerTwo, units=layers[2])
fc = tf.layers.dense(inputs=rnn_layer, units=layers[2])
predictions = tf.layers.dense(inputs=fc, units=numStateVariables + 1)
sr_input = tf.placeholder(tf.float32, [None, numStateVariables + 1])
losses = tf.losses.mean_squared_error(predictions=predictions, labels=sr_input)
train_op = tf.train.AdamOptimizer(learningRate).minimize(losses)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(numEpochs):
        train_losses = []
        print("Processing epoch: ",epoch)
        for observationId in range(0, len(trainObservations[QA_INDICE]), batch_size):
            l, _ = sess.run([losses, train_op],feed_dict={
                qa_input: trainObservations[QA_INDICE][observationId:observationId+batch_size],
                sr_input: trainObservations[SR_INDICE][observationId:observationId+batch_size]
            })
            train_losses.append(l)
        testLoss = sess.run([losses], feed_dict={
            qa_input: testObservations[QA_INDICE][:numTestObservations],
            sr_input: testObservations[SR_INDICE][:numTestObservations]
        })
        trainLossVal = np.average(train_losses[-10:])
        testLossVal = testLoss[0]
        print({
            "trainingLoss": trainLossVal,
            "testLoss": testLossVal,
            "overfitting": testLossVal / trainLossVal
        })
    print("Beginning simulation")
    done = False
    state = env.reset()
    stepNum = 0
    while not done:
        resultZero = follow(sess, state, 0, searchDepth)
        resultOne = follow(sess, state, 1, searchDepth)
        actionChosen = 0
        if resultOne > resultZero:
            actionChosen = 1
        modelInput = np.concatenate((state, [actionChosen]))
        expectedResult = sess.run([predictions], feed_dict={
            qa_input: [
                np.concatenate((state, [0])),
                np.concatenate((state, [1]))
            ]
        })
        newState, reward, done, _ = env.step(actionChosen)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Step: ",stepNum)
        print("Result 0: ",resultZero)
        print("Result 1: ",resultOne)
        print("ActionChosen: ",actionChosen)
        print("positionIn", state[0])
        print("velocityIn", state[1])
        print("angleIn", state[2])
        print("tipVelocityIn", state[3])
        print("positionPredictionLeft", expectedResult[0][0][0])
        print("velocityPredictionLeft", expectedResult[0][0][1])
        print("anglePredictionLeft", expectedResult[0][0][2])
        print("tipVelocityPredictionLeft", expectedResult[0][0][3])
        print("positionPredictionRight", expectedResult[0][1][0])
        print("velocityPredictionRight", expectedResult[0][1][1])
        print("anglePredictionRight", expectedResult[0][1][2])
        print("tipVelocityPredictionRight", expectedResult[0][1][3])
        print("positionActual", newState[0])
        print("velocityActual", newState[1])
        print("angleActual", newState[2])
        print("tipVelocityActual", newState[3])
        print("Done: ",done)
        stepNum = stepNum + 1
        state = newState
print(observedMax)
print(observedMin)
print("Simulation complete")




