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
numEpochs = 20
layers = [16, 16, 16]
learningRate = 0.0001

env = gym.make('CartPole-v0')
def gatherObservations(necessaryObservations):
    all_observations = []
    all_qa = []
    all_sr = []
    numObservations = 0
    while numObservations < necessaryObservations:
        state = env.reset()
        done = False
        while done == False:
            actionChosen = np.random.randint(0, 2)
            newState, reward, done, _ = env.step(actionChosen)
            qa = np.concatenate((state, [actionChosen]))
            all_qa.append(qa)
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

print("Getting train data")
trainObservations = gatherObservations(numTrainObservations)
trainBatches = getBatches(trainObservations, batch_size)

print("Getting test data")
testObservations = gatherObservations(numTrainObservations)
testBatches = getBatches(testObservations, numTestObservations)

print("Data retrieved")
qa_input = tf.placeholder(tf.float32, [None, maxBackpropLength, numStateVariables + 1])
layerOne = tf.nn.rnn_cell.LSTMCell(num_units=layers[0], state_is_tuple=True)
layerTwo = tf.nn.rnn_cell.LSTMCell(num_units=layers[1], state_is_tuple=True)
joined = tf.nn.rnn_cell.MultiRNNCell([layerOne, layerTwo], state_is_tuple=True)
rnn_layer, last_states = tf.nn.dynamic_rnn(cell=joined,dtype=tf.float32,inputs=qa_input)
fc = tf.layers.dense(inputs=rnn_layer, units=layers[2])
predictions = tf.layers.dense(inputs=fc, units=numStateVariables + 1)
sr_input = tf.placeholder(tf.float32, [None, maxBackpropLength, numStateVariables + 1])
losses = tf.losses.mean_squared_error(predictions=predictions, labels=sr_input)
train_op = tf.train.AdamOptimizer(learningRate).minimize(losses)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(numEpochs):
        train_losses = []
        print("Processing epoch: ",epoch)
        for batchId in range(len(trainBatches[QA_INDICE])):
            l, _ = sess.run([losses, train_op],feed_dict={
                qa_input: trainBatches[QA_INDICE][batchId],
                sr_input: trainBatches[SR_INDICE][batchId]
            })
            train_losses.append(l)
        testLoss = sess.run([losses], feed_dict={
            qa_input: testBatches[QA_INDICE][0],
            sr_input: testBatches[SR_INDICE][0]
        })
        trainLossVal = np.average(train_losses[-10:])
        testLossVal = testLoss[0]



