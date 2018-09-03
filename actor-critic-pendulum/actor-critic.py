import tensorflow as tf
import numpy as np
import random
from collections import deque
import gym

numStateVariables = 4
numActions = 1
numRewards = 1

__ENVIRONMENT = 0
__ACTION = 1
__NEW_ENVIRONMENT = 2
__REWARD = 3
__NEXT_ACTION = 4

__RENDERING = False
__CRITIC_LEARNING_RATE = 1e-3
__ACTOR_LEARNING_RATE = 1e-4
__MAX_MEMORY_LENGTH = 250
__BATCH_SIZE=64
__GAMMA = .95
__EPSILON_DECAY = .995
__EPSILON_INITIAL = .1
__ACTOR_ARCHITECTURE = [64, 64]
__CRITIC_ARCHITECTURE = [64, 64]
__CRITIC_TAU = .001
__ACTOR_TAU = .001

def getColumn(arr, column):
    return [row[column] for row in arr]
class ActorCritic:
    def __init__(self, sess, env):
        self.sess = sess
        self.env = env
        self.state = env.reset()
        self.previousState = None
        self.episodeMemory = deque(None, __MAX_MEMORY_LENGTH)
        self.epsilon = __EPSILON_INITIAL
        self.actor = Actor(sess)
        self.critic = Critic(sess)
    def train(self):
        trainingEpisodes = random.sample(self.episodeMemory, min(length(self.episodeMemory), __BATCH_SIZE))
        trainingEpisodes = self.actor.addPredictedActions(trainingEpisodes)
        self.critic.train(trainingEpisodes)
        self.actor.train(self.critic.getActorGradients(trainingEpisodes), trainingEpisodes)
        self.critic.updateTarget()
        self.actor.updateTarget()
    def act(self):
        if random.uniform(0, 1) > self.episilon:
            actionChosen = self.actor.act(self.state)
        else:
            actionChosen = random.uniform(-1, 1)
        self.previousState = self.state
        self.state, reward, done, _ = env.step(actionChosen * 2)
        self.episilon = self.episilon * __EPSILON_DECAY

class Actor:
    def __init__(self, sess):
        self.sess = sess
        self.evalNetwork = ActorNetwork(sess, "evalActor")
        self.targetNetwork = ActorNetwork(sess, "targetActor")
    def train(self, criticActionGradients, environmentInputs):
        self.sess.run(self.evalNetwork.optimize, feed_dict={
            actionGradient: criticActionGradients,
            environmentInput: environmentInputs
        })
    def updateEval(tau=__CRITIC_TAU):
        for i in range(len(self.targetNetwork.networkParams)):
            self.evalNetwork.networkParams[i].assign(tf.mul(self.evalNetwork.networkParams[i], 1. - tau) + tf.mul(self.targetNetwork.networkParams[i], tau))
    def updateTarget(tau=__CRITIC_TAU):
        for i in range(len(self.targetNetwork.networkParams)):
            self.targetNetwork.networkParams[i].assign(tf.mul(self.targetNetwork.networkParams[i], 1. - tau) + tf.mul(self.evalNetwork.networkParams[i], tau))
    def act(self, environment):
        return self.sess.run(self.evalNetwork.actionOutput, feed_dict={
            self.environmentInput: environment
        })[0]
    def addPredictedActions(self, memoryUnits):
        predictedActions = self.sess.run(self.targetNetwork.actionOutput, feed_dict={
            self.targetNetwork.environmentInput: getColumn(memoryUnits, __NEW_ENVIRONMENT)
        })[0]
        return np.hstack([memoryUnits, predictedActions])

class ActorNetwork:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.buildNetwork()
    def buildNetwork(self):
        with tf.variable_scope(self.name):
            self.environmentInput = tf.placeholder(tf.float32, [None, numStateVariables])
            prevLayer = self.environmentInput
            for i in range(__ACTOR_ARCHITECTURE):
                currentLayer = tf.layers.dense(inputs=prevLayer, units=__ACTOR_ARCHITECTURE[i], activation=tf.nn.leaky_relu)
                prevLayer = currentLayer
            self.actionOutput = tf.layers.dense(inputs=prevLayer, units=numActions, activation=tf.nn.tanh)
            self.actionGradient = tf.placeholder(tf.float32, [None,])
            self.networkParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            self.appliedActionGradient = tf.gradients(self.actionOutput, self.networkParams, -self.actionGradient) / __BATCH_SIZE # Not sure if necessary yet
            self.optimize = tf.train.AdamOptimizer(self.__ACTOR_LEARNING_RATE).apply_gradients(self.actionGradient, self.networkParams)


class Critic:
    def __init__(self, sess):
        self.sess = sess
        self.evalNetwork = CriticNetwork(sess, "evalCritic")
        self.targetNetwork = CriticNetwork(sess, "targetCritic")
    def getActorGradients(self, memoryUnits):
        return self.sess.run(self.actorGradients, feed_dict={
            self.environmentInput: getColumn(memoryUnit, __ENVIRONMENT),
            self.actionInput: getColumn(memoryUnit, __ACTION)    
        })
    def setActorOutput(self, actorOutput):
        self.actorOutput = actorOutput
    def updateEval(tau=__CRITIC_TAU):
        for i in range(len(self.targetNetwork.networkParams)):
            self.evalNetwork.networkParams[i].assign(tf.mul(self.evalNetwork.networkParams[i], 1. - tau) + tf.mul(self.targetNetwork.networkParams[i], tau))
    def updateTarget(tau=__CRITIC_TAU):
        for i in range(len(self.targetNetwork.networkParams)):
            self.targetNetwork.networkParams[i].assign(tf.mul(self.targetNetwork.networkParams[i], 1. - tau) + tf.mul(self.evalNetwork.networkParams[i], tau))
    def train(self, memoryUnits):
        predictedNextStateValues = self.sess.run(self.predictedStateValue, feed_dict={
            self.environmentInput: getColumn(memoryUnit, __NEW_ENVIRONMENT),
            self.actionInput: getColumn(memoryUnit, __NEXT_ACTION),
        })[0]
        self.sess.run(self.trainingOperation, feed_dict={
            self.environmentInput: getColumn(memoryUnits, __ENVIRONMENT),
            self.actionInput: getColumn(memoryUnits, __ACTION),
            self.actualStateValue: np.sum([getColumn(memoryUnit, __REWARD), predictedNextStateValues], axis=0)
        })

class CriticNetwork:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.buildNetwork()
        self.networkParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    def buildNetwork(self):
        with tf.variable_scope(self.name):
            self.environmentInput = tf.placeholder(tf.float32, [None, numStateVariables])
            self.actionInput = tf.placeholder(tf.float32, [None, numActions])
            self.hiddenLayers = []
            prevLayer = [self.environmentInput, self.actionInput]
            for i in range(__CRITIC_ARCHITECTURE):
                currentLayer = tf.layers.dense(inputs=prevLayer, units=__CRITIC_ARCHITECTURE[i], activation=tf.nn.leaky_relu)
                self.hiddenLayers.append(currentLayer)
                prevLayer = currentLayer
            self.predictedStateValue = tf.layers.dense(inputs=prevLayer, units=1)
            self.actualStateValue = tf.placeholder(tf.float32, [None,])
            self.loss = tf.losses.mean_squared_error(labels=self.actualStateValue, predictions=self.predictedStateValue)
            self.trainingOperation = tf.train.AdamOptimizer(__CRITIC_LEARNING_RATE).minimize(self.loss)
            self.actorGradients = tf.gradients(self.predictedStateValue, self.actionInput)[0] 


with tf.Session() as sess:
    env = gym.make('pendulum-v0')
    agent = ActorCritic(sess, env)
    while(True):
        agent.act()
        agent.train()
        if (__RENDERING):
            env.render()