import tensorflow as tf
import numpy as np
from random import shuffle
import gym

numStateVariables = 4
numActions = 1
numRewards = 1

__LEARNING_RATE = 1e-5
__GAMMA = .95
__EPSILON_DECAY = .995
__EPSILON_INITIAL = .1
__ACTOR_ARCHITECTURE = [64, 64]
__CRITIC_ARCHITECTURE = [64, 64]

class ActorCritic:
    def __init__(self, sess):
        self.episodeMemory = []
        self.targetActor = Actor()
        self.actor = Actor()
        self.targetCritic = Critic()
        self.critic = Critic()

    def trainTargetActor(self):
        actionGradient = self.actor.getActionGradient()
        self.targetActor.optimizeAgainst(actionGradient)


tf.nn.leaky_relu(features,alpha=0.2,name=None)

class Actor:
    def __init__(self):
        self.environmentInput = tf.placeholder(tf.float32, [None, numStateVariables])
        prevLayer = self.environmentInput
        for i in range(__ACTOR_ARCHITECTURE):
            currentLayer = tf.layers.dense(inputs=prevLayer, units=__ACTOR_ARCHITECTURE[i], activation=tf.nn.leaky_relu)
            prevLayer = currentLayer
        outputLogit = tf.layers.dense(inputs=prevLayer, units=numActions, activation=tf.nn.tanh)
        self.outputLogit = outputLogit * 2
        self.actionGradient = tf.placeholder(tf.float32, [None, numActions])
        
    def optimizeAgainst(self, actionGradient):
        tf.train.AdamOptimizer(self.learning_rate).apply_gradients(actionGradient, )


class Critic:
    def __init__(self):
        self.environmentInput = tf.placeholder(tf.float32, [None, numStateVariables])
        self.actionInput = tf.placeholder(tf.float32, [None, numActions])
        prevLayer = [self.environmentInput, self.actionInput]
        for i in range(__CRITIC_ARCHITECTURE):
            currentLayer = tf.layers.dense(inputs=prevLayer, units=__CRITIC_ARCHITECTURE[i], activation=tf.nn.leaky_relu)
            prevLayer = currentLayer
        self.stateValue = tf.layers.dense(inputs=prevLayer, units=1, activation=tf.nn.sigmoid)
    def getActionGradient(self):
        return tf.gradients(self.stateValue,self.actionInput)



env = gym.make('pendulum-v0')

trainingOperation = tf.train.AdamOptimizer(__LEARNING_RATE).minimize(loss)

env.reset()