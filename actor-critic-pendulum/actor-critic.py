import tensorflow as tf
import numpy as np
import random
from collections import deque
import gym

numStateVariables = 3
numActions = 1
numRewards = 1

ENVIRONMENT = 0
ACTION = 1
NEW_ENVIRONMENT = 2
REWARD = 3
NEXT_ACTION = 4

RENDERING = True
CRITIC_LEARNING_RATE = 10e-3
ACTOR_LEARNING_RATE = 10e-4
MAX_MEMORY_LENGTH = 10e6
BATCH_SIZE = 64
GAMMA = .99
EPSILON_DECAY = .997
EPSILON_INITIAL = 0
EPSILON_RAND_STEPS = 1
ACTOR_ARCHITECTURE = [400,300]
CRITIC_ARCHITECTURE = [400,300]
CRITIC_TAU = .001
ACTOR_TAU = .001
STEPS_TO_RESET = 512
BETA = 0

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def getColumn(arr, column):
    out = []
    for i in range(len(arr)):
        if (hasattr(arr[i][column], "__len__")):
            out.append(arr[i][column])
        else:
            out.append([arr[i][column]])
    return out

class ActorCritic:
    def __init__(self, sess, env):
        self.sess = sess
        self.env = env
        self.state = env.reset()
        self.previousState = None
        self.episodeMemory = deque([], MAX_MEMORY_LENGTH)
        self.randomSteps = 0
        self.epsilon = EPSILON_INITIAL
        self.actor = Actor(sess)
        self.critic = Critic(sess)
    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.actor.copyEvalToTargetOp)
        self.sess.run(self.critic.copyEvalToTargetOp)
    def train(self):
        trainingEpisodes = random.sample(self.episodeMemory, min(len(self.episodeMemory), BATCH_SIZE))
        trainingEpisodes = self.actor.addPredictedActions(trainingEpisodes)
        self.critic.train(trainingEpisodes)
        self.actor.train(self.critic.getActorGradients(trainingEpisodes), trainingEpisodes)
        self.sess.run(self.critic.updateTargetOp)
        self.sess.run(self.actor.updateTargetOp)
    def testEpsilon(self):
        if random.uniform(0, 1) < self.epsilon:
            self.randomSteps = EPSILON_RAND_STEPS
    def act(self):
        if self.randomSteps == 0:
            noise, actionChosen = self.actor.act(self.state)
        else:
            actionChosen = [random.uniform(-1, 1)]
            self.randomSteps = self.randomSteps - 1
        print("s: {:6.6f}, a: {:6.6f}, n: {:6.6f}".format(self.critic.predictStateValue(self.state, actionChosen)[0], actionChosen[0], noise[0]))
        self.previousState = self.state
        self.state, reward, done, _ = env.step(actionChosen * 2)
        self.episodeMemory.append([
            self.previousState,
            actionChosen,
            self.state,
            reward
        ])
        self.epsilon = self.epsilon * EPSILON_DECAY

class Actor:
    def __init__(self, sess):
        self.sess = sess
        self.evalNetwork = ActorNetwork(sess, "evalActor")
        self.targetNetwork = ActorNetwork(sess, "targetActor")
        self.createUpdateFunctions()
        self.actionNoise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(numActions), theta=0.15, sigma=.2)

    def createUpdateFunctions(self):
        self.updateTargetOp = [tf.assign(t, (1 - ACTOR_TAU) * t + ACTOR_TAU * e) for t, e in zip(self.targetNetwork.networkParams, self.evalNetwork.networkParams)]
        self.copyEvalToTargetOp = [tf.assign(t, e) for t, e in zip(self.evalNetwork.networkParams, self.targetNetwork.networkParams)]
    def train(self, criticActionGradients, memoryUnits):
        self.sess.run(self.evalNetwork.optimize, feed_dict={
            self.evalNetwork.actionGradient: criticActionGradients,
            self.evalNetwork.environmentInput: getColumn(memoryUnits, ENVIRONMENT)
        })
    def act(self, environment):
        actionNoise = self.actionNoise()
        action = self.sess.run(self.evalNetwork.actionOutput, feed_dict={
            self.evalNetwork.environmentInput: [environment]
        })[0]
        return actionNoise, action + actionNoise
    def addPredictedActions(self, memoryUnits):
        predictedActions = self.sess.run(self.targetNetwork.actionOutput, feed_dict={
            self.targetNetwork.environmentInput: getColumn(memoryUnits, NEW_ENVIRONMENT)
        })
        return np.hstack((memoryUnits, predictedActions))

class ActorNetwork:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.buildNetwork()
    def buildNetwork(self):
        with tf.variable_scope(self.name):
            self.environmentInput = tf.placeholder(tf.float32, [None, numStateVariables])
            prevLayer = self.environmentInput
            for i in range(len(ACTOR_ARCHITECTURE)):
                currentLayer = tf.layers.dense(inputs=prevLayer, units=ACTOR_ARCHITECTURE[i], activation=tf.nn.leaky_relu)
                prevLayer = currentLayer
            self.actionOutput = tf.layers.dense(inputs=prevLayer, units=numActions, activation=tf.nn.tanh)
            self.actionGradient = tf.placeholder(tf.float32, [None, 1])
            self.networkParams = tf.trainable_variables(scope=self.name)
            self.appliedActionGradient = tf.gradients(tf.div(self.actionOutput, BATCH_SIZE), self.networkParams, -self.actionGradient)
            self.optimize = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE).apply_gradients(zip(self.appliedActionGradient, self.networkParams))


class Critic:
    def __init__(self, sess):
        self.sess = sess
        self.evalNetwork = CriticNetwork(sess, "evalCritic")
        self.targetNetwork = CriticNetwork(sess, "targetCritic")
        self.createUpdateFunctions()
    def createUpdateFunctions(self):
        self.updateTargetOp = [tf.assign(t, (1 - CRITIC_TAU) * t + CRITIC_TAU * e) for t, e in zip(self.targetNetwork.networkParams, self.evalNetwork.networkParams)]
        self.copyEvalToTargetOp = [tf.assign(t, e) for t, e in zip(self.evalNetwork.networkParams, self.targetNetwork.networkParams)]
    def getActorGradients(self, memoryUnits):
        return self.sess.run(self.evalNetwork.actorGradients, feed_dict={
            self.evalNetwork.environmentInput: getColumn(memoryUnits, ENVIRONMENT),
            self.evalNetwork.actionInput: getColumn(memoryUnits, ACTION)    
        })
    def setActorOutput(self, actorOutput):
        self.actorOutput = actorOutput
    def train(self, memoryUnits):
        predictedNextStateValues = self.sess.run(self.targetNetwork.predictedStateValue, feed_dict={
            self.targetNetwork.environmentInput: getColumn(memoryUnits, NEW_ENVIRONMENT),
            self.targetNetwork.actionInput: getColumn(memoryUnits, NEXT_ACTION),
        })
        self.sess.run(self.evalNetwork.trainingOperation, feed_dict={
            self.evalNetwork.environmentInput: getColumn(memoryUnits, ENVIRONMENT),
            self.evalNetwork.actionInput: getColumn(memoryUnits, ACTION),
            self.evalNetwork.actualStateValue: np.sum([getColumn(memoryUnits, REWARD), GAMMA * predictedNextStateValues], axis=0)
        })
    def predictStateValue(self, state, action):
        return self.sess.run(self.evalNetwork.predictedStateValue, feed_dict={
            self.evalNetwork.environmentInput: [state],
            self.evalNetwork.actionInput: [action]
        })[0]

class CriticNetwork:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.buildNetwork()
        self.networkParams = tf.trainable_variables(scope=self.name)
    def buildNetwork(self):
        with tf.variable_scope(self.name):
            self.environmentInput = tf.placeholder(tf.float32, [None, numStateVariables])
            self.firstHiddenLayer = tf.layers.dense(inputs=self.environmentInput, units=CRITIC_ARCHITECTURE[0], activation=tf.nn.leaky_relu, kernel_regularizer=)
            self.actionInput = tf.placeholder(tf.float32, [None, numActions])
            self.concatInput = tf.concat([self.firstHiddenLayer, self.actionInput], 1)
            self.secondHiddenLayer = tf.layers.dense(inputs=self.concatInput, units=CRITIC_ARCHITECTURE[1], activation=tf.nn.leaky_relu)
            self.predictedStateValue = tf.layers.dense(inputs=self.secondHiddenLayer, units=1)
            self.actualStateValue = tf.placeholder(tf.float32, [None,1])
            self.predictionLoss = tf.losses.mean_squared_error(labels=self.actualStateValue, predictions=self.predictedStateValue)
            self.weightDecayLoss = tf.losses.l2_decay
            self.trainingOperation = tf.train.AdamOptimizer(CRITIC_LEARNING_RATE).minimize(self.predictionLoss)
            self.actorGradients = tf.gradients(self.predictedStateValue, self.actionInput)[0] 


with tf.Session() as sess:
    env = gym.make('Pendulum-v0')
    agent = ActorCritic(sess, env)
    agent.initialize()
    stepNumber = 0
    while(True):
        if (RENDERING):
            env.render()
        agent.testEpsilon()
        agent.act()
        agent.train()
        stepNumber = stepNumber + 1
        if stepNumber % STEPS_TO_RESET == 0:
            env.reset()
            stepNumber = 0