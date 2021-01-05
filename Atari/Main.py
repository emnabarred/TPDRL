import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch.optim as optim
from CartPole.Strategy import EpsilonGreedyStrategy
from CartPole.ReplayMemory import Memory
from Atari.Net import ConvNet

# parametres
testing = False # test = False : train
numEpisodes = 200
maxStep = 100
epsilon = 1 # ratio d'exploration (exploration rate)
epsilonMax = 1
epsilonMin = 0.01
epsilonDecay = 0.1 # valeur de décret de l'exploration rate
gamma = 0.999 # ratio de réduction (discount rate)
alpha = 0.0001 # ratio d'apprentissage (learning Rate)
bufferSize = 10000
batchSize = 30
trainStep = 100
numTests = 100

env = gym.make('BreakoutNoFrameskip-v4')
actionSize = env.action_space.n
stateShape = env.observation_space.shape[0]