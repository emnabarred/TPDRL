import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch.optim as optim
from CartPole.Strategy import EpsilonGreedyStrategy
from CartPole.Agent import Agent
from CartPole.ReplayMemory import Memory
from CartPole.Net import Net

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
trainStep = 100 #N étapes d’apprentissage
numTests = 100

env = gym.make('CartPole-v1').unwrapped
actionSize = env.action_space.n
stateShape = env.observation_space.shape[0]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
strategy = EpsilonGreedyStrategy(epsilon, epsilonMax, epsilonDecay)

memory = Memory(bufferSize)

policyNet = Net(actionSize).to(device)
targetNet = Net(actionSize).to(device)
try:
    policyNet.load_state_dict(torch.load("save/policymodelCartpool.data", map_location=device))
    targetNet.load_state_dict(torch.load("save/policymodelCartpool.data", map_location=device))
    print("loaded from save")
except:
    pass
targetNet.load_state_dict(policyNet.state_dict())
#Pour préciser que ça ne sera utiliser que pour evaluer
targetNet.eval()
optimizer = optim.Adam(params=policyNet.parameters(), lr=alpha)
criterion = nn.MSELoss().to(device)
agent = Agent(strategy, actionSize, device, memory, targetNet, policyNet, optimizer, criterion)
scores = []

def plotting(score):
    plt.clf()
    plt.plot(score)
    plt.title("Suivi évolution")
    plt.xlabel("Épisode")
    plt.ylabel("Pas")
    plt.grid()
    plt.savefig('evolutionCartPoleEgreedy.png')
    plt.pause(0.001)

#training
if not testing:
    for e in range(numEpisodes):
        state = env.reset()
        score = 0
        env.render()
        for s in range(maxStep):
            action = agent.select_action(state)
            nextState, reward, done, info = env.step(action)
            agent.remember(state, action, nextState, reward, done)
            score += reward
           # print(agent.memory.memoryFSpace())
            if agent.memory.memoryFSpace() > batchSize:
                agent.learn(trainStep, batchSize, gamma)
            state = nextState
        scores.append(score)
        if e % 50 == 0 :  # Sauvegarde du model tout les 50 episodes
            print("Saved !")
            torch.save(agent.policyNet.state_dict(), "save/policymodelCartpool.data")

        plotting(scores)  # Affichage reward temps réel

    torch.save(agent.policyNet.state_dict(), "save/policymodel_cartpool.data")

score = []
#testing
for k in range(numTests):
    state = env.reset()
    s = 0
    done = False
    while not done:
        env.render()
        action = agent.select_action(state, test=True)
        state, reward, done, info = env.step(action)
        s += reward
    score.append(s)
   # print("Test Episode ", k + 1, " : ", s)

print("récompense moyenne : ", np.mean(score))

env.close()



