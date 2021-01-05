import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch.optim as optim
from CartPole.Strategy import EpsilonGreedyStrategy
from CartPole.Agent import Agent
from CartPole.ReplayMemory import Memory
from CartPole.DQNN import Net

# parametres
test = False # test = False : train
numEpisodes = 200
maxStep = 100
epsilon = 1 # ratio d'exploration (exploration rate)
epsilonMax = 1
epsilonMin = 0.01
epsilonDecay = 0.1 # valeur de décret de l'exploration rate
LR = 1e-4
gamma = 0.999 # ratio de réduction (discount rate)
alpha = 0.001 # ratio d'apprentissage (learning Rate)
bufferSize = 10000
batchSize = 100
trainStep = 1000
numTests = 10

env = gym.make('CartPole-v1').unwrapped
actionSize = env.action_space.n
stateShape = env.observation_space.shape[0]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
strategy = EpsilonGreedyStrategy(epsilon, epsilonMin, epsilonDecay)

memory = Memory(bufferSize)

policyNet = Net(actionSize).to(device)
targetNet = Net(actionSize).to(device)
try:
    policyNet.load_state_dict(torch.load("save/policymodelCartpool.data", map_location=device))
    targetNet.load_state_dict(torch.load("save/policymodelCartpool.data", map_location=device))
except:
    pass
targetNet.load_state_dict(policyNet.state_dict())
#Pour préciser que ça ne sera utiliser que pour evaluer
targetNet.eval()
optimizer = optim.Adam(params=policyNet.parameters(), lr=LR)
criterion = nn.MSELoss().to(device)
agent = Agent(strategy, actionSize, device, memory, targetNet, policyNet, optimizer)
scores = []

def plotting(score):
    plt.figure(2)
    plt.clf()
    plt.title("Reward")
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.plot(score)
    plt.grid()
    plt.pause(0.001)

if not test:
    for e in range(numEpisodes):
        state = env.reset()
        score = 0

        for s in range(maxStep):
            env.render()

            action = agent.select_action(state)
            nextState, reward, done, info = env.step(action)
            agent.remember(state, action, nextState, reward, done)

            score += reward
            if memory.memoryFSpace() > batchSize:
                agent.learn(trainStep, batchSize, gamma)
            state = nextState
        scores.append(score)

        print("Episode : ", e, " | Steps : ", scores[-1])

        if e % 20 == 0 :  # Sauvegarde du DQN tout les 20 episodes
            print("Saved !")
            torch.save(policyNet.state_dict(), "save/policymodelCartpool.data")

        plotting(scores)  # Affichage reward temps réel

    torch.save(policyNet.state_dict(), "save/policymodel_cartpool.data")

score = []
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
    print("Test Episode ", k + 1, " : ", s)

print("AVG : ", np.mean(score))
print("STD : ", np.std(score))
env.close()



