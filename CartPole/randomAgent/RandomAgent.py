import gym
import matplotlib.pyplot as plt
from CartPole.randomAgent.ReplayExperience import Replay

# parametres
numEpisods = 10
maxStep = 100
bufferSize = 10000
batchSize = 100

# variables
sumreward = 0
rewardPerEpisod = {}

# instances
env = gym.make('CartPolee-v1')
memory = Replay(bufferSize)


for i_episode in range(numEpisods):
    state = env.reset()
    done = False
    observation = env.reset()
    for t in range(maxStep):
        env.render()
        action = env.action_space.sample()
        nextState, reward, done, info = env.step(action)
        # ajout de l'interaction dans la memoire
        memory.fillMemoryBuffer((state, action, nextState, reward, done))

        # mise à jour de l'état vers le nouvel etat ou se trouve l'agent après l'execution de l'action
        state = nextState
        #calcul récompense
        sumreward += reward

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            rewardPerEpisod[sumreward] = i_episode+1
            sumreward = 0
            break

print(memory.getBatch(batchSize))
print("reward:episode")
print(rewardPerEpisod)

lists = sorted(rewardPerEpisod.items())
x, y = zip(*lists)
plt.clf()
plt.plot(x, y)
plt.title("Suivi évolution")
plt.xlabel("Récompense")
plt.ylabel("Épisode")
plt.grid()
plt.savefig('exemple.png')
plt.pause(0.001)
env.close()
