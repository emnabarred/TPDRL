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

# instances
env = gym.make('CartPole-v1')
memory = Replay(bufferSize)
scores= []

def plotting(score):
    plt.clf()
    plt.plot(score)
    plt.title("Suivi évolution")
    plt.xlabel("Épisode")
    plt.ylabel("Pas")
    plt.grid()
    plt.savefig('evolutionRandomAgent.png')
    plt.pause(0.001)

for i_episode in range(numEpisods):
    state = env.reset()
    done = False
    observation = env.reset()
    for t in range(maxStep):
        env.render()
        action = env.action_space.sample()
        nextState, reward, done, _ = env.step(action)
        # ajout de l'experience dans la memoire de replay
        memory.fillMemoryBuffer((state, action, nextState, reward, done))

        # mise à jour de l'état vers le nouvel etat ou se trouve l'agent après l'execution de l'action
        state = nextState
        #calcul récompense
        sumreward += reward

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            scores.append(sumreward)
            sumreward = 0
            break


plotting(scores)
print(memory.getBatch(batchSize))
print("reward:episode")
env.close()


