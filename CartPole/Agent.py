import random
import torch

class Agent():
    def __init__(self, strategy, actionSize, device, memory, targetNet, policyNet, optimizer, criterion):
        self.currentStep = 0
        self.steoCounter = 0
        self.strategy = strategy
        self.actionsSize = actionSize
        self.device = device
        self.memory = memory
        self.targetNet = targetNet
        self.policyNet = policyNet
        self.optimizer = optimizer
        self.criterion = criterion


    def select_action(self, state, test=False):
        lr = self.strategy.getExplorationRate(self.currentStep)
        self.currentStep += 1
        state = torch.FloatTensor(state).to(self.device)
        Q = self.targetNet(state).view([-1])
        # si test est false, c'est à dire qu'on fait de l'exploitation
        # exploitation: on prend l'action qui a la q valeur maximale
        if test:
            action = torch.argmax(Q).item()


        # En mode test: si le learning rate est inferieur à
        # la variable aléatoire r, c'est qu'il faut exploiter
        else:
            if lr < random.uniform(0, 1) or test:
                action = torch.argmax(Q).item()
            #sinon on explore au hazard parmis les actions possible
            else:
                action = random.randrange(self.actionsSize)

        self.targetNet.train()
        return action

    def remember(self, state, action, nextState, reward, done):
        self.memory.fillMemoryBuffer([state, action, nextState, reward, done])

    def learn(self, trainStep, batchSize, gamma):
        r = self.strategy.getExplorationRate(self.currentStep)
        self.sc =+1
        print(self.sc)

        if self.sc < trainStep:
            return

        state, action, nextState, reward, done = self.memory.getBatch(batchSize)
        Qcalcul = self.targetNet(state).gather(1, action.long().unsqueeze(1))
        Qcalcul = Qcalcul.reshape([batchSize])
        nextQ = self.targetNet(nextState).detach
        Qtarget = reward + gamma * nextQ.max(1)[0].reshape([batchSize])

        # Optimization
        self.optimizer.zero_grad()
        loss = self.criterion(Qcalcul, Qtarget)
        loss.backward()
        self.optimizer.step()

        # LOG
        if self.sc % 1000 == 0:
            print("Step ", self.sc, " : Loss = ", loss, ", Epsilon : ", r)