import random
import torch

NB_EPOCHS = 10

class Agent():
    def __init__(self, strategy, actionSize, device, memory, targetNet, policyNet, optimizer, criterion):
        self.currentStep = 0
        self.stepCounter = 0
        self.strategy = strategy
        self.actionsSize = actionSize
        self.device = device
        self.memory = memory
        self.targetNet = targetNet
        self.policyNet = policyNet
        self.optimizer = optimizer
        self.criterion = criterion

    def select_action(self, state, test=False):
        r = random.uniform(0, 1)
        self.currentStep += 1
        state = torch.FloatTensor(state).to(self.device)
        Q = self.policyNet(state)
        # si test est false, c'est à dire qu'on fait de l'exploitation
        # exploitation: on prend l'action qui a la q valeur maximale
        if test:
            action = torch.argmax(Q).item()
        # En mode test: si le learning rate est inferieur à
        # la variable aléatoire r, c'est qu'il faut exploiter
        else:
            if r < self.strategy.epsilon:
                # on explore au hazard parmis les actions possible
                action = random.randrange(self.actionsSize)
            else:
                # on choisit la meilleure action
                action = torch.argmax(Q).item()
        return action

    def remember(self, state, action, nextState, reward, done):
        self.memory.fillMemoryBuffer([state, action, nextState, reward, done])

    def learn(self, trainStep, batchSize, gamma):
        r = self.strategy.getExplorationRate(self.currentStep)
        self.stepCounter +=1

        if self.stepCounter < trainStep:
            return

        batch_loader_loader = torch.utils.data.DataLoader(self.memory.getBatch(batchSize), batch_size=1, shuffle=True)

        for (state, action, nextState, reward, done) in batch_loader_loader:
            self.optimizer.zero_grad()
            Qcalcul = torch.max(self.policyNet(state.float()))
            Qtarget = torch.max(self.targetNet(state.float()))

            loss = self.criterion(Qtarget, Qcalcul)
            loss.backward()
            self.optimizer.step()

        # LOG
        if self.stepCounter % 1000 == 0:
            self.targetNet.load_state_dict(self.policyNet.state_dict())