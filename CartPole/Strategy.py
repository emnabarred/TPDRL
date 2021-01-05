import math

class EpsilonGreedyStrategy():

    def __init__(self, epsilon, epsilonMax, epsilonDecay):
        self.epsilon = epsilon
        self.epsilonMax = epsilonMax
        self.epsilonDecay = epsilonDecay

    def getExplorationRate(self, currentStep):
        return self.epsilonMax + (self.epsilonDecay - self.epsilonMax) * math.exp(-1. * currentStep * self.epsilonDecay)