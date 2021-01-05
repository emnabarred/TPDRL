import random

class Replay():
    def __init__(self, bufferSize):
        self.bufferSize = bufferSize
        self.memoryBuffer = []

    def fillMemoryBuffer(self, interaction):
        if len(self.memoryBuffer) <= self.bufferSize:
            self.memoryBuffer.append(interaction)
        else:
            self.memoryBuffer.pop(0)
            self.memoryBuffer.append(interaction)

    def getBatch(self, batchSize):
        if batchSize <= len(self.memoryBuffer):
            return random.sample(self.memoryBuffer, batchSize)
        else:
            return random.sample(self.memoryBuffer, len(self.memoryBuffer))
