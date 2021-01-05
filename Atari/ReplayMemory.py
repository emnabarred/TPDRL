import random

class Memory():
    def __init__(self, bufferSize):
        self.bufferSize = bufferSize
        self.memoryBuffer = []

    def memoryFSpace(self):
        return len(self.memoryBuffer)

    def fillMemoryBuffer(self, interaction):
        if self.memoryFSpace() <= self.bufferSize:
            self.memoryBuffer.append(interaction)
        else:
            self.memoryBuffer.pop(0)
            self.memoryBuffer.append(interaction)

    def getBatch(self, batchSize):
        if batchSize <= self.memoryFSpace():
            return random.sample(self.memoryBuffer, batchSize)
        else:
            return random.sample(self.memoryBuffer, self.memoryFSpace())
