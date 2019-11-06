from Activation.Sigmoid import Sigmoid
from Network.FeedForward import FeedForward
from Train.Backpropagation import Backpropagation
from Activation.HyperbolicTangent import HyperbolicTangent

sigmoid = Sigmoid()

networkLayer = [2,2,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.3)

trainingSet = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

while True:
    backpropagation.initialise()
    result = backpropagation.train(trainingSet)

    if(result):
        break

feedForward.save('./network.txt')

feedForward = FeedForward.load('./network.txt')

feedForward.activate([0,0])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([0,1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,0])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,1])
outputs = feedForward.getOutputs()
print(outputs[0])
