import unittest

from FeedForward import FeedForward
from Sigmoid import Sigmoid
from Backpropagation import Backpropagation

class BackpropagationTest(unittest.TestCase):
    def testItLearnsOrFunction(self):
        sigmoid = Sigmoid()

        networkLayer = [2,2,1]

        feedForward = FeedForward(networkLayer, sigmoid)

        backpropagation = Backpropagation(feedForward,0.7,0.3, 0.001)

        trainingSet = [
            [0,0,0],
            [0,1,1],
            [1,0,1],
            [1,1,1]
        ]

        while True:
            backpropagation.initialise()
            result = backpropagation.train(trainingSet)

            if(result):
                break

        feedForward.activate([0,0])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < 0.1)

        feedForward.activate([0,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([1,0])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)