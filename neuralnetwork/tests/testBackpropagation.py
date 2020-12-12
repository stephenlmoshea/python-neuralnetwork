import unittest

from FeedForward import FeedForward
from Sigmoid import Sigmoid
from HyperbolicTangent import HyperbolicTangent
from Backpropagation import Backpropagation

class TestBackpropagation(unittest.TestCase):
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

    def testItLearnsAndFunction(self):
        sigmoid = Sigmoid()

        networkLayer = [2,2,1]

        feedForward = FeedForward(networkLayer, sigmoid)

        backpropagation = Backpropagation(feedForward,0.7,0.3)

        trainingSet = [
            [0,0,1],
            [0,1,0],
            [1,0,0],
            [1,1,1]
        ]

        while True:
            backpropagation.initialise()
            result = backpropagation.train(trainingSet)

            if(result):
                break

        feedForward.activate([0,0])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([0,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < 0.1)

        feedForward.activate([1,0])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < 0.1)

        feedForward.activate([1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

    def testItLearnsXOrFunctionWithTwoHiddenUnits(self):
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
        self.assertTrue(outputs[0] < 0.1)

    def testItLearnsXOrFunctionWithThreeHiddenUnits(self):
        sigmoid = Sigmoid()

        networkLayer = [2,3,1]

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
        self.assertTrue(outputs[0] < 0.1)

    def testItLearnsXOrFunctionWithFourHiddenUnits(self):
        sigmoid = Sigmoid()

        networkLayer = [2,4,1]

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
        self.assertTrue(outputs[0] < 0.1)

    def testItLearnsXOrFunctionWithThreeInputNodes(self):
        sigmoid = Sigmoid()

        networkLayer = [3,4,1]

        feedForward = FeedForward(networkLayer, sigmoid)

        backpropagation = Backpropagation(feedForward,0.7,0.3,0.0005, 5000)

        trainingSet = [
            [0,0,0,0],
            [0,0,1,1],
            [0,1,0,1],
            [0,1,1,0],
            [1,0,0,1],
            [1,0,1,0],
            [1,1,0,0],
            [1,1,1,1]
        ]

        while True:
            backpropagation.initialise()
            result = backpropagation.train(trainingSet)

            if(result):
                break

        feedForward.activate([0,0,0])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < 0.1)

        feedForward.activate([0,1,0])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([1,1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([1,1,0])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < 0.1)

    def testItLearnsOrFunctionUsingHyperbolicTangent(self):
        hyperbolicTangent = HyperbolicTangent()

        networkLayer = [2,2,1]

        feedForward = FeedForward(networkLayer, hyperbolicTangent)

        backpropagation = Backpropagation(feedForward,0.7,0.3, 0.001)

        trainingSet = [
            [-1,-1,-1],
            [-1,1,1],
            [1,-1,1],
            [1,1,1]
        ]

        while True:
            backpropagation.initialise()
            result = backpropagation.train(trainingSet)

            if(result):
                break

        feedForward.activate([-1,-1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < -0.9)

        feedForward.activate([-1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([1,0])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

    def testItLearnsAndFunctionUsingHyperbolicTangent(self):
        hyperbolicTangent = HyperbolicTangent()

        networkLayer = [2,2,1]

        feedForward = FeedForward(networkLayer, hyperbolicTangent)

        backpropagation = Backpropagation(feedForward,0.7,0.3, 0.001)

        trainingSet = [
            [-1,-1,1],
            [-1,1,-1],
            [1,-1,-1],
            [1,1,1]
        ]

        while True:
            backpropagation.initialise()
            result = backpropagation.train(trainingSet)

            if(result):
                break

        feedForward.activate([-1,-1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([-1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < -0.9)

        feedForward.activate([1,-1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < -0.9)

        feedForward.activate([1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

    def testItLearnsXOrFunctionWithTwoHiddenUnitsUsingHyperbolicTangent(self):
        hyperbolicTangent = HyperbolicTangent()

        networkLayer = [2,2,1]

        feedForward = FeedForward(networkLayer, hyperbolicTangent)

        backpropagation = Backpropagation(feedForward,0.7,0.3)

        trainingSet = [
            [-1,-1,-1],
            [-1,1,1],
            [1,-1,1],
            [1,1,-1]
        ]

        while True:
            backpropagation.initialise()
            result = backpropagation.train(trainingSet)

            if(result):
                break

        feedForward.activate([-1,-1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < -0.9)

        feedForward.activate([-1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([1,-1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < -0.9)

    def testItLearnsXOrFunctionWithThreeHiddenUnitsUsingHyperbolicTangent(self):
        hyperbolicTangent = HyperbolicTangent()

        networkLayer = [2,3,1]

        feedForward = FeedForward(networkLayer, hyperbolicTangent)

        backpropagation = Backpropagation(feedForward,0.7,0.3)

        trainingSet = [
            [-1,-1,-1],
            [-1,1,1],
            [1,-1,1],
            [1,1,-1]
        ]

        while True:
            backpropagation.initialise()
            result = backpropagation.train(trainingSet)

            if(result):
                break

        feedForward.activate([-1,-1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < -0.9)

        feedForward.activate([-1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([1,-1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < -0.9)

    def testItLearnsXOrFunctionWithFourHiddenUnitsUsingHyperbolicTangent(self):
        hyperbolicTangent = HyperbolicTangent()

        networkLayer = [2,4,1]

        feedForward = FeedForward(networkLayer, hyperbolicTangent)

        backpropagation = Backpropagation(feedForward,0.7,0.3)

        trainingSet = [
            [-1,-1,-1],
            [-1,1,1],
            [1,-1,1],
            [1,1,-1]
        ]

        while True:
            backpropagation.initialise()
            result = backpropagation.train(trainingSet)

            if(result):
                break

        feedForward.activate([-1,-1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < -0.9)

        feedForward.activate([-1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([1,-1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward.activate([1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < -0.9)

    def testItSavesAndLoadsStateFromFile(self):
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

        feedForward.save('./network')

        feedForward2 = FeedForward.load('./network')

        feedForward2.activate([0,0])
        outputs = feedForward2.getOutputs()
        self.assertTrue(outputs[0] < 0.1)

        feedForward2.activate([0,1])
        outputs = feedForward2.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward2.activate([1,0])
        outputs = feedForward2.getOutputs()
        self.assertTrue(outputs[0] > 0.9)

        feedForward2.activate([1,1])
        outputs = feedForward2.getOutputs()
        self.assertTrue(outputs[0] < 0.1)