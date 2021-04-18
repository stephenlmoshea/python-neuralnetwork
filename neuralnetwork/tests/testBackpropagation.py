import unittest

from FeedForward import FeedForward
from Sigmoid import Sigmoid
from HyperbolicTangent import HyperbolicTangent
from Backpropagation import Backpropagation
from .testBase import TestBase

class TestBackpropagation(TestBase):

    def testCalculateNodeDeltasWithTwoOutputs(self):
        networkLayer = [2, 2, 2]
        sigmoid = Sigmoid()
        feedForward = FeedForward(networkLayer, sigmoid)

        backpropagation = Backpropagation(feedForward,0.7,0.3, 0.005, 1)
        backpropagation.initialise()
        self.initialiseNetworkWithTwoOutputs(feedForward)

        trainingSet = [0, 0, 0, 0]

        feedForward.activate(trainingSet)
        backpropagation.calculateNodeDeltas(trainingSet)

        expectedNodeDeltas = [0.0, 0.0, 0.0006232698073582778, -0.00030381413438377187, -0.126246516536673, -0.1246820107165854]

        self.assertEquals(backpropagation.getNodeDeltas(), expectedNodeDeltas)

    def testCalculateGradientsWithTwoOutputs(self):
        networkLayer = [2, 2, 2]
        sigmoid = Sigmoid()
        feedForward = FeedForward(networkLayer, sigmoid)

        backpropagation = Backpropagation(feedForward,0.7,0.3, 0.005, 1)
        backpropagation.initialise()
        self.initialiseNetworkWithTwoOutputs(feedForward)

        trainingSet = [0, 0, 0, 0]

        feedForward.activate(trainingSet)
        backpropagation.calculateNodeDeltas(trainingSet)
        backpropagation.calculateGradients()

        expectedGradients = []

        expectedGradients = [0] * feedForward.getTotalNumNodes()
        for i in range(feedForward.getTotalNumNodes()):
            expectedGradients[i] = [0] * feedForward.getTotalNumNodes()

        expectedGradients[0][2] = 0
        expectedGradients[0][3] = -0
        expectedGradients[1][2] = 0
        expectedGradients[1][3] = -0
        expectedGradients[2][4] = -0.062176480401586354
        expectedGradients[2][5] = -0.06140596040523789
        expectedGradients[3][4] = -0.062176480401586354
        expectedGradients[3][5] = -0.06140596040523789

        self.assertEquals(backpropagation.getGradients(), expectedGradients)

        expectedBiasGradients = []

        expectedBiasGradients = [0] * feedForward.getTotalNumNodes()
        for i in range(feedForward.getTotalNumNodes()):
            expectedBiasGradients[i] = [0] * feedForward.getTotalNumNodes()

        expectedBiasGradients[0][2] = 0.0006232698073582778
        expectedBiasGradients[0][3] = -0.00030381413438377187
        expectedBiasGradients[1][4] = -0.126246516536673
        expectedBiasGradients[1][5] = -0.1246820107165854

        self.assertEquals(backpropagation.getBiasGradients(), expectedBiasGradients)

    
    def testCalculateWeightUpdatesWithTwoOutputs(self):
        networkLayer = [2, 2, 2]
        sigmoid = Sigmoid()
        feedForward = FeedForward(networkLayer, sigmoid)

        backpropagation = Backpropagation(feedForward,0.7,0.3, 0.005, 1)
        backpropagation.initialise()
        self.initialiseNetworkWithTwoOutputs(feedForward)

        trainingSet = [0, 0, 0, 0]

        feedForward.activate(trainingSet)
        backpropagation.calculateNodeDeltas(trainingSet)
        backpropagation.calculateGradients()
        backpropagation.calculateWeightUpdates()

        expectedWeightUpdates = []

        expectedWeightUpdates = [0] * feedForward.getTotalNumNodes()
        for i in range(feedForward.getTotalNumNodes()):
            expectedWeightUpdates[i] = [0] * feedForward.getTotalNumNodes()

        expectedWeightUpdates[0][2] = 0
        expectedWeightUpdates[0][3] = 0
        expectedWeightUpdates[1][2] = 0
        expectedWeightUpdates[1][3] = 0
        expectedWeightUpdates[2][4] = -0.04352353628111044
        expectedWeightUpdates[2][5] = -0.04298417228366652
        expectedWeightUpdates[3][4] = -0.04352353628111044
        expectedWeightUpdates[3][5] = -0.04298417228366652

        self.assertEquals(backpropagation.getWeightUpdates(), expectedWeightUpdates)

        expectedBiasWeightUpdates = []

        expectedBiasWeightUpdates = [0] * feedForward.getTotalNumNodes()
        for i in range(feedForward.getTotalNumNodes()):
            expectedBiasWeightUpdates[i] = [0] * feedForward.getTotalNumNodes()

        expectedBiasWeightUpdates[0][2] = 0.0004362888651507944
        expectedBiasWeightUpdates[0][3] = -0.0002126698940686403
        expectedBiasWeightUpdates[1][4] = -0.08837256157567108
        expectedBiasWeightUpdates[1][5] = -0.08727740750160977

        self.assertEquals(backpropagation.getBiasWeightUpdates(), expectedBiasWeightUpdates)


    def testApplyWeightChangesWithTwoOutputs(self):
        networkLayer = [2, 2, 2]
        sigmoid = Sigmoid()
        feedForward = FeedForward(networkLayer, sigmoid)

        backpropagation = Backpropagation(feedForward,0.7,0.3, 0.005, 1)
        backpropagation.initialise()
        self.initialiseNetworkWithTwoOutputs(feedForward)

        trainingSet = [0, 0, 0, 0]

        feedForward.activate(trainingSet)
        backpropagation.calculateNodeDeltas(trainingSet)
        backpropagation.calculateGradients()
        backpropagation.calculateWeightUpdates()

        expectedWeights = []

        expectedWeights = [0] * feedForward.getTotalNumNodes()
        for i in range(feedForward.getTotalNumNodes()):
            expectedWeights[i] = [0] * feedForward.getTotalNumNodes()

        expectedWeights[0][2] = 0.01
        expectedWeights[0][3] = -0.01
        expectedWeights[1][2] = 0.04
        expectedWeights[1][3] = 0.04
        expectedWeights[2][4] = 0
        expectedWeights[2][5] = -0.02
        expectedWeights[3][4] = -0.02
        expectedWeights[3][5] = 0.03

        self.assertEquals(feedForward.getWeights(), expectedWeights)

        expectedBiasWeights = []

        expectedBiasWeights = [0] * feedForward.getTotalNumNodes()
        for i in range(feedForward.getTotalNumNodes()):
            expectedBiasWeights[i] = [0] * feedForward.getTotalNumNodes()

        expectedBiasWeights[0][2] = -0.03
        expectedBiasWeights[0][3] = -0.03
        expectedBiasWeights[1][4] = 0.03
        expectedBiasWeights[1][5] = -0.01

        self.assertEquals(feedForward.getBiasWeights(), expectedBiasWeights)

    def testCalculateNetworkErrorWithTwoOutputs(self):
        networkLayer = [2, 2, 2]
        sigmoid = Sigmoid()
        feedForward = FeedForward(networkLayer, sigmoid)

        backpropagation = Backpropagation(feedForward,0.7,0.3, 0.005, 1)
        backpropagation.initialise()
        self.initialiseNetworkWithTwoOutputs(feedForward)

        trainingSet = [0, 0, 0, 0]

        feedForward.activate(trainingSet)
        backpropagation.calculateNodeDeltas(trainingSet)
        backpropagation.calculateGradients()
        backpropagation.calculateWeightUpdates()

        self.assertEquals(backpropagation.calculateNetworkError(trainingSet),0.5037955652561967)
        feedForward.activate([0,0])
        outputs = feedForward.getOutputs()
        print(outputs)

    def testItLearnsWeightsAndOutputsForXORFunctionWithTwoOutputs(self):
        nodesPerLayer = [2, 2, 2]
        activation = Sigmoid()
        network = FeedForward(nodesPerLayer, activation)

        backpropagation = Backpropagation(network,0.7,0.3,0.005,1)

        self.initialiseNetworkWithTwoOutputs(network)

        trainingSet = [
            [0,0,0,0],
            [0,1,0,1],
            [1,0,1,0],
            [1,1,0,0]
        ]

        while True:
            result = backpropagation.train(trainingSet)

            if(result):
                break

        network.activate([0,0])
        outputs = network.getOutputs()

        self.assertEquals(outputs[0], 0.07397475104807594)
        self.assertEquals(outputs[1], 0.07640587319838178)

        network.activate([0,1])
        outputs = network.getOutputs()
        
        self.assertEquals(outputs[0], 0.0011872968318554168)
        self.assertEquals(outputs[1], 0.9006706090890231)

        network.activate([1,0])
        outputs = network.getOutputs()
        
        self.assertEquals(outputs[0], 0.9022231252649556)
        self.assertEquals(outputs[1], 0.0008008541187349612)

        network.activate([1,1])
        outputs = network.getOutputs()

        self.assertEquals(outputs[0], 0.06389865849681756)
        self.assertEquals(outputs[1], 0.06729508546056018)

        expectedWeights = []
        expectedWeights = [0] * network.getTotalNumNodes()
        for i in range(network.getTotalNumNodes()):
            expectedWeights[i] = [0] * network.getTotalNumNodes()

        expectedWeights[0][2] = 3.047244161837838
        expectedWeights[0][3] = -3.705464338045177
        expectedWeights[1][2] = -2.5961449172696365
        expectedWeights[1][3] = 3.951078577457027
        expectedWeights[2][4] = 2.610569918098228
        expectedWeights[2][5] = -4.73001794729599
        expectedWeights[3][4] = -7.044199442098878
        expectedWeights[3][5] = 5.530035155194141

        self.assertEquals(expectedWeights, network.getWeights())

        expectedBiasWeights = []
        expectedBiasWeights = [0] * network.getTotalNumNodes()
        for i in range(network.getTotalNumNodes()):
            expectedBiasWeights[i] = [0] * network.getTotalNumNodes()

        expectedBiasWeights[0][2] = 0.5516274985420081
        expectedBiasWeights[0][3] = 0.3297732838238489
        expectedBiasWeights[1][4] = -0.08597799352073608
        expectedBiasWeights[1][5] = -2.70779954102211

        self.assertEquals(expectedBiasWeights, network.getBiasWeights())

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

    def testItLearnsXOrFunctionWithTwoHiddenUnitsAndTwoOutputs(self):
        sigmoid = Sigmoid()

        networkLayer = [2,2,2]

        feedForward = FeedForward(networkLayer, sigmoid)

        backpropagation = Backpropagation(feedForward,0.7,0.3,0.005)

        trainingSet = [
            [0,0,0,0],
            [0,1,0,1],
            [1,0,1,0],
            [1,1,0,0]
        ]

        while True:
            backpropagation.initialise()
            result = backpropagation.train(trainingSet)

            if(result):
                break

        feedForward.activate([0,0])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < 0.1)
        self.assertTrue(outputs[1] < 0.1)

        feedForward.activate([0,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < 0.1)
        self.assertTrue(outputs[1] > 0.9)

        feedForward.activate([1,0])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)
        self.assertTrue(outputs[1] < 0.1)

        feedForward.activate([1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < 0.1)
        self.assertTrue(outputs[1] < 0.1)

    def testItLearnsXOrFunctionWithThreeHiddenUnitsAndTwoOutputs(self):
        sigmoid = Sigmoid()

        networkLayer = [2,3,2]

        feedForward = FeedForward(networkLayer, sigmoid)

        backpropagation = Backpropagation(feedForward,0.7,0.3,0.002)

        trainingSet = [
            [0,0,0,0],
            [0,1,0,1],
            [1,0,1,0],
            [1,1,0,0]
        ]

        while True:
            backpropagation.initialise()
            result = backpropagation.train(trainingSet)

            if(result):
                break

        feedForward.activate([0,0])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < 0.1)
        self.assertTrue(outputs[1] < 0.1)

        feedForward.activate([0,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < 0.1)
        self.assertTrue(outputs[1] > 0.9)

        feedForward.activate([1,0])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] > 0.9)
        self.assertTrue(outputs[1] < 0.1)

        feedForward.activate([1,1])
        outputs = feedForward.getOutputs()
        self.assertTrue(outputs[0] < 0.1)
        self.assertTrue(outputs[1] < 0.1)

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