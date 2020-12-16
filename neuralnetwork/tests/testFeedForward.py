import unittest

from Backpropagation import Backpropagation
from FeedForward import FeedForward
from HyperbolicTangent import HyperbolicTangent
from Sigmoid import Sigmoid
from .testBase import TestBase


class TestFeedForward(TestBase):
    network = ''
    nodesPerLayer = []
    activation = ''
    layers = []
    weights = []
    biasWeights = []

    def setUp(self):
        self.nodesPerLayer = [2, 2, 1]
        self.layers = self.getLayers()
        self.activation = Sigmoid()
        self.network = FeedForward(self.nodesPerLayer, self.activation)

    def testItCreatesANetworkWithValidParams(self):
        self.assertEquals(self.layers, self.network.getNetworkLayers())
        self.assertEquals(self.activation, self.network.getActivation())

    def testNetworkWeightsAreInitialisedToRandomValuesInCorrectRange(self):
        self.network.initialise()
        weights = self.network.getWeights()
        for i,connections in enumerate(weights):
            for j,value in enumerate(connections):
                if value != 0:
                    self.assertLessEqual(abs(value),0.05)

    def testNetworkValuesAreInitialised(self):
        self.network.initialise()
        values = self.network.getValues()
        for i,value in enumerate(values):
            self.assertLessEqual(abs(value),0)

    def testNetworkLayersAreFullyConnected(self):
        self.network.initialise()
        weights = self.network.getWeights()
        self.assertLessEqual(abs(weights[0][2]),0.05)
        self.assertLessEqual(abs(weights[0][3]),0.05)
        self.assertLessEqual(abs(weights[1][2]),0.05)
        self.assertLessEqual(abs(weights[1][3]),0.05)
        self.assertLessEqual(abs(weights[2][4]),0.05)
        self.assertLessEqual(abs(weights[3][4]),0.05)

    def testActivatingNetworkWithValidInputsProducesValidOutput(self):
        self.initialiseNetwork()
        self.network.activate([0,1])
        outputs = self.network.getOutputs()
        self.assertEquals(round(outputs[0],2), 0.49)

    def testActivatingNetworkWithValidInputsProducesValidOutputs(self):
        nodesPerLayer = [2, 2, 2]
        activation = Sigmoid()
        network = FeedForward(nodesPerLayer, activation)

        self.initialiseNetworkWithTwoOutputs(network)
        network.activate([0,1])
        outputs = network.getOutputs()
        self.assertEquals(round(outputs[0],3), 0.505)
        self.assertEquals(round(outputs[1],3), 0.499)
        
    def getLayers(self):
        return [
                    {
                        'num_nodes' : 2,
                        'start_node' : 0,
                        'end_node' : 1
                    },
                    {
                        'num_nodes' : 2,
                        'start_node' : 2,
                        'end_node' : 3
                    },
                    {
                        'num_nodes' : 1,
                        'start_node' : 4,
                        'end_node' : 4
                    }
                ]

    def getLayersForTwoOutputs(self):
        return [
                    {
                        'num_nodes' : 2,
                        'start_node' : 0,
                        'end_node' : 1
                    },
                    {
                        'num_nodes' : 2,
                        'start_node' : 2,
                        'end_node' : 3
                    },
                    {
                        'num_nodes' : 2,
                        'start_node' : 4,
                        'end_node' : 5
                    }
                ]

    def initialiseNetwork(self):
        self.network.initialise()
        weights = self.getWeights()
        biasWeights = self.getBiasWeights()

        self.network.setWeights(weights)
        self.network.setBiasWeights(biasWeights)

        return {
            'weights' : weights,
            'biasWeights' : biasWeights
        }

    def getWeights(self):
        weights = []
        weights = [0] * self.network.getTotalNumNodes()
        for i in range(self.network.getTotalNumNodes()):
            weights[i] = [0] * self.network.getTotalNumNodes()

        weights[0][2] = -0.04
        weights[0][3] = 0.03
        weights[1][2] = -0.04
        weights[1][3] = -0.02
        weights[2][4] = -0.05
        weights[3][4] = -0.03
        
        return weights

    def getBiasWeights(self):
        biasWeights = []
        biasWeights = [0] * self.network.getTotalNumNodes()
        for i in range(self.network.getTotalNumNodes()):
            biasWeights[i] = [0] * self.network.getTotalNumNodes()

        biasWeights[0][2] = 0.04
        biasWeights[0][3] = 0.02
        biasWeights[1][4] = -0.02
        
        return biasWeights


