import unittest

class TestBase(unittest.TestCase):
    
    def initialiseNetworkWithTwoOutputs(self, network):
        network.initialise()
        weights = self.getWeightsForTwoOutputs(network)
        biasWeights = self.getBiasWeightsForTwoOutputs(network)

        network.setWeights(weights)
        network.setBiasWeights(biasWeights)

        return {
            'weights' : weights,
            'biasWeights' : biasWeights
        }

    def getWeightsForTwoOutputs(self, network):
        weights = []
        weights = [0] * network.getTotalNumNodes()
        for i in range(network.getTotalNumNodes()):
            weights[i] = [0] * network.getTotalNumNodes()

        weights[0][2] = 0.01
        weights[0][3] = -0.01
        weights[1][2] = 0.04
        weights[1][3] = 0.04
        weights[2][4] = 0
        weights[2][5] = -0.02
        weights[3][4] = -0.02
        weights[3][5] = 0.03
        
        return weights

    def getBiasWeightsForTwoOutputs(self, network):
        biasWeights = []
        biasWeights = [0] * network.getTotalNumNodes()
        for i in range(network.getTotalNumNodes()):
            biasWeights[i] = [0] * network.getTotalNumNodes()

        biasWeights[0][2] = -0.03
        biasWeights[0][3] = -0.03
        biasWeights[1][4] = 0.03
        biasWeights[1][5] = -0.01
        
        return biasWeights