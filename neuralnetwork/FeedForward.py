from numpy import random
import pickle


class FeedForward:
    networkLayers = []
    activation = ""
    totalNumNodes = 0
    net = []
    weights = []
    biasWeights = []
    values = []

    def __init__(self, networkLayers, activation):
        """
        __init__ [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        networkLayers : [type]
            [description]
        activation : [type]
            [description]
        """
        self.networkLayers = []
        startNode = 0
        endNode = 0
        for layer, numNodes in enumerate(networkLayers):
            if layer > 0:
                startNode += networkLayers[layer - 1]
            endNode += numNodes
            self.networkLayers.append(
                {
                    "num_nodes": numNodes,
                    "start_node": startNode,
                    "end_node": endNode - 1,
                }
            )
        self.totalNumNodes = sum(networkLayers)
        self.activation = activation

    def initialise(self):
        """
        initialise [summary]
        
        [extended_summary]
        """
        self.net = []
        self.weights = [0] * self.totalNumNodes
        for i in range(self.totalNumNodes):
            self.weights[i] = [0] * self.totalNumNodes
        self.values = []
        self.biasWeights = [0] * self.totalNumNodes
        for i in range(self.totalNumNodes):
            self.biasWeights[i] = [0] * self.totalNumNodes
        self.initialiseValues()
        self.initialiseWeights()

    def initialiseValues(self):
        """
        initialiseValues [summary]
        
        [extended_summary]
        """
        self.values = [0.00] * self.totalNumNodes
        self.net = [0.00] * self.totalNumNodes

    def initialiseWeights(self):
        """
        initialiseWeights S
        
        [extended_summary]
        """
        for num, layer in enumerate(self.networkLayers):
            if num < len(self.networkLayers) - 1:
                for i in range(layer["start_node"], layer["end_node"] + 1):
                    for j in range(
                        self.networkLayers[num + 1]["start_node"],
                        self.networkLayers[num + 1]["end_node"] + 1,
                    ):
                        self.weights[i][j] = random.randint(-5, 5) / 100
                for b in range(
                    self.networkLayers[num + 1]["start_node"],
                    self.networkLayers[num + 1]["end_node"] + 1,
                ):
                    self.biasWeights[num][b] = random.randint(-5, 5) / 100

    def activate(self, inputs):
        """
        activate [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        inputs : [type]
            [description]
        """
        for z in range(0, self.networkLayers[0]["num_nodes"]):
            self.values[z] = inputs[z]
        for num, layer in enumerate(self.networkLayers):
            if num > 0:
                for j in range(layer["start_node"], layer["end_node"] + 1):
                    net = 0
                    for i in range(
                        self.networkLayers[num - 1]["start_node"],
                        self.networkLayers[num - 1]["end_node"] + 1,
                    ):
                        net += float(self.values[i]) * self.weights[i][j]
                    net += self.biasWeights[num - 1][j]
                    self.net[j] = net
                    self.values[j] = self.activation.getActivation(net)

    def getOutputs(self):
        """
        getOutputs [summary]
        
        [extended_summary]
        
        Returns
        -------
        [type]
            [description]
        """
        startNode = self.networkLayers[len(self.networkLayers) - 1]["start_node"]
        endNode = self.networkLayers[len(self.networkLayers) - 1]["end_node"]
        return self.values[startNode : endNode + 1]

    def getNetworkLayers(self):
        """
        getNetworkLayers [summary]
        
        [extended_summary]
        
        Returns
        -------
        [type]
            [description]
        """
        return self.networkLayers

    def getValue(self, index):
        """
        getValue [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        index : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        return self.values[index]

    def getValues(self):
        """
        getValues [summary]
        
        [extended_summary]
        
        Returns
        -------
        [type]
            [description]
        """
        return self.values

    def getActivation(self):
        """
        getActivation [summary]
        
        [extended_summary]
        
        Returns
        -------
        [type]
            [description]
        """
        return self.activation

    def getNet(self, index):
        """
        getNet [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        index : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        return self.net[index]

    def getWeight(self, index):
        """
        getWeight [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        index : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        return self.weights[index]

    def getBiasWeights(self):
        """
        getBiasWeights [summary]
        
        [extended_summary]
        
        Returns
        -------
        [type]
            [description]
        """
        return self.biasWeights

    def getBiasWeight(self, index):
        """
        getBiasWeight [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        index : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        return self.biasWeights[index]

    def setBiasWeights(self, biasWeights):
        """
        setBiasWeights [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        biasWeights : [type]
            [description]
        """
        self.biasWeights = biasWeights

    def updateWeight(self, i, j, weight):
        """
        updateWeight [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        i : [type]
            [description]
        j : [type]
            [description]
        weight : [type]
            [description]
        """
        self.weights[i][j] += weight

    def updateBiasWeight(self, i, j, weight):
        """
        updateBiasWeight [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        i : [type]
            [description]
        j : [type]
            [description]
        weight : [type]
            [description]
        """
        self.biasWeights[i][j] += weight

    def getTotalNumNodes(self):
        """
        getTotalNumNodes [summary]
        
        [extended_summary]
        
        Returns
        -------
        [type]
            [description]
        """
        return self.totalNumNodes

    def save(self, filename):
        """
        save [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        filename : [type]
            [description]
        """
        with open(filename, "wb") as network_file:
            pickle.dump(self, network_file)

    @staticmethod
    def load(filename):
        """
        load [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        filename : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        with open(filename, "rb") as network_file:
            network = pickle.load(network_file)
            return network
