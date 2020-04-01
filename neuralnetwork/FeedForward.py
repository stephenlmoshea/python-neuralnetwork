import numpy as np
import pickle


class FeedForward:
    """
     [summary]
    
    [extended_summary]
    
    Returns
    -------
    [type]
        [description]
    """

    networkLayers = np.array([])
    activation = ""
    totalNumNodes = 0
    net = np.array([])
    weights = np.array([])
    biasWeights = np.array([])
    values = np.array([])

    def __init__(self, networkLayers, activation, dtype=np.float64):
        """
        __init__ [summary]
       
        [extended_summary]
       
        Parameters
        ----------
        networkLayers : [type]
            [description]
        activation : [type]
            [description]
        dtype : data type objects, optional
            The data-type objects (dtype) can be set to single (np.float32) or double (np.float64) precission, by default np.float64       
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
        self.totalNumNodes = np.sum(networkLayers)
        self.activation = activation
        self.dtype = dtype

    def initialise(self):
        """initialise bias-neurons and their weights.
        
        Bias-neurons- and their weights-matrices will be initialise by using np.zeros. The data-type (dtype) can be set 
        in the __init__.py, and can be choosen between single (np.float32) and double (np.float64) precission.
        """
        self.weights = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.biasWeights = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )

        self.initialiseValuesNet()
        self.initialiseWeights()

    def initialiseValuesNet(self):
        """initialiseValuesNet values and their net.
        
        [extended_summary]
        """

        self.values = np.zeros(self.totalNumNodes, dtype=self.dtype)
        self.net = np.zeros(self.totalNumNodes, dtype=self.dtype)

    def initialiseWeights(self, low=-5, high=+5):
        """initialiseWeights for the weights and the bias-weights.
        
        Based on the zero-matrix, the weights- and bias-weights-matrix will be filled with random-int, which
        becomes float by np.divide. 
        
        Notes
        -----
        np.divide is important because it will also keep the dtype-format consistent.
        
        Parameters
        ----------
        low : int, optional
            lowest random-value, by default -5
        high : int, optional
            highes random-value, by default +5
        """

        self.weights = np.divide(
            np.random.randint(low=low, high=high, size=self.weights.shape),
            100.0,
            dtype=self.dtype,
        )
        self.biasWeights = np.divide(
            np.random.randint(low=low, high=high, size=self.biasWeights.shape),
            100.0,
            dtype=self.dtype,
        )

    def activate(self, inputs):
        """activate the forward propagation.
        
        For activate the forward propagation, the values(dendrites) will be elementwise combined with the 
        weights (synapes) plus the bias. This will be performed by numpy. Furthermore, the activation functions
        will be activated to the values.
        
        Notes:
        ------
        A more detail description is provided by Deep Learning: Ian Goodfellow et al. page 205
        
        Parameters
        ----------
        inputs : array
            inputs as float array to be processed
        """
        # Defining the h^0 = x
        _end = self.networkLayers[0]["num_nodes"]
        self.values[0:_end] = inputs[0:_end]

        # Connecting the layers (input, hidden, and target) via j-index
        for i, layer in enumerate(self.networkLayers[1:]):
            # Prevouis layer
            j = np.arange(
                self.networkLayers[i]["start_node"],
                self.networkLayers[i]["end_node"] + 1,
            )
            # Current layer
            k = np.arange(layer["start_node"], layer["end_node"] + 1)

            # Apply feedback transformation
            self.net[k] = np.add(
                self.biasWeights[i, k], np.dot(self.values[j], self.weights[j, k])
            )
            
            # Apply activation function
            self.values[k] = self.activation.getActivation(self.net[k])

    def getOutputs(self):
        """getOutputs returns the predicted values
        
        Returns
        -------
        array
            Returns the predicted values as float array
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
