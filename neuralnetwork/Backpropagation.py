import logging
import numpy as np


class Backpropagation:
    """Backpropagation
    
    The Backpropagation class calculates the minimum value of the error function in relation to the training-set and the activation function.
    The technique for achieving this goal is called the delta rule or gradient descent. 
    
    """

    nodeDeltas = np.array([])
    gradients = np.array([])
    biasGradients = np.array([])
    learningRate = np.array([])
    eta = np.array([])
    weightUpdates = np.array([])
    biasWeightUpdates = np.array([])
    minimumError = ""
    maxNumEpochs = ""
    numEpochs = ""
    network = np.array([])
    delta = np.float64
    networkLayers = []

    def __init__(
        self, network, learningRate, eta, minimumError=0.005, maxNumEpochs=2000
    ):
        """
        __init__ [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        network : class
            class of FeedForward-Routine
        learningRate : float
            Learning rate of the MLP
        eta : float
            Error correction factor
        minimumError : float, optional
            Minimal error to stop the training, by default 0.005
        maxNumEpochs : int, optional
            Maxinum numbers of epochs before stopping the training, by default 2000
        """
        self.network = network
        self.learningRate = learningRate
        self.eta = eta
        self.minimumError = minimumError
        self.maxNumEpochs = maxNumEpochs
        self.initialise()

    def initialise(self):
        """
        initialise [summary]
        
        [extended_summary]
        """
        self.network.initialise()
        self.nodeDeltas = np.array([])
        self.gradients = np.array([])
        self.biasGradients = np.array([])
        self.totalNumNodes = self.network.getTotalNumNodes()
        self.dtype = self.network.getDtype()
        self.networkLayers = self.network.getNetworkLayers()
        # initiale the weight, bias, and gradients matrices
        self.weightUpdates = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.biasWeightUpdates = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.gradients = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.biasGradients = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.initialiseValues()

    def initialiseValues(self):
        """
        initialiseValues inital the values array
        """
        self.nodeDeltas = np.zeros(self.totalNumNodes, dtype=self.dtype)

    def train(self, trainingSets):
        """train the mlp-network.
        
        Training of the mlp-network for a given `trainingSets` for maximum number of epchos. 
        
        Parameters
        ----------
        trainingSets : array
            The training set is provided as float-array where X- and y-values are keeped together.
        
        Returns
        -------
         : bool
            Return a bool for indicating successful (True) or failed (False) learning.
        """
        self.numEpochs = 1
        logging.basicConfig(level=logging.DEBUG)
        # Have to change to a for-if slope
        while True:
            if self.numEpochs > self.maxNumEpochs:
                return False
            sumNetworkError = 0
            for i in range(len(trainingSets)):
                # Switching to FeedForworad.py
                self.network.activate(trainingSets[i])
                outputs = self.network.getOutputs()
                # Come back to Backpropagation.py
                self.calculateNodeDeltas(trainingSets[i])
                self.calculateGradients()
                self.calculateWeightUpdates()
                self.applyWeightChanges()
                sumNetworkError += self.calculateNetworkError(trainingSets[i])
            globalError = sumNetworkError / len(trainingSets)
            logging.info("--------------------------------")
            logging.info("Num Epochs: {}".format(self.numEpochs))
            logging.info("Global Error: {}".format(globalError))
            self.numEpochs = self.numEpochs + 1
            if globalError < self.minimumError:
                break
        return True

    def calculateNodeDeltas(self, trainingSet):
        """calculateNodeDeltas, error of each node.
        
        
        Parameters
        ----------
        trainingSets : array
            The training set is provided as float-array where X- and y-values are keeped together.
        """
        idealOutputs = trainingSet[
            -1 * self.networkLayers[len(self.networkLayers) - 1]["num_nodes"]
        ]
        # Initial phase
        startNode = self.networkLayers[len(self.networkLayers) - 1]["start_node"]
        endNode = self.networkLayers[len(self.networkLayers) - 1]["end_node"]
        actl_index = np.arange(startNode, endNode + 1, dtype=np.int)
        activation = self.network.getActivation()
        """
        j = 0
        for i in range(startNode, endNode + 1):
            if isinstance(idealOutputs, list):
                error = self.network.getValueEntry(i) - idealOutputs[j]
            else:
                error = self.network.getValueEntry(i) - idealOutputs
            self.nodeDeltas[i] = (-1 * error) * activation.getDerivative(
                self.network.getNetEntry(i)
            )
            j = j + 1
        print("a", self.nodeDeltas)
        """
        error = self.network.getValueEntry(actl_index) - idealOutputs

        self.nodeDeltas[actl_index] = np.multiply(
            -error,
            activation.getDerivative(self.network.getNetEntry(actl_index)),
            dtype=self.dtype,
        )

        for k in range(len(self.networkLayers) - 2, 0, -1):
            startNode = self.networkLayers[k]["start_node"]
            endNode = self.networkLayers[k]["end_node"]
            actl_index = np.arange(startNode, endNode + 1, dtype=np.int)
            connectNode = np.arange(len(self.network.getWeight()), dtype=np.int)
            # Calculating the node deltas
            self.nodeDeltas[actl_index] = np.multiply(
                np.dot(
                    self.network.getWeightEntry(actl_index),
                    self.nodeDeltas[connectNode],
                ),
                activation.getDerivative(self.network.getNetEntry(actl_index)),
                dtype=self.dtype,
            )

    def calculateGradients(self):
        """calculateGradients, gradient of each value and bias.
        """

        for num, layer in enumerate(self.networkLayers[:-1]):
            prev_index = np.arange(
                layer["start_node"], layer["end_node"] + 1, dtype=np.int
            )  # similiar to i
            actl_index = np.arange(
                self.networkLayers[num + 1]["start_node"],
                self.networkLayers[num + 1]["end_node"] + 1,
                dtype=np.int,
            )  # similiar to j
            # Value-Gradient
            self.gradients[prev_index, actl_index] = np.multiply(
                self.network.getValueEntry(prev_index),
                self.nodeDeltas[actl_index],
                dtype=self.dtype,
            )
            # Bias-Gradient
            self.biasGradients[num, actl_index] = self.nodeDeltas[actl_index]

    def calculateWeightUpdates(self):
        """calculateWeightUpdates of the 'new' weights and bias-weights.
        """
        for num, layer in enumerate(self.networkLayers[:-1]):
            prev_index = np.arange(
                layer["start_node"], layer["end_node"] + 1, dtype=np.int
            )  # similiar to i
            actl_index = np.arange(
                self.networkLayers[num + 1]["start_node"],
                self.networkLayers[num + 1]["end_node"] + 1,
                dtype=np.int,
            )  # similiar to j
            # Updating the weights
            #print(prev_index)
            self.weightUpdates[prev_index, actl_index] = np.add(
                np.multiply(
                    self.learningRate,
                    self.gradients[prev_index, actl_index],
                    dtype=self.dtype,
                ),
                np.multiply(
                    self.eta,
                    self.weightUpdates[prev_index, actl_index],
                    dtype=self.dtype,
                ),
                dtype=self.dtype,
            )
            # Updating the bias-weights
            self.biasWeightUpdates[num, actl_index] = np.add(
                np.multiply(
                    self.learningRate,
                    self.biasGradients[num, actl_index],
                    dtype=self.dtype,
                ),
                np.multiply(
                    self.eta, self.biasWeightUpdates[num, actl_index], dtype=self.dtype,
                ),
            )
            
    def applyWeightChanges(self):
        """applyWeightChanges of the gradient correction to the layers.
        """
        for num, layer in enumerate(self.networkLayers[:-1]):
            prev_index = np.arange(
                layer["start_node"], layer["end_node"] + 1, dtype=np.int
            )  # similiar to i
            actl_index = np.arange(
                self.networkLayers[num + 1]["start_node"],
                self.networkLayers[num + 1]["end_node"] + 1,
                dtype=np.int,
            )  # similiar to j
            self.network.updateWeight(
                prev_index, actl_index, self.weightUpdates[prev_index, actl_index]
            )
            self.network.updateBiasWeight(
                num, actl_index, self.biasWeightUpdates[num, actl_index]
            )
        
        
    def calculateNetworkError(self, trainingSet):
        """
        calculateNetworkError [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        trainingSet : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        idealOutputs = trainingSet[
            -1 * self.networkLayers[len(self.networkLayers) - 1]["num_nodes"]
        ]
        startNode = self.networkLayers[len(self.networkLayers) - 1]["start_node"]
        endNode = self.networkLayers[len(self.networkLayers) - 1]["end_node"]
        numNodes = self.networkLayers[len(self.networkLayers) - 1]["num_nodes"]
        j = 0
        sum = 0
        for i in range(startNode, endNode + 1):
            if isinstance(idealOutputs, list):
                error = idealOutputs[j] - self.network.getValueEntry(i)
            else:
                error = idealOutputs - self.network.getValueEntry(i)
            sum += error * error
            j = j + 1
        globalError = (1 / numNodes) * sum
        return globalError
