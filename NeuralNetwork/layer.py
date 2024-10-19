import numpy as np
from neuron import Neuron
from activation import *
from optimizer import *

class Layer:

    def __init__(self, units=1, activationName='Sigmoid', randomActivation=True, optimizerName='Adam'):
        """
        Initialize the Layer class.
        
        Parameters:
        -----------
        units : int
            Number of neurons (units) in the layer.
        activationName : str
            The name of the activation function to use for each neuron.
        randomActivation : bool
            If True, randomly selects an activation function for each neuron.
        optimizerName : str
            The name of the optimizer to use for updating weights.
        """
        self.units = units  # Number of neurons (units) in the layer
        self.layer = np.empty(units, Neuron)  # Initialize an array of neurons
        self.activationName = activationName  # Activation function name
        self.randomActivation = randomActivation  # Whether to randomize the activation function
        self.m = 0  # Number of training examples
        self.n = 0  # Number of input features
        self.optimizer = OptimizerSelector(optimizerName=optimizerName).optimizer  # Optimizer for weight updates

        # Initialize each neuron in the layer
        for i in range(units):            
            if randomActivation:
                # Randomly select an activation function from subclasses of the Activation class
                print(Activation.__subclasses__())
                dictClass = Activation.__subclasses__()
                activationCl = np.random.choice(dictClass)
                activationName = activationCl.__name__
            self.layer[i] = Neuron(activationName)  # Assign the selected activation function to the neuron

    def initialize(self, a_in):
        """
        Initialize the output, weights, biases, and other internal variables for the layer.
        
        Parameters:
        -----------
        a_in : np.ndarray
            Input data for the layer (activations from the previous layer or input data for the first layer).
        """
        # Initialize output and gradients based on input shape
        if self.m != a_in.shape[0]:
            self.m = a_in.shape[0]  # Number of training examples
            self.a_out = np.zeros(shape=(self.m, self.units), dtype=float)  # Output activations for this layer
            self.da_l = np.zeros(shape=(self.m, self.units), dtype=float)  # Gradient with respect to activations
            self.z_out = np.zeros(shape=(self.m, self.units), dtype=float)  # Linear combination z for each neuron
            self.z_prime = np.zeros(shape=(self.m, self.units), dtype=float)  # Derivative of the activation function
            self.dz = np.zeros(shape=(self.m, self.units), dtype=float)  # Gradient with respect to z

        # Initialize weights and biases if the number of input features has changed
        if self.n != a_in.shape[1]:      
            self.n = a_in.shape[1]  # Number of input features
            self.weights = np.zeros(shape=(self.units, self.n), dtype=float)  # Weights for each neuron
            self.bias = np.zeros(shape=(self.units), dtype=float)  # Biases for each neuron
            self.dw = np.zeros(shape=(self.units, self.n), dtype=float)  # Gradient with respect to weights
            self.db = np.zeros(shape=(self.units), dtype=float)  # Gradient with respect to biases

    def dense(self, a_in):
        """
        Perform a forward pass through the layer (dense layer).
        
        Parameters:
        -----------
        a_in : np.ndarray
            Input data or activations from the previous layer.
        """
        self.a_in = a_in  # Store input activations
        for i in range(self.units):
            # Compute z (linear combination) and a (activated output) for each neuron
            self.z_out[:, i], self.a_out[:, i] = self.layer[i].activate(self.a_in, self.weights[i, :], self.bias[i])

    def get_a_out(self, a_in):
        """
        Get the output activations for the input.
        
        Parameters:
        -----------
        a_in : np.ndarray
            Input data or activations from the previous layer.
        
        Returns:
        --------
        a_out : np.ndarray
            Activated output from the layer.
        """
        a_out = np.zeros(shape=(a_in.shape[0], self.units), dtype=float)
        for i in range(self.units):
            # Compute activated output for each neuron
            _, a_out[:, i] = self.layer[i].activate(a_in, self.weights[i, :], self.bias[i])
        return a_out

    def getWeights(self):
        """
        Get the current weights and biases of the layer.
        
        Returns:
        --------
        tuple : (weights, biases)
            Weights and biases of the layer.
        """
        return self.weights, self.bias

    def display(self):
        """
        Display the weights and biases of the layer.
        """
        print("======================================")
        print("Layer Weights: ", self.weights)
        print("Layer Bias: ", self.bias)
        print("======================================")

    def setWeights(self, weights=None, bias=None):
        """
        Set the weights and biases of the layer. If not provided, initialize them using the activation-specific method.
        
        Parameters:
        -----------
        weights : np.ndarray, optional
            Weights to set for the layer.
        bias : np.ndarray, optional
            Biases to set for the layer.
        """
        if weights is not None:
            self.weights = weights  # Set provided weights
            self.bias = bias  # Set provided biases
        else:
            # Initialize weights and biases using the activation's specific weight initialization method
            for i in range(len(self.layer)):
                self.weights[i, :] = self.layer[i].activation.weightinitial(self.n)  # Initialize weights
                self.bias[i] = self.layer[i].activation.weightinitial(1)  # Initialize biases

    def updateWeight(self):
        """
        Update the weights and biases using the optimizer.
        """
        self.weights, self.bias = self.optimizer.update(self.weights, self.bias, self.dw, self.db)

    def set_backward(self, da):
        """
        Backward pass: Calculate the gradients with respect to weights, biases, and activations.
        
        Parameters:
        -----------
        da : np.ndarray
            Gradient of the loss with respect to the activations (output) of the layer.
        """
        for n in range(self.units):
            # Compute the gradients for each neuron
            self.dw[n, :], self.db[n], self.dz[:, n] = self.layer[n].gradian(da[:, n])
