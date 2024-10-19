from activation import *

class Neuron:
    def __init__(self, activationName):
        """
        Initializes the Neuron with the specified activation function.
        
        Parameters:
        -----------
        activationName : str
            Name of the activation function to be used by the neuron (e.g., 'Sigmoid', 'ReLU', 'SoftMax').
        """
        self.activation = ActivationSelector(activationName).activation  # Selects the activation function

    def activate(self, a_in, weights, bias):
        """
        Performs the forward pass: Computes the linear combination (z) and the activated output (g(z)).
        
        Parameters:
        -----------
        a_in : np.ndarray
            Input data or activations from the previous layer.
        weights : np.ndarray
            Weights of the neuron.
        bias : float
            Bias of the neuron.
        
        Returns:
        --------
        tuple : (z, g(z))
            z : np.ndarray
                Linear combination of inputs and weights.
            g(z) : np.ndarray
                Activated output using the selected activation function.
        """
        self.weights = weights  # Store the weights
        self.bias = bias  # Store the bias
        self.a_in = a_in  # Store the input activations

        # Compute z = a_in * weights + bias
        self.z = np.dot(a_in, weights) + bias

        # Compute the activated output using the activation function
        self.gz = self.activation.gFunction(self.z)

        # Compute the derivative of the activation function for backpropagation
        self.z_prime = self.activation.activation_prime(self.z)

        return self.z, self.gz

    def gradian(self, da):
        """
        Performs the backward pass: Computes the gradients with respect to weights, biases, and input activations.
        
        Parameters:
        -----------
        da : np.ndarray
            Gradient of the loss function with respect to the output activations of this neuron.
        
        Returns:
        --------
        tuple : (dw, db, dz)
            dw : np.ndarray
                Gradient of the loss function with respect to the weights.
            db : float
                Gradient of the loss function with respect to the bias.
            dz : np.ndarray
                Gradient of the loss function with respect to the linear combination (z).
        """
        m, n = self.a_in.shape  # m = number of training examples, n = number of input features
        self.da = da  # Store the gradient from the next layer or loss function

        # Step 1: Compute dZ[l] = dA[l] * activation_prime(Z[l])
        # Handle SoftMax separately since its derivative is a Jacobian matrix
        if len(self.z_prime.shape) > 1:  # If z_prime is a matrix (SoftMax case)
            self.dz = np.dot(self.z_prime, self.da)  # Use matrix multiplication for SoftMax
        else:
            self.dz = self.da * self.z_prime  # For other activation functions

        # Step 2: Compute dW[l] = (1/m) * dZ[l] * A[l-1]^T
        # Compute the gradient with respect to the weights (dw)
        self.dw = 1 / m * np.dot(self.dz, self.a_in)

        # Step 3: Compute db[l] = (1/m) * sum(dZ[l])
        # Compute the gradient with respect to the bias (db)
        self.db = np.mean(self.dz)  # Average over the batch

        return self.dw, self.db, self.dz
