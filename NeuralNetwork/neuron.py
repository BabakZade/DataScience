
from activation import *


class Neuron:

    def __init__(self, activationName):
        self.activation = ActivationSelector(activationName).activation 
       



    def activate(self, a_in, weights, bias):
        self.weights = weights
        self.bias = bias
        self.a_in = a_in
        self.z =  np.dot(a_in, weights) + bias
        self.gz = self.activation.gFunction(self.z)
        self.z_prime = self.activation.activation_prime(self.z)
        return self.z, self.gz

    def gradian(self, da):
        m, n = self.a_in.shape
        # Output layer: calculate dA directly from the loss
        self.da = da
        # Step 1: Compute dZ[l] = dA[l] * activation_prime(Z[l])
        if len(self.z_prime.shape) > 1: # for softmax prime
            self.dz = np.dot(self.z_prime, self.da) 
        else:
            self.dz = self.da * self.z_prime

        

        # Step 2: Compute dW[l] = (1/m) * dZ[l] * A[l-1]^T
        # Make sure to transpose A[l-1] for the correct dimensions
        self.dw = 1/ m * (np.dot( self.dz, self.a_in ))

        # Step 3: Compute db[l] = (1/m) * sum(dZ[l])
        # Sum over the batch axis, axis=1
        self.db = np.mean(self.dz)
        return self.dw, self.db, self.dz



