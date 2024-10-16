import numpy as np
import math

class ActivationSelector:
    """
    Selects and provides different activation functions.
    """
    def __init__(self, activation_name: str):
        self.activations = {
            "Sigmoid": Sigmoid,
            "ReLU": ReLU,
            "SoftMax": SoftMax,
            "Tanh": Tanh
        }
        self.activation = self.activations[activation_name]()


class Activation:
    """
    Base class for activation functions.
    Provides a default interface for activation and weight initialization.
    """
    def __init__(self):
        self.name = "Activation"
        self.math = "NotYetField"

    def gFunction(self, z: np.ndarray):
        pass

    def computeGZ(self, x_train: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
        z = np.dot(x_train, w) + b
        gz = np.vectorize(self.gFunction)(z).T
        return gz

    def djdw_db(self, x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float, lambdaReg: float) -> tuple[np.ndarray, float]:
        m, n = x_train.shape
        y_hat = self.computeGZ(x_train, w, b)
        djdw = 1 / m * (np.dot((y_hat - y_train), x_train) + lambdaReg * w)
        djdb = np.mean(y_hat - y_train)
        return djdw, djdb

    # Xavier Initialization
    def xavier_init(self, fan_in, fan_out=1):
        stddev = math.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, stddev, size=(fan_out, fan_in))

    # He Initialization
    def he_init(self, fan_in):
        stddev = math.sqrt(2 / fan_in)
        return np.random.normal(0, stddev, size=fan_in)


class Sigmoid(Activation):
    """
    Implements the Sigmoid activation function and uses Xavier initialization.
    """
    def __init__(self):
        self.name = "Sigmoid"
        self.math = "1/(1 + e^-z)"
        self.settings()

    def settings(self):
        self.eps = 1e-8
        self.zClipper = 500

    def gFunction(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -self.zClipper, self.zClipper)
        return 1 / (1 + np.exp(-z))

    def calculateCost(self, x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float, lambdaReg: float) -> float:
        m, n = x_train.shape
        y_hat = self.computeGZ(x_train, w, b)
        y_hat = np.clip(y_hat, self.eps, 1 - self.eps)
        cross_entropy_loss = -np.mean(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat))
        reg_term = lambdaReg / 2 / m * np.sum(w ** 2)
        return cross_entropy_loss + reg_term

    def weightinitial(self, fan_in, fan_out = 1):
        """
        Initialize weights using Xavier initialization for Sigmoid activation.
        """
        return self.xavier_init(fan_in, fan_out)

    def activation_prime(self, z):
        return self.gFunction(z) * (1 - self.gFunction(z))


class ReLU(Activation):
    """
    Implements the ReLU activation function and uses He initialization.
    """
    def __init__(self):
        self.name = "ReLU"
        self.math = "max(0, z)"

    def gFunction(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def weightinitial(self, fan_in, fan_out = 1):
        """
        Initialize weights using He initialization for ReLU activation.
        """
        return self.he_init(fan_in)

    def activation_prime(self, z):
        return np.where(z > 0, 1, 0)


class SoftMax(Activation):
    """
    Implements the SoftMax activation function and can use Xavier initialization.
    """
    def __init__(self):
        self.name = "SoftMax"
        self.math = "e^z / sum(e^z)"

    def gFunction(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))  # Subtract max for numerical stability
        return exp_z / exp_z.sum(axis=-1, keepdims=True)

    def weightinitial(self, fan_in, fan_out = 1):
        """
        Initialize weights using Xavier initialization for SoftMax activation.
        """
        return self.xavier_init(fan_in, fan_out)

    def activation_prime(self, z: np.ndarray) -> np.ndarray:
        s = self.gFunction(z).reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        return jacobian


class Tanh(Activation):
    """
    Implements the Tanh (Hyperbolic Tangent) activation function.
    Uses Xavier initialization since Tanh is similar to Sigmoid in nature.
    """
    def __init__(self):
        self.name = "Tanh"
        self.math = "(e^z - e^-z) / (e^z + e^-z)"

    def gFunction(self, z: np.ndarray) -> np.ndarray:
        """
        Applies the Tanh activation function: (e^z - e^-z) / (e^z + e^-z).
        
        Parameters:
        ----------- 
        z : np.ndarray
            The linear combination of weights and inputs.
        
        Returns:
        --------
        np.ndarray
            The Tanh-transformed output.
        """
        return np.tanh(z)

    def weightinitial(self, fan_in, fan_out = 1):
        """
        Initialize weights using Xavier initialization for Tanh activation.
        """
        return self.xavier_init(fan_in, fan_out)

    def activation_prime(self, z: np.ndarray) -> np.ndarray:
        """
        Derivative of the Tanh activation function.
        
        Returns:
        --------
        np.ndarray
            The derivative of the Tanh activation function.
        """
        return 1 - np.tanh(z) ** 2
