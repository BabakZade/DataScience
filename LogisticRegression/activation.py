import numpy as np

class Activation:
    """
    Base class for activation functions. Provides the framework for the forward pass (activation),
    cost function calculation, and gradient computations.

    Attributes:
    -----------
    name : str
        The name of the activation function.
    math : str
        A string representation of the mathematical expression of the activation function.

    Methods:
    --------
    gFunction(z: np.ndarray) -> np.ndarray:
        Computes the activation function (to be implemented by subclasses).
    
    computeGZ(x_train: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
        Computes the dot product of the weights and inputs, adds bias, and applies the activation function.
    
    djdw_db(x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float, lambdaReg: float) -> tuple[np.ndarray, float]:
        Computes the gradient of the cost function with respect to the weights and bias, including regularization.
    
    calculateCost(x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float, lambdaReg: float) -> float:
        Computes the cost function with regularization (to be implemented by subclasses).
    """
    
    def __init__(self):
        """
        Initializes the base activation function with a default name and math description.
        """
        self.name = "Activation"
        self.math = "NotYetField"

    def gFunction(self, z: np.ndarray):
        """
        Placeholder for the actual activation function, to be implemented in derived classes.

        Parameters:
        -----------
        z : np.ndarray
            The linear combination of weights and inputs (dot product) to be transformed by the activation function.
        
        Returns:
        --------
        np.ndarray
            The transformed output after applying the activation function.
        """
        pass
    
    def computeGZ(self, x_train: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
        """
        Computes the linear combination of input features, weights, and bias, then applies the activation function.

        Parameters:
        -----------
        x_train : np.ndarray
            The input data (features).
        w : np.ndarray
            The weights of the model.
        b : float
            The bias term of the model.

        Returns:
        --------
        np.ndarray
            The activated output of the model.
        """
        z = np.dot(x_train, w) + b
        gz = np.vectorize(self.gFunction)(z).T
        return gz

    def djdw_db(self, x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float, lambdaReg: float) -> tuple[np.ndarray, float]:
        """
        Computes the gradient of the cost function with respect to the weights (djdw) and bias (djdb).

        Includes regularization for the weights to reduce overfitting.

        Parameters:
        -----------
        x_train : np.ndarray
            The input data (features).
        y_train : np.ndarray
            The true labels for the training data.
        w : np.ndarray
            The weights of the model.
        b : float
            The bias term of the model.
        lambdaReg : float
            The regularization parameter (L2 regularization).

        Returns:
        --------
        tuple[np.ndarray, float]
            - djdw: The gradient of the cost function with respect to the weights.
            - djdb: The gradient of the cost function with respect to the bias.
        """
        m, n = x_train.shape
        y_had = self.computeGZ(x_train, w, b)
        djdw = 1 / m * (np.dot((y_had - y_train), x_train) + lambdaReg * w)  # Regularization term added to djdw
        djdb = np.mean(y_had - y_train)
        return djdw, djdb

    def calculateCost(self, x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float, lambdaReg: float) -> float:
        """
        Placeholder for the cost function, to be implemented in derived classes.

        Parameters:
        -----------
        x_train : np.ndarray
            The input data (features).
        y_train : np.ndarray
            The true labels for the training data.
        w : np.ndarray
            The weights of the model.
        b : float
            The bias term of the model.
        lambdaReg : float
            The regularization parameter (L2 regularization).

        Returns:
        --------
        float
            The computed cost with regularization (to be implemented in subclasses).
        """
        pass



class Sigmoid(Activation):
    """
    Implements the Sigmoid activation function and its corresponding cost function for logistic regression.

    Attributes:
    -----------
    eps : float
        A small epsilon value to prevent numerical errors in logarithms (avoiding log(0)).
    zClipper : float
        A large positive and negative value to clip z for preventing overflow in the exponential function.

    Methods:
    --------
    gFunction(z: np.ndarray) -> np.ndarray:
        Applies the Sigmoid activation function to the input z.
    
    calculateCost(x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float, lambdaReg: float) -> float:
        Computes the cost function using cross-entropy loss for logistic regression with L2 regularization.
    """

    def __init__(self):
        """
        Initializes the Sigmoid activation function with its mathematical expression and settings.
        """
        self.name = "Sigmoid"
        self.math = "1/(1 + e^-z)"
        self.settings()

    def settings(self) -> None:
        """
        Sets specific parameters for the Sigmoid function such as epsilon for numerical stability and z clipping.
        """
        self.eps = 1e-8  # A small epsilon value to prevent log(0)
        self.zClipper = 500  # Clip z to avoid overflow

    def gFunction(self, z: np.ndarray) -> np.ndarray:
        """
        Applies the Sigmoid activation function: 1 / (1 + exp(-z)).
        
        Parameters:
        -----------
        z : np.ndarray
            The linear combination of weights and inputs.
        
        Returns:
        --------
        np.ndarray
            The Sigmoid-transformed output.
        """
        z = np.clip(z, -self.zClipper, self.zClipper)  # Clip z to avoid overflow
        return 1 / (1 + np.exp(-z))

    def calculateCost(self, x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float, lambdaReg: float) -> float:
        """
        Computes the cross-entropy cost for logistic regression with the Sigmoid activation function and L2 regularization.
        
        Parameters:
        -----------
        x_train : np.ndarray
            The input data (features).
        y_train : np.ndarray
            The true labels for the training data.
        w : np.ndarray
            The weights of the model.
        b : float
            The bias term of the model.
        lambdaReg : float
            The regularization parameter (L2 regularization).

        Returns:
        --------
        float
            The computed cross-entropy cost with L2 regularization.
        """
        y_hat = self.computeGZ(x_train, w, b)
        y_hat = np.clip(y_hat, self.eps, 1 - self.eps)  # Clip predictions to avoid log(0)
        cross_entropy_loss = -np.mean(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat))
        reg_term = lambdaReg / 2 * np.mean(w ** 2)  # L2 regularization term
        return cross_entropy_loss + reg_term


class ReLU(Activation):
    """
    Implements the ReLU (Rectified Linear Unit) activation function.

    Methods:
    --------
    gFunction(z: np.ndarray) -> np.ndarray:
        Applies the ReLU activation function to the input z.
    """

    def __init__(self):
        """
        Initializes the ReLU activation function with its mathematical expression.
        """
        self.name = "ReLU"
        self.math = "max(0, z)"

    def gFunction(self, z: np.ndarray) -> np.ndarray:
        """
        Applies the ReLU activation function: max(0, z).
        
        Parameters:
        ----------- 
        z : np.ndarray
            The linear combination of weights and inputs.
        
        Returns:
        --------
        np.ndarray
            The ReLU-transformed output.
        """
        return np.maximum(0, z)
