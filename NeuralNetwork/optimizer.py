import numpy as np

class OptimizerSelector:
    """
    Handles different optimization techniques for training neural networks or machine learning models.
    
    Parameters:
    -----------
    optimizerName : str
        The name of the optimizer to use ('GradientDescent', 'Adam', 'RMSprop').
    """
    def __init__(self, optimizerName: str):
        # Map optimizer names to their respective classes.
        self.optimizerEngine = {
            "GradientDescent": GradientDescent,
            'Adam': Adam,
            'RMSprop': RMSprop
        }
        # Select the optimizer based on the given name.
        self.optimizer = self.optimizerEngine[optimizerName]()

class Optimizer:
    """
    Base class for optimization algorithms.
    
    Attributes:
    -----------
    alpha : float
        Learning rate used for updating parameters.
        
    Methods:
    --------
    generalSettings() -> None:
        Sets general configurations for the optimizer (learning rate).
    update(oldW, oldB, djdw, djdb) -> tuple:
        Placeholder method to be implemented in derived classes, for updating weights and bias.
    """
    
    def __init__(self):
        """
        Initializes the optimizer by setting general and specific settings.
        """
        self.generalSettings()  # General settings (e.g., learning rate)
        self.settings()  # Specific settings for derived optimizers (if any)
        pass

    def generalSettings(self) -> None:
        """
        Sets general configurations for the optimizer, such as the learning rate.
        """
        self.alpha = 0.0001  # Default learning rate

    def settings(self) -> None:
        """
        Placeholder for optimizer-specific settings in derived classes.
        """
        pass

    def update(self, oldW: np.ndarray, oldB: float, djdw: np.ndarray, djdb: float) -> tuple[np.ndarray, float]:
        """
        Placeholder for updating weights and biases. Must be implemented by derived classes.
        
        Parameters:
        -----------
        oldW : np.ndarray
            Current weights of the model.
        oldB : float
            Current bias of the model.
        djdw : np.ndarray
            Gradient of the cost function with respect to the weights.
        djdb : float
            Gradient of the cost function with respect to the bias.

        Returns:
        --------
        tuple : (np.ndarray, float)
            Updated weights and bias.
        """
        pass


class GradientDescent:
    """
    Implements basic Gradient Descent optimization.
    
    Attributes:
    -----------
    learning_rate : float
        Learning rate for updating the weights and biases.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate  # Set learning rate

    def update(self, w, b, dw, db):
        """
        Updates weights and biases using Gradient Descent.
        
        Parameters:
        -----------
        w : np.ndarray
            Current weights.
        b : np.ndarray
            Current biases.
        dw : np.ndarray
            Gradients with respect to the weights.
        db : np.ndarray
            Gradients with respect to the biases.
        
        Returns:
        --------
        tuple : (np.ndarray, np.ndarray)
            Updated weights and biases.
        """
        # Update weights and biases based on gradients and learning rate.
        w_update = w - self.learning_rate * dw
        b_update = b - self.learning_rate * db

        return w_update, b_update


class Adam:
    """
    Implements Adam optimization algorithm.
    
    Attributes:
    -----------
    learning_rate : float
        Learning rate for updating the weights and biases.
    beta1 : float
        Exponential decay rate for the first moment (mean) estimates.
    beta2 : float
        Exponential decay rate for the second moment (variance) estimates.
    epsilon : float
        Small constant to prevent division by zero.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate  # Set learning rate
        self.beta1 = beta1  # First moment decay rate
        self.beta2 = beta2  # Second moment decay rate
        self.epsilon = epsilon  # Small constant to prevent division by zero
        
        # Initialize first and second moment estimates for weights and biases
        self.m_w, self.v_w = None, None  # First and second moment for weights
        self.m_b, self.v_b = None, None  # First and second moment for biases
        self.t = 0  # Time step (for bias correction)

    def update(self, w, b, dw, db):
        """
        Updates weights and biases using Adam optimization algorithm.
        
        Parameters:
        -----------
        w : np.ndarray
            Current weights.
        b : np.ndarray
            Current biases.
        dw : np.ndarray
            Gradients with respect to the weights.
        db : np.ndarray
            Gradients with respect to the biases.
        
        Returns:
        --------
        tuple : (np.ndarray, np.ndarray)
            Updated weights and biases.
        """
        # Initialize moment estimates if not already initialized
        if self.m_w is None:
            self.m_w = np.zeros_like(w)
            self.v_w = np.zeros_like(w)
        if self.m_b is None:
            self.m_b = np.zeros_like(b)
            self.v_b = np.zeros_like(b)

        # Increment the time step for bias correction
        self.t += 1

        # Update biased first moment estimate for weights and biases
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db

        # Update biased second moment estimate for weights and biases
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)

        # Compute bias-corrected first and second moment estimates for weights and biases
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

        # Update weights and biases using the Adam update rule
        w_update = w - self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        b_update = b - self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return w_update, b_update


class RMSprop:
    """
    Implements RMSprop optimization algorithm.
    
    Attributes:
    -----------
    learning_rate : float
        Learning rate for updating the weights and biases.
    beta : float
        Exponential decay rate for the moving average of squared gradients.
    epsilon : float
        Small constant to prevent division by zero.
    """
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate  # Set learning rate
        self.beta = beta  # Decay rate for the moving average of squared gradients
        self.epsilon = epsilon  # Small constant to prevent division by zero
        self.cache_w = None  # Cache for weights (squared gradients)
        self.cache_b = None  # Cache for biases (squared gradients)

    def update(self, w, b, dw, db):
        """
        Updates weights and biases using RMSprop optimization algorithm.
        
        Parameters:
        -----------
        w : np.ndarray
            Current weights.
        b : np.ndarray
            Current biases.
        dw : np.ndarray
            Gradients with respect to the weights.
        db : np.ndarray
            Gradients with respect to the biases.
        
        Returns:
        --------
        tuple : (np.ndarray, np.ndarray)
            Updated weights and biases.
        """
        # Initialize cache for squared gradients if not already initialized
        if self.cache_w is None:
            self.cache_w = np.zeros_like(w)
        if self.cache_b is None:
            self.cache_b = np.zeros_like(b)

        # Update the cache with the current squared gradients
        self.cache_w = self.beta * self.cache_w + (1 - self.beta) * dw**2
        self.cache_b = self.beta * self.cache_b + (1 - self.beta) * db**2

        # Update weights and biases using RMSprop update rule
        w_update = w - self.learning_rate * dw / (np.sqrt(self.cache_w) + self.epsilon)
        b_update = b - self.learning_rate * db / (np.sqrt(self.cache_b) + self.epsilon)

        return w_update, b_update
