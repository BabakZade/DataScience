import numpy as np

class Optimizer:
    """
    Base class for optimization algorithms. This class provides a general structure for setting up 
    optimization configurations and implementing iterative updates of weights and biases.

    Attributes:
    -----------
    alpha : float
        The learning rate used for updating parameters.

    Methods:
    --------
    generalSettings() -> None:
        Sets general configurations for the optimizer, such as the learning rate.
    
    settings(nFeature: int) -> None:
        Abstract method to be implemented in derived classes to set optimizer-specific configurations.
    
    iteration(oldW: np.ndarray, oldB: float, djdw: np.ndarray, djdb: float) -> tuple[np.ndarray, float]:
        Abstract method to be implemented in derived classes to update the model's weights and bias.
    """
    
    def __init__(self, nFeature: int):
        """
        Initializes the optimizer with the general and specific settings for the optimization algorithm.
        
        Parameters:
        -----------
        nFeature : int
            The number of features in the input data, used for setting up optimization parameters.
        """
        self.generalSettings()       
        self.settings(nFeature)
        pass

    def generalSettings(self) -> None:
        """
        Sets general configurations for the optimizer. The primary configuration is the learning rate (alpha).
        """
        self.alpha = 0.00001  # General learning rate

    def settings(self, nFeature: int) -> None:
        """
        Placeholder for derived classes to define optimizer-specific settings.

        Parameters:
        -----------
        nFeature : int
            The number of features in the dataset, which can be used for setting optimizer parameters.
        """
        pass

    def iteration(self, oldW: np.ndarray, oldB: float, djdw: np.ndarray, djdb: float) -> tuple[np.ndarray, float]:
        """
        Placeholder for derived classes to define how weights and bias should be updated in each iteration.
        
        Parameters:
        -----------
        oldW : np.ndarray
            The current weights of the model.
        oldB : float
            The current bias of the model.
        djdw : np.ndarray
            The gradient of the cost function with respect to the weights.
        djdb : float
            The gradient of the cost function with respect to the bias.

        Returns:
        --------
        tuple[np.ndarray, float]
            Updated weights and bias after performing a single iteration.
        """
        pass


class GradientDescent(Optimizer):
    """
    Implements the basic Gradient Descent optimization algorithm, a first-order iterative optimization method.

    Methods:
    --------
    iteration(oldW: np.ndarray, oldB: float, djdw: np.ndarray, djdb: float) -> tuple[np.ndarray, float]:
        Updates the weights and bias using the gradient descent formula.
    """
    
    def __init__(self, nFeature: int):
        """
        Initializes the Gradient Descent optimizer by calling the base optimizer's initialization method.

        Parameters:
        -----------
        nFeature : int
            The number of features in the input data, used for setting up optimization parameters.
        """
        super().__init__(nFeature)

    def iteration(self, oldW: np.ndarray, oldB: float, djdw: np.ndarray, djdb: float) -> tuple[np.ndarray, float]:
        """
        Updates the weights and bias using the gradient descent update rule:
        
        newW = oldW - alpha * djdw
        newB = oldB - alpha * djdb
        
        Parameters:
        -----------
        oldW : np.ndarray
            The current weights of the model.
        oldB : float
            The current bias of the model.
        djdw : np.ndarray
            The gradient of the cost function with respect to the weights.
        djdb : float
            The gradient of the cost function with respect to the bias.

        Returns:
        --------
        tuple[np.ndarray, float]
            Updated weights and bias after performing a single iteration.
        """
        newW = oldW - self.alpha * djdw  # Update weights
        newB = oldB - self.alpha * djdb  # Update bias
        return newW, newB


class Adam(Optimizer):
    """
    Implements the Adam (Adaptive Moment Estimation) optimization algorithm. Adam is an extension of 
    stochastic gradient descent that maintains a moving average of the gradients and squared gradients.

    Attributes:
    -----------
    beta : float
        The decay rate for the moving average of gradients (momentum).
    m_tw : np.ndarray
        The moving average of gradients for the weights.
    m_tb : float
        The moving average of gradients for the bias.

    Methods:
    --------
    settings(nFeature: int) -> None:
        Sets optimizer-specific settings like beta and initializes moving averages for weights and bias.
    
    iteration(oldW: np.ndarray, oldB: float, djdw: np.ndarray, djdb: float) -> tuple[np.ndarray, float]:
        Updates the weights and bias using the Adam optimization rule.
    """

    def __init__(self, nFeature: int):
        """
        Initializes the Adam optimizer by calling the base optimizer's initialization method.

        Parameters:
        -----------
        nFeature : int
            The number of features in the input data, used for setting up optimization parameters.
        """
        super().__init__(nFeature)

    def settings(self, nFeature: int) -> None:
        """
        Sets optimizer-specific parameters for Adam, such as the beta coefficient for momentum
        and initializes moving averages for the gradients of weights and bias.

        Parameters:
        -----------
        nFeature : int
            The number of features in the input data, used to initialize moving averages for gradients.
        """
        self.beta = 0.9  # Decay rate for moving averages (momentum)
        self.m_tw = np.zeros(nFeature)  # Initialize moving average for weights
        self.m_tb = 0  # Initialize moving average for bias

    def iteration(self, oldW: np.ndarray, oldB: float, djdw: np.ndarray, djdb: float) -> tuple[np.ndarray, float]:
        """
        Updates the weights and bias using the Adam optimization algorithm:
        
        m_tw = beta * m_tw + (1 - beta) * djdw
        m_tb = beta * m_tb + (1 - beta) * djdb
        
        newW = oldW - alpha * m_tw
        newB = oldB - alpha * m_tb
        
        Parameters:
        -----------
        oldW : np.ndarray
            The current weights of the model.
        oldB : float
            The current bias of the model.
        djdw : np.ndarray
            The gradient of the cost function with respect to the weights.
        djdb : float
            The gradient of the cost function with respect to the bias.

        Returns:
        --------
        tuple[np.ndarray, float]
            Updated weights and bias after performing a single iteration.
        """
        # Update moving averages of the gradients
        self.m_tw = self.beta * self.m_tw + (1 - self.beta) * djdw
        self.m_tb = self.beta * self.m_tb + (1 - self.beta) * djdb
        
        # Update weights and bias
        newW = oldW - self.alpha * self.m_tw
        newB = oldB - self.alpha * self.m_tb

        return newW, newB
