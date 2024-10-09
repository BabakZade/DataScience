import numpy as np

class Scale:
    """
    Base class for scaling features. Provides a framework for scaling the training data and new input data.
    It defines the basic attributes for scaling such as min, max, mean, and standard deviation of the dataset.

    Attributes:
    -----------
    x_min : float
        The minimum value for scaling (default set to -3).
    x_max : float
        The maximum value for scaling (default set to 3).
    x_std : float
        The standard deviation for scaling (default set to 1).
    x_mean : float
        The mean value for scaling (default set to 0).

    Methods:
    --------
    scaleTrain(x_train: np.ndarray) -> None:
        Placeholder method for scaling the training data. To be implemented in derived classes.
    
    scaleNew(x_new: np.ndarray) -> None:
        Placeholder method for scaling new input data. To be implemented in derived classes.
    """

    def __init__(self):
        """
        Initializes the base scaling class with default values for min, max, mean, and standard deviation.
        """
        self.x_min = -3
        self.x_max = 3
        self.x_std = 1
        self.x_mean = 0

    def scaleTrain(self, x_train: np.ndarray) -> None:
        """
        Placeholder method to be implemented in derived classes for scaling the training data.

        Parameters:
        -----------
        x_train : np.ndarray
            The training data to be scaled.
        """
        pass

    def scaleNew(self, x_new: np.ndarray) -> None:
        """
        Placeholder method to be implemented in derived classes for scaling new data.

        Parameters:
        -----------
        x_new : np.ndarray
            The new input data to be scaled.
        """
        pass


class MinMax(Scale):
    """
    Implements MinMax scaling, which scales features to a specified range (default: [0, 1]).
    
    Attributes:
    -----------
    minX : np.ndarray
        The minimum values of the features in the training data.
    maxX : np.ndarray
        The maximum values of the features in the training data.

    Methods:
    --------
    scaleTrain(x_train: np.ndarray) -> np.ndarray:
        Scales the training data based on the min and max values in the dataset.

    scaleNew(x_new: np.ndarray) -> np.ndarray:
        Scales new input data using the same min and max values used during training.
    """

    def __init__(self):
        """
        Initializes the MinMax scaling class and inherits default settings from the Scale class.
        """
        super().__init__()

    def scaleTrain(self, x_train: np.ndarray) -> np.ndarray:
        """
        Scales the training data to a specified range [0, 1] using the MinMax scaling method.

        Parameters:
        -----------
        x_train : np.ndarray
            The training data to be scaled.

        Returns:
        --------
        np.ndarray
            The scaled training data.
        """
        self.minX = np.min(x_train, axis=0)
        self.maxX = np.max(x_train, axis=0)
        x_train = (x_train - self.x_min) / (self.x_max - self.x_min)  # Scaled to range [0, 1]
        return x_train

    def scaleNew(self, x_new: np.ndarray) -> np.ndarray:
        """
        Scales new input data to a range [0, 1] based on the scaling factors learned from the training data.

        Parameters:
        -----------
        x_new : np.ndarray
            The new input data to be scaled.

        Returns:
        --------
        np.ndarray
            The scaled new input data.
        """
        return (x_new - self.x_min) / (self.x_max - self.x_min)


class StndZ(Scale):
    """
    Implements Z-score standardization, which scales features based on mean and standard deviation.
    
    Attributes:
    -----------
    meanX : np.ndarray
        The mean values of the features in the training data.
    stdX : np.ndarray
        The standard deviation of the features in the training data.

    Methods:
    --------
    scaleTrain(x_train: np.ndarray) -> np.ndarray:
        Standardizes the training data by removing the mean and scaling to unit variance.

    scaleNew(x_new: np.ndarray) -> np.ndarray:
        Standardizes new input data using the same mean and standard deviation used during training.
    """

    def __init__(self):
        """
        Initializes the StndZ class and inherits default settings from the Scale class.
        """
        super().__init__()

    def scaleTrain(self, x_train: np.ndarray) -> np.ndarray:
        """
        Standardizes the training data using Z-score normalization (mean = 0, std = 1).

        Parameters:
        -----------
        x_train : np.ndarray
            The training data to be standardized.

        Returns:
        --------
        np.ndarray
            The standardized training data.
        """
        self.meanX = np.mean(x_train, axis=0)
        self.stdX = np.std(x_train, axis=0)
        x_train = (x_train - self.x_mean) / self.x_std  # Standardize to mean 0 and unit variance
        return x_train

    def scaleNew(self, x_new: np.ndarray) -> np.ndarray:
        """
        Standardizes new input data using the mean and standard deviation from the training data.

        Parameters:
        -----------
        x_new : np.ndarray
            The new input data to be standardized.

        Returns:
        --------
        np.ndarray
            The standardized new input data.
        """
        return (x_new - self.x_mean) / self.x_std
