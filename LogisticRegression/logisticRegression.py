import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from activation import *
from scaler import *
from optimizer import *

class LogisticRegression:
    """
    Logistic Regression Classifier.

    This class implements logistic regression for binary classification tasks. It includes options for different
    activation functions, scaling techniques, and optimization algorithms.

    Parameters:
    -----------
    x_train : np.ndarray
        Training data (features).
    y_train : np.ndarray
        Training data (labels).
    classifierTreshhold : float, optional (default=0.5)
        Threshold for classification (default: 0.5).
    costfunctionName : str, optional (default='sigmoid')
        Name of the cost function/activation function to use ('sigmoid', 'relu').
    scalerName : str, optional (default='minmax')
        Name of the scaler to use ('minmax', 'stdZ').
    optimizerName : str, optional (default='gd')
        Name of the optimizer to use ('gd', 'adam').
    ifDetail : str, optional (default='true')
        Whether to log the training details.

    Methods:
    --------
    settings() -> None:
        Set default training parameters such as convergence gap and learning rate details.

    test(x_test: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
        Test the trained model on the test set and return performance metrics like precision, recall, F1 score, etc.

    predict(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Predict output for new input data based on the learned weights and bias.

    trainModel() -> None:
        Train the logistic regression model using the specified optimizer.

    optimizer() -> None:
        Perform the optimization to minimize the cost function by updating weights and bias using the chosen algorithm.

    calculateCost(weight: np.ndarray, bias: float) -> float:
        Calculate the cost using the chosen cost function.
    """
    
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, classifierTreshhold: float = 0.5, costfunctionName: str = 'sigmoid', scalerName: str = 'minmax', optimizerName: str = 'gd', ifDetail: str = 'true'):
        self.x_train = x_train
        self.y_train = y_train
        self.classifierTreshhold = classifierTreshhold

        # Cost function and scaler initialization
        self.cost = CostFunction(costfunctionName).cost
        self.scaler = ScaleFunction(scalerName).scaler

        # Scale the training data
        self.scaler.scaleTrain(x_train)

        # Set the optimizer engine and weight initialization
        self.m, self.n = x_train.shape
        self.optimzerEngin = OptimizerEngine(optimizerName, self.n).optimizer

        self.weight = np.random.randn(self.n) * 0.01  # Initialize weights randomly
        self.bias = 0  # Initialize bias to zero
        
        self.ifDetail = ifDetail  # Detail flag for logging
        self.settings()  # Set default training parameters


    def settings(self) -> None:
        """
        Set default parameters for training such as the convergence gap and iteration details.
        """
        self.gap = 0.000001  # Minimum weight change for stopping
        self.epsilon = 0.000001  # Minimum cost change for stopping
        self.detailIteration = 500  # Interval for detailed logging


    def test(self, x_test: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
        """
        Test the trained model on the test set. Calculate and return confusion matrix and performance metrics.

        Parameters:
        -----------
        x_test : np.ndarray
            Test data (features).
        y_test : np.ndarray
            Actual labels for the test data.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]
            - Predicted output probabilities.
            - Binary predictions.
            - Confusion matrix as a pandas DataFrame (TP, FP, TN, FN).
            - Dictionary containing Precision, Recall, F1 Score, and Accuracy.
        """
        
        # Scale the test data
        x_test = self.scaler.scaleNew(x_test)

        # Compute predictions
        z_test = np.dot(x_test, self.weight) + self.bias
        y_pred_prob = self.cost.gFunction(z_test)  # Predicted probabilities
        y_pred = y_pred_prob > self.classifierTreshhold  # Binary predictions

        # Compute confusion matrix components
        TP = np.sum((y_pred == 1) & (y_test == 1))  # True Positives
        FP = np.sum((y_pred == 1) & (y_test == 0))  # False Positives
        TN = np.sum((y_pred == 0) & (y_test == 0))  # True Negatives
        FN = np.sum((y_pred == 0) & (y_test == 1))  # False Negatives

        # Create a confusion matrix as a table
        confusion_matrix = pd.DataFrame({
            "Predicted Positive": [TP, FP],
            "Predicted Negative": [FN, TN]
        }, index=["Actual Positive", "Actual Negative"])

        # Calculate performance metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        # Print confusion matrix and metrics
        print("\nConfusion Matrix:")
        print(confusion_matrix)
        print(f"\nPrecision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        return y_pred_prob, y_pred, confusion_matrix, {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
            "Accuracy": accuracy
        }

    
    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the output for new input data based on the learned weights and bias.

        Parameters:
        -----------
        x_test : np.ndarray
            Test data (features).

        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            - Predicted output probabilities.
            - Binary predictions.
        """
        
        # Scale the test data
        x_test = self.scaler.scaleNew(x_test)

        # Compute predictions
        z_test = np.dot(x_test, self.weight) + self.bias
        y_pred_prob = self.cost.gFunction(z_test)  # Predicted probabilities
        y_pred = y_pred_prob > self.classifierTreshhold  # Binary predictions

        return y_pred_prob, y_pred
    

    def trainModel(self) -> None:
        """
        Train the model using the selected optimizer and compute the cost.

        Returns:
        --------
        None
        """
        self.optimizer()  # Perform optimization (e.g., gradient descent)
        self.calculateCost(self.weight, self.bias)  # Compute the cost function


    def optimizer(self) -> None:
        """
        Perform the optimization (gradient descent or Adam) to update weights and bias.
        The optimization adjusts parameters iteratively to minimize the cost function.

        Returns:
        --------
        None
        """
        
        tmpB = self.bias  # Temporary variable for bias
        tmpW = self.weight  # Temporary variable for weights
        iteration = 0
        oldCost = self.calculateCost(tmpW, tmpB)  # Initial cost
        newCost = oldCost
        
        # If logging is enabled, store cost history for visualization
        if self.ifDetail:
            cost_history = []  # List to store cost function values per iteration

        # Gradient descent loop
        while True:
            if self.ifDetail:
                cost_history.append(newCost)  # Append current cost to history for visualization

            iteration += 1
            djdw, djdb = self.cost.djdw_db(self.x_train, self.y_train, self.weight, self.bias)
            tmpW, tmpB = self.optimzerEngin.iteration(self.weight, self.bias, djdw, djdb)

            oldCost = newCost  # Store previous cost
            newCost = self.calculateCost(tmpW, tmpB)  # Recompute cost with updated parameters

            # Convergence criteria: stop if weights/biases or cost changes are below thresholds
            maxGap = np.max(np.abs(tmpW - self.weight))  # Max change in weights
            if abs(tmpB - self.bias) < self.gap and maxGap < self.gap:
                break  # Stop if changes are below the threshold
            if abs(newCost - oldCost) < self.epsilon:
                break  # Stop if cost changes are below the epsilon threshold

            # Log progress (if logging is enabled)
            if self.ifDetail and iteration % self.detailIteration == 0:
                print(f"Iteration {iteration}, Cost: {newCost:.3f}, MaxWGap: {maxGap:.4f}, B: {tmpB: .4f}")
            
            # Update the model's weights and bias
            self.weight = tmpW
            self.bias = tmpB

        # Plot the cost history if logging is enabled
        if self.ifDetail:
            plt.plot(range(len(cost_history)), cost_history)  # Plot cost over iterations
            plt.show()


    def calculateCost(self, weight: np.ndarray, bias: float) -> float:
        """
        Compute the cost using the current weights and bias.

        Parameters:
        -----------
        weight : np.ndarray
            The weights of the model.
        bias : float
            The bias term of the model.

        Returns:
        --------
        float
            The calculated cost.
        """
        return self.cost.calculateCost(self.x_train, self.y_train, weight, bias)


class CostFunction:
    """
    Handles different cost functions (sigmoid, ReLU) for logistic regression.

    Parameters:
    -----------
    costName : str
        The name of the cost function to use.
    """
    def __init__(self, costName: str):
        self.costFunction = {
            "sigmoid": Sigmoid,
            'reLU': ReLU
        }
        self.cost = self.costFunction[costName]()


class ScaleFunction:
    """
    Handles different scaling functions (MinMax, Standard Z-Score).

    Parameters:
    -----------
    scaleName : str
        The name of the scaling function to use.
    """
    def __init__(self, scaleName: str):
        self.scaleFunction = {
            "minmax": MinMax,
            'stdZ': StndZ
        }
        self.scaler = self.scaleFunction[scaleName]()


class OptimizerEngine:
    """
    Handles different optimization techniques for logistic regression.

    Parameters:
    -----------
    optimizerName : str
        The name of the optimizer to use ('gd', 'adam').
    mFeature : int
        Number of features in the dataset.
    """
    def __init__(self, optimizerName: str, nFeature: int):
        self.optimizerEngine = {
            "gd": GradientDescent,
            'adam': Adam
        }
        self.optimizer = self.optimizerEngine[optimizerName](nFeature)
