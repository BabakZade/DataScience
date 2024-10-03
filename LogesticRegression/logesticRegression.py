
import numpy as np
import pandas as pd

class LogesticRegression:
    def __init__(self, x_train, y_train, classifierTreshhold, costfunctionName, ifDetail):
        self.x_train = x_train
        self.y_train = y_train
        self.classifierTreshhold = classifierTreshhold

        self.cost = CostFunction(costfunctionName).cost

        self.x_min = np.min(x_train, axis=0)  # Minimum values of features for scaling
        self.x_max = np.max(x_train, axis=0)  # Maximum values of features for scaling
        self.x_mean = np.mean(x_train, axis=0)  # Maximum values of features for scaling
        self.scale_X()  # Scale the features

        totalNumberFeature = len(np.shape(x_train))
        self.weight = np.zeros(totalNumberFeature)  # Initialize weights to zero
        self.bias = 0  # Initialize bias to zero

        
        self.ifDetail = ifDetail  # Detail flag
        self.settings()  # Set default training parameters



    def scale_X(self):
        """
        Normalize the input features to the range [0, 1] for better gradient descent performance.
        """
        self.x_train = 1 / (self.x_max - self.x_mean) * (self.x_train - self.x_min)
        pass
    
    def tuneLearningRate(self, iteration, oldCost, newCost):
        """
        Adjust the learning rate during training. Reduce learning rate if the cost increases,
        and increase it periodically to escape local minima.
        """
        if oldCost < newCost:
            self.alpha /= self.alphaDecreasingStep  # Reduce learning rate if cost increased
        if iteration % self.alphaSpanChange == 0:
            self.alpha *= self.alphIncreaingStep  # Increase learning rate periodically

    def settings(self) -> None:
        """
        Set default parameters for gradient descent like learning rate, convergence gap, and iteration details.
        """
        self.alpha = 0.01  # Initial learning rate
        self.gap = 0.000001  # Minimum weight change for stopping
        self.epsilon = 0.00000001  # Minimum cost change for stopping
        self.detailIteration = 500  # Interval for detailed logging
        self.alphaSpanChange = 500  # Iteration frequency to increase learning rate
        self.alphIncreaingStep = 2  # Learning rate increase multiplier
        self.alphaDecreasingStep = 5  # Learning rate decrease divisor

    def yHat(self, x_new):
        """
        Predict output for new input data based on the learned weights and bias.

        Parameters:
        -----------
        x_new : np.ndarray
            New input data.

        Returns:
        --------
        np.ndarray
            Predicted output.
        """
        
        x_new = (x_new - self.x_mean) / (self.x_max - self.x_min)  # Scale new data
        z_new = np.dot(self.weight * x_new) + self.bias  # Compute predictions
        y_new = self.cost.valuFunction(y_new)

        return y_new, y_new > 0.5
    


    def calculateY_predict(self) -> None:
        """
        Compute predictions on the training data using the current model parameters (weight, bias).
        """
        self.y_predict = np.dot(self.x_train, np.array(self.weight)) + np.array(self.bias)

    def trainModel(self) -> None:
        """
        Train the model using gradient descent and compute the Mean Squared Error (MSE).
        """
        self.GD()  # Perform gradient descent
        self.calculateY_predict()  # Compute predictions
        self.calculateMSE(self.weight, self.bias)  # Compute MSE

    def dJdB(self) -> float:
        """
        Compute the partial derivative of the cost function with respect to the bias.
        
        Returns:
        --------
        float
            Gradient of cost with respect to bias.
        """
        self.calculateY_predict()  # Update predictions
        tmpdJdB = np.mean(self.y_predict - self.y_train)  # Compute gradient for bias
        return tmpdJdB

    def GD(self) -> None:
        """
        Perform gradient descent to optimize the weights and bias.
        Placeholder method, should be implemented in specific regression models.
        """
        pass

    def calculateMSE(self, weight: float, bias: float) -> float:
        """
        Compute the Mean Squared Error (MSE) for the model.

        Parameters:
        ----------- 
        weight : float
            Weight parameter.
        bias : float
            Bias parameter.

        Returns:
        --------
        float
            The MSE value.
        """
        pass  # To be implemented







class CostFunction:
    def __init__(self, costName):
        self.costFunction = {
            "sigmoid": Sigmoid
        }
        self.cost = self.costFunction[costName]

        
    



class Sigmoid():
    def __init__(self):
        self.name = "Sigmoid"
        self.math = "1/(1 + e^-z)"
    

    def valuFunction(self, z):
        return 1/(1 + np.exp(-z))
    

    def djdw():
        pass





