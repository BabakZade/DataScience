import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

# Base class for regression models
class Regression:
    """
    A base class for regression models, supporting gradient descent and prediction functionality.
    Can be extended for specific types of regression models (univariate, multivariate).
    """
    
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, ifDetail: bool) -> None:
        """
        Initialize the Regression class with training data and settings.
        
        Parameters:
        -----------
        x_train : np.ndarray
            Input features (can be univariate or multivariate).
        y_train : np.ndarray
            Output target values.
        ifDetail : bool
            Whether to show detailed logs/plots during training.
        """
        self.x_train = x_train
        self.y_train = y_train
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

    def yHat(self, x_new: np.ndarray) -> np.ndarray:
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
        y_new = np.dot(self.weight * x_new) + self.bias  # Compute predictions
        return y_new

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


class UniVarModel(Regression):
    """
    A univariate regression model class that inherits from the Regression base class. This class implements
    gradient descent to train the model and predicts outcomes based on a single feature (univariate).
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, ifDetail: bool) -> None:
        """
        Initialize the Model class for univariate regression by calling the parent class (Regression) constructor.

        Parameters:
        -----------
        x_train : np.ndarray
            The input (features) training data for univariate regression.
        y_train : np.ndarray
            The output (target) training data for univariate regression.
        ifDetail : bool
            Flag to indicate if detailed information (such as plots) should be shown during training.
        """
        super(UniVarModel, self).__init__(x_train, y_train, ifDetail)  # Call the parent class constructor

    def trainModel(self) -> None:
        """
        Train the univariate regression model using gradient descent, then compute predictions and Mean Squared Error (MSE).
        If detailed mode is enabled, plots the training progress, including a contour plot of the cost function.
        """
        self.GD()  # Perform Gradient Descent
        self.calculateY_predict()  # Calculate predictions after training
        self.calculateMSE(self.weight, self.bias)  # Calculate Mean Squared Error
        if self.ifDetail:
            self.plotCountorForTraining()  # Plot contour if detailed mode is enabled
    def calculateY_predict(self) -> None:
        """
        Compute predictions on the training data using the current model parameters (weight, bias).
        """
        self.y_predict = self.x_train * self.weight + self.bias
    def GD(self) -> None:
        """
        Perform Gradient Descent optimization to update model parameters (weight and bias).
        Iteratively adjusts parameters to minimize the cost function (MSE).
        """
        # If detailed, initialize arrays to store weight and bias history for visualization
        if self.ifDetail:
            self.arrayOfW = []  # Store weight history for plotting
            self.arrayOfB = []  # Store bias history for plotting
        
        tmpB = self.bias  # Temporary variable to store updated bias
        tmpW = self.weight  # Temporary variable to store updated weight
        iteration = 0
        oldCost = self.calculateMSE(tmpW, tmpB)  # Initial cost
        newCost = oldCost
        
        # If detailed, store the cost history to visualize later
        if self.ifDetail:
            cost_history = []  # Store cost function values for plotting

        # Gradient Descent Loop
        while True:
            if self.ifDetail:
                cost_history.append(newCost)  # Save the current cost to history

            iteration += 1
            tmpW = self.weight - self.alpha * self.dJdW()  # Update weight using gradient and learning rate
            tmpB = self.bias - self.alpha * self.dJdB()  # Update bias using gradient and learning rate
            oldCost = newCost  # Store the old cost
            newCost = self.calculateMSE(tmpW, tmpB)  # Recompute cost with updated parameters

            # Check for convergence: if change in weight, bias, or cost is below a threshold, stop
            maxGap = np.max(np.abs(tmpW - self.weight))  # Max change in weights
            if abs(tmpB - self.bias) < self.gap and maxGap < self.gap:
                break
            if abs(newCost - oldCost) < self.epsilon:
                break

            # Dynamically tune the learning rate based on performance
            self.tuneLearningRate(iteration, oldCost, newCost)

            # Print details every 500 iterations if detailed logging is enabled
            if self.ifDetail and iteration % self.detailIteration == 0:
                print(f"Iteration {iteration}, Cost: {newCost:.3f}, maxGapW: {maxGap:.4f}, B: {tmpB:.4f}, alpha: {self.alpha:.4f}")
                if tmpB != self.bias:
                    self.arrayOfB.append(tmpB)  # Record bias history
                if tmpW != self.weight:
                    self.arrayOfW.append(tmpW)  # Record weight history

            # Update the model's parameters
            self.weight = tmpW
            self.bias = tmpB

        # If detailed mode is enabled, plot the cost history after training
        if self.ifDetail:
            plt.plot(range(len(cost_history)), cost_history)  # Plot cost history over iterations
            plt.show()

    def dJdW(self) -> float:
        """
        Compute the partial derivative of the cost function with respect to the weight.

        Returns:
        --------
        float
            The computed gradient for the weight, indicating how the cost function changes with respect 
            to changes in the weight.
        """
        self.calculateY_predict()  # Update predictions
        # Compute the gradient of the cost with respect to weight
        tmpdJdW = np.mean(np.dot((self.y_predict - self.y_train), self.x_train))
        return tmpdJdW
    
    def plotResult(self) -> None:
        """
        Plot the results of the trained model, showing the training data points and the model's predictions.
        This includes a scatter plot of the training data and a line plot of the model's predictions.
        """
        plt.scatter(self.x_train, self.y_train)  # Scatter plot of the data points
        plt.plot(self.x_train, self.y_predict)  # Plot predicted line after training
        plt.title(f'MSE = {self.calculateMSE(self.weight, self.bias)}')  # Display the MSE in the title
        plt.show()

    def calculateMSE(self, weight: float, bias: float) -> float:
        """
        Calculate the Mean Squared Error (MSE) for given weight and bias values.

        Parameters:
        ----------- 
        weight : float
            The weight (slope) parameter.
        bias : float
            The bias (intercept) parameter.

        Returns:
        --------
        float
            The computed MSE for the given weight and bias.
        """
        y_predict = weight * self.x_train + bias  # Compute predicted values
        return 1 / (2 * len(self.x_train)) * np.sum(np.square(y_predict - self.y_train))  # Return MSE

    def plotCountorForTraining(self) -> None:
        """
        Plot a contour and 3D surface of the cost function (MSE) as a function of the weight and bias
        during the training process. This visualizes how the model converges over time.
        """
        # Determine the size for visualization (limit history if necessary)
        size = np.min([len(self.arrayOfB), len(self.arrayOfW), 100])
        arrayW = self.arrayOfW[len(self.arrayOfW) - size: len(self.arrayOfW)]  # Subset weight history
        arrayB = self.arrayOfB[len(self.arrayOfB) - size: len(self.arrayOfB)]  # Subset bias history
        arrayMSE = np.zeros(len(arrayW) * len(arrayB))  # Initialize MSE grid
        arrayMSE = arrayMSE.reshape((len(arrayW), len(arrayB)))  # Reshape into a 2D grid
        minMSE = sys.float_info.max  # Track the minimum MSE
        minI = -1
        minJ = -1

        # Compute MSE grid based on weight and bias values during training
        for i in range(len(arrayW)):
            for j in range(len(arrayB)):
                arrayMSE[i][j] = self.calculateMSE(arrayW[i], arrayB[j])
                if arrayMSE[i][j] < minMSE:
                    minJ = j
                    minI = i

        # Plot 2D scatter and contour plots
        f, axes = plt.subplots(nrows=2, ncols=2)
        axes[0][0].scatter(arrayB, arrayMSE[minI, :])
        axes[0][1].scatter(arrayW, arrayMSE[:, minJ])

        X, Y = np.meshgrid(arrayW, arrayB)  # Create grid for 3D plot
        axes[1][0].contour(X, Y, arrayMSE)
        axes[1][0].set_title('J(w,b)')
        axes[1][0].set_xlabel('weight')
        axes[1][0].set_ylabel('bias')

        # Create a 3D surface plot
        axes[1][1].remove()  # Remove placeholder axis
        axes[1][1] = f.add_subplot(224, projection='3d')
        axes[1][1].plot_surface(X, Y, arrayMSE, rstride=8, cstride=8, alpha=0.8, cmap=cm.coolwarm)
        axes[1][1].set_xticklabels('')
        axes[1][1].set_yticklabels('')
        axes[1][1].set_zticklabels('')

        f.subplots_adjust(hspace=0, wspace=0)  # Adjust layout spacing
        plt.show()  # Display the plot



class MultiVarModel(Regression):
    """
    A multivariate regression model class that inherits from the Regression base class.
    Implements gradient descent to train the model and predicts outcomes based on multiple features (multivariate).
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, ifDetail: bool) -> None:
        """
        Initialize the MultiVarModel class by calling the parent class (Regression) constructor.

        Parameters:
        -----------
        x_train : np.ndarray
            The input (features) training data for multivariate regression (can have multiple features).
        y_train : np.ndarray
            The output (target) training data for multivariate regression.
        ifDetail : bool
            Flag to indicate if detailed information (such as plots) should be shown during training.
        """
        super(MultiVarModel, self).__init__(x_train, y_train, ifDetail)  # Call parent class constructor

    def trainModel(self) -> None:
        """
        Train the multivariate regression model using gradient descent, then compute predictions and Mean Squared Error (MSE).
        If detailed mode is enabled, logs the training progress for visualization.
        """
        self.GD()  # Perform Gradient Descent
        self.calculateY_predict()  # Calculate predictions after training
        self.calculateMSE(self.weight, self.bias)  # Calculate Mean Squared Error (MSE)

    def GD(self) -> None:
        """
        Perform Gradient Descent optimization to update the model's weight and bias parameters.
        Iteratively adjusts parameters to minimize the cost function (MSE).
        """
        tmpB = 0.0  # Temporary variable to store bias updates
        tmpW = np.zeros(np.shape(self.x_train)[-1])  # Temporary variable to store weight updates
        iteration = 0
        oldCost = self.calculateMSE(tmpW, tmpB)  # Calculate initial cost
        newCost = oldCost
        
        # If detailed logging is enabled, store the cost history for plotting later
        if self.ifDetail:
            cost_history = []  # List to store cost function values for each iteration

        # Gradient Descent Loop
        while True:
            if self.ifDetail:
                cost_history.append(newCost)  # Append current cost to history for visualization

            iteration += 1
            tmpW = self.weight - self.alpha * self.dJdW()  # Update weight using gradient and learning rate
            tmpB = self.bias - self.alpha * self.dJdB()  # Update bias using gradient and learning rate
            oldCost = newCost  # Store previous cost
            newCost = self.calculateMSE(tmpW, tmpB)  # Recompute cost with updated parameters

            # Convergence criteria: stop if weight/bias changes or cost changes are smaller than thresholds
            maxGap = np.max(np.abs(tmpW - self.weight))  # Max change in weights
            if abs(tmpB - self.bias) < self.gap and maxGap < self.gap:
                break  # Stop if weight and bias changes are below the gap threshold
            if abs(newCost - oldCost) < self.epsilon:
                break  # Stop if cost change is below the epsilon threshold

            # Dynamically tune the learning rate based on performance
            self.tuneLearningRate(iteration, oldCost, newCost)

            # Log progress if detailed logging is enabled (every few iterations)
            if self.ifDetail and iteration % self.detailIteration == 0:
                print(f"Iteration {iteration}, Cost: {newCost:.3f}, MaxWGap: {maxGap:.4f}, B: {tmpB: .4f}, alpha: {self.alpha: .4f}")
            
            # Update the model's weight and bias
            self.weight = tmpW
            self.bias = tmpB

        # Plot the cost history if detailed logging is enabled
        if self.ifDetail:
            plt.plot(range(len(cost_history)), cost_history)  # Plot cost over iterations
            plt.show()

    def dJdW(self) -> float:
        """
        Compute the partial derivative of the cost function with respect to the weights (gradient).

        Returns:
        --------
        np.ndarray
            The computed gradient for the weights, indicating how the cost function changes with respect 
            to changes in each weight (feature coefficient).
        """
        self.calculateY_predict()  # Ensure predictions are up-to-date
        # Compute the gradient of the cost function with respect to weights
        tmpdJdW = np.mean(np.array(self.y_predict - self.y_train).reshape(-1, 1) * self.x_train, axis=0)
        return tmpdJdW

    def dJdB(self) -> float:
        """
        Compute the partial derivative of the cost function with respect to the bias (gradient).

        Returns:
        --------
        float
            The computed gradient for the bias, indicating how the cost function changes with respect to changes in bias.
        """
        self.calculateY_predict()  # Ensure predictions are up-to-date
        # Compute the gradient of the cost function with respect to the bias
        tmpdJdB = np.mean(self.y_predict - self.y_train)
        return tmpdJdB

    def calculateMSE(self, weight: np.ndarray, bias: float) -> float:
        """
        Calculate the Mean Squared Error (MSE) for the current model parameters (weights and bias).

        Parameters:
        -----------
        weight : np.ndarray
            The weight (coefficients for the features) of the regression model.
        bias : float
            The bias (intercept) of the regression model.

        Returns:
        --------
        float
            The computed MSE for the given weight and bias.
        """
        # Compute predictions based on current weights and bias
        y_predict = np.dot(self.x_train, weight) + bias
        # Calculate and return the Mean Squared Error (MSE)
        return 1 / (2 * len(self.x_train)) * np.sum(np.square(y_predict - self.y_train))



if __name__ == '__main__':
    pass