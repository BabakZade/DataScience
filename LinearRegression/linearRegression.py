import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

class Regression:
    """
    A base class for regression models. This class implements functionality for performing gradient descent 
    and predicting outcomes for a given dataset. It can be extended for specific types of regression, such 
    as univariate or multivariate regression.

    Attributes:
    -----------
    x_train : np.ndarray
        The input (features) training data.
    y_train : np.ndarray
        The output (target) training data.
    weight : float or np.ndarray
        The weight (or coefficient) parameter for the regression model.
    bias : float
        The bias (intercept) parameter for the regression model.
    ifDetail : bool
        Flag to indicate if detailed information (such as plots) should be shown during training.
    alpha : float
        The learning rate for gradient descent.
    gap : float
        The threshold for the minimum difference between weight updates to stop gradient descent.
    epsilon : float
        The threshold for the minimum difference between cost updates to stop gradient descent.
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, ifDetail: bool) -> None:
        """
        Initialize the Regression class with training data and settings.

        Parameters:
        -----------
        x_train : np.ndarray
            The input (features) training data, which can be univariate or multivariate.
        y_train : np.ndarray
            The output (target) training data.
        ifDetail : bool
            Flag to indicate if detailed information (such as plots) should be shown during training.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.weight = np.zeros(np.shape(x_train)[-1])
        self.bias = 0
        self.ifDetail = ifDetail
        self.settings()

    def settings(self) -> None:
        """
        Set up the default settings for gradient descent parameters such as learning rate, gap (minimum 
        difference between weight updates), and epsilon (minimum difference between cost updates).
        """
        self.alpha = 0.001  # Learning rate
        self.gap = 0.0001  # Minimum difference between weight updates to stop iteration
        self.epsilon = 0.0000001  # Minimum difference between cost updates to stop iteration

    def yHat(self, x_new: np.ndarray) -> np.ndarray:
        """
        Compute predicted y values for new input data based on the trained model parameters.

        Parameters:
        -----------
        x_new : np.ndarray
            The new input data for which predictions are to be made.

        Returns:
        --------
        np.ndarray
            The predicted y values corresponding to the input data.
        """
        y_new = np.dot(self.weight * x_new) + self.bias
        return y_new

    def calculateY_predict(self) -> None:
        """
        Calculate predicted values (`self.y_predict`) for the training data (`self.x_train`) using the 
        current model parameters (`self.weight`, `self.bias`).
        """
        self.y_predict = np.dot(self.x_train, np.array(self.weight)) + np.array(self.bias)

    def trainModel(self) -> None:
        """
        Train the regression model using gradient descent and compute the predictions and Mean Squared Error (MSE).
        This function calls the gradient descent method (`GD`) to optimize the weight and bias, then computes
        predictions and calculates the MSE. If detailed mode is enabled, it will also plot the training progress.
        """
        self.GD()  # Perform Gradient Descent
        self.calculateY_predict()  # Calculate predictions after training
        self.calculateMSE(self.weight, self.bias)  # Calculate Mean Squared Error

    def dJdB(self) -> float:
        """
        Compute the partial derivative of the cost function with respect to the bias.

        Returns:
        --------
        float
            The computed gradient for the bias, which indicates how the cost function changes with respect 
            to changes in the bias.
        """
        self.calculateY_predict()  # Ensure y_predict is up to date
        tmpdJdB = np.mean(self.y_predict - self.y_train)  # Compute the gradient for the bias
        return tmpdJdB

    def GD(self) -> None:
        """
        Perform gradient descent optimization to update the model's weight and bias. This method should
        be implemented in the child class or extended in this class for specific types of regression.

        It will iteratively adjust `self.weight` and `self.bias` to minimize the cost function.

        Currently, this is a placeholder method.
        """
        pass  # To be implemented

    def calculateMSE(self, weight: float, bias: float) -> float:
        """
        Calculate the Mean Squared Error (MSE) for the model given specific weight and bias values.

        Parameters:
        -----------
        weight : float
            The weight (or coefficient) of the regression model.
        bias : float
            The bias (intercept) of the regression model.

        Returns:
        --------
        float
            The Mean Squared Error (MSE) value based on the provided weight and bias.
        
        Currently, this is a placeholder method.
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
        super(UniVarModel, self).__init__(x_train, y_train, ifDetail)

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

    def GD(self) -> None:
        """
        Perform Gradient Descent optimization to update model parameters (`weight` and `bias`).
        Iteratively adjusts parameters to minimize the cost function (MSE).
        """
        if self.ifDetail:
            self.arrayOfW = []  # Store weight history for plotting
            self.arrayOfB = []  # Store bias history for plotting
        
        tmpB = 0.0
        tmpW = 0.0
        iteration = 0
        oldCost = self.calculateMSE(tmpW, tmpB)  # Initial cost
        newCost = oldCost
        if self.ifDetail:
            cost_history = []  # Store cost function values for plotting

        while True:
            if self.ifDetail:
                cost_history.append(newCost)  # Save the current cost to history

            iteration += 1
            tmpW = self.weight - self.alpha * self.dJdW()  # Update weight
            tmpB = self.bias - self.alpha * self.dJdB()  # Update bias
            oldCost = newCost
            newCost = self.calculateMSE(tmpW, tmpB)  # Recompute cost with new parameters

            # Convergence criteria
            if abs(tmpB - self.bias) < self.gap and abs(tmpW - self.weight) < self.gap:
                break
            if abs(newCost - oldCost) < self.epsilon:
                break

            if self.ifDetail and iteration % 500 == 0:  # Log progress every 500 iterations
                print(f"Iteration {iteration}, Cost: {newCost:.3f}, W: {tmpW:.4f}, B: {tmpB: .4f}")
                if tmpB != self.bias:
                    self.arrayOfB.append(tmpB)
                if tmpW != self.weight:
                    self.arrayOfW.append(tmpW)

            self.weight = tmpW
            self.bias = tmpB

        if self.ifDetail:
            plt.plot(range(len(cost_history)), cost_history)  # Plot cost history
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
        self.calculateY_predict()
        tmpdJdW = np.mean(np.dot((self.y_predict - self.y_train), self.x_train))
        return tmpdJdW
    
    def plotResult(self) -> None:
        """
        Plot the results of the trained model, showing the training data points and the model's predictions.
        This includes a scatter plot of the training data and a line plot of the model's predictions.
        """
        plt.scatter(self.x_train, self.y_train)  # Scatter plot of the data
        plt.plot(self.x_train, self.y_predict)  # Plot predicted line
        plt.title(f'MSE = {self.calculateMSE(self.weight, self.bias)}')  # Title showing the MSE
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
        y_predict = weight * self.x_train + bias
        return 1 / (2 * len(self.x_train)) * np.sum(np.square(y_predict - self.y_train))

    def plotCountorForTraining(self) -> None:
        """
        Plot a contour and 3D surface of the cost function (MSE) as a function of the weight and bias
        during the training process. This visualizes how the model converges over time.
        """
        size = np.min([len(self.arrayOfB), len(self.arrayOfW), 100])
        arrayW = self.arrayOfW[len(self.arrayOfW) - size: len(self.arrayOfW)]
        arrayB = self.arrayOfB[len(self.arrayOfB) - size: len(self.arrayOfB)]
        arrayMSE = np.zeros(len(arrayW) * len(arrayB))
        arrayMSE = arrayMSE.reshape((len(arrayW), len(arrayB)))
        minMSE = sys.float_info.max
        minI = -1
        minJ = -1
        for i in range(len(arrayW)):
            for j in range(len(arrayB)):
                arrayMSE[i][j] = self.calculateMSE(arrayW[i], arrayB[j])
                if arrayMSE[i][j] < minMSE:
                    minJ = j
                    minI = i

        f, axes = plt.subplots(nrows=2, ncols=2)
        axes[0][0].scatter(arrayB, arrayMSE[minI, :])
        axes[0][1].scatter(arrayW, arrayMSE[:, minJ])

        X, Y = np.meshgrid(arrayW, arrayB)
        axes[1][0].contour(X, Y, arrayMSE)
        axes[1][0].set_title('J(w,b)')
        axes[1][0].set_xlabel('weight')
        axes[1][0].set_ylabel('bias')

        axes[1][1].remove()
        axes[1][1] = f.add_subplot(224, projection='3d')

        axes[1][1].plot_surface(X, Y, arrayMSE, rstride=8, cstride=8, alpha=0.8, cmap=cm.coolwarm)
        axes[1][1].set_xticklabels('')
        axes[1][1].set_yticklabels('')
        axes[1][1].set_zticklabels('')

        f.subplots_adjust(hspace=0, wspace=0)
        plt.show()

class MultiVarModel(Regression):

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
        super(MultiVarModel, self).__init__(x_train, y_train, ifDetail)
    
    def trainModel(self) -> None:
        """
        Train the univariate regression model using gradient descent, then compute predictions and Mean Squared Error (MSE).
        If detailed mode is enabled, plots the training progress, including a contour plot of the cost function.
        """
        self.GD()  # Perform Gradient Descent
        self.calculateY_predict()  # Calculate predictions after training
        self.calculateMSE(self.weight, self.bias)  # Calculate Mean Squared Error
    def GD(self) -> None:
        """
        Perform Gradient Descent optimization to update model parameters (`weight` and `bias`).
        Iteratively adjusts parameters to minimize the cost function (MSE).
        """
        
        tmpB = 0.0
        tmpW = np.zeros(np.shape(self.x_train)[-1])
        iteration = 0
        oldCost = self.calculateMSE(tmpW, tmpB)  # Initial cost
        newCost = oldCost
        if self.ifDetail:
            cost_history = []  # Store cost function values for plotting

        while True:
            if self.ifDetail:
                cost_history.append(newCost)  # Save the current cost to history

            iteration += 1
            tmpW = self.weight - self.alpha * self.dJdW()  # Update weight

            
            tmpB = self.bias - self.alpha * self.dJdB()  # Update bias
            oldCost = newCost
            newCost = self.calculateMSE(tmpW, tmpB)  # Recompute cost with new parameters

            # Convergence criteria
            maxGap = np.max(np.abs(tmpW - self.weight))
            if abs(tmpB - self.bias) < self.gap and maxGap < self.gap:
                break
            if abs(newCost - oldCost) < self.epsilon:
                break

            if self.ifDetail and iteration % 500 == 0:  # Log progress every 500 iterations
                print(f"Iteration {iteration}, Cost: {newCost:.3f}, MaxWGap: {maxGap:.4f}, B: {tmpB: .4f}")


            self.weight = tmpW
            self.bias = tmpB

        if self.ifDetail:
            plt.plot(range(len(cost_history)), cost_history)  # Plot cost history
            plt.show()

    def dJdW(self) -> float:
        """
        Compute the partial derivative of the cost function with respect to the weight.

        Returns:
        --------
        float
            The computed gradient for the weight.
        """
        self.calculateY_predict()
        tmpdJdW = np.mean(np.array(self.y_predict - self.y_train).reshape(-1, 1) * self.x_train, axis = 0)
        return tmpdJdW
    
    def dJdB(self) -> float:
        """
        Compute the partial derivative of the cost function with respect to the bias.

        Returns:
        --------
        float
            The computed gradient for the bias.
        """
        self.calculateY_predict()
        tmpdJdB = np.mean(self.y_predict - self.y_train)
        return tmpdJdB

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
        y_predict = np.dot( self.x_train, weight) + bias
        return 1 / (2 * len(self.x_train)) * np.sum(np.square(y_predict - self.y_train))

if __name__ == '__main__':
    pass