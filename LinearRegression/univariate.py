import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
class Model:
    def __init__(self, x_train, y_train, weight, bias) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.weight = weight
        self.bias = bias
        pass

    def calculateYhat(self):
        self.y_predict = self.weight * self.x_train + self.bias
        pass

    def calculateCoeff(self):
        pass
    
    def trainModel(self):
        self.calculateCoeff()
        self.calculateYhat()
        self.calculateMSE()
        self.plotResult()

    def plotResult(self):
        
        plt.scatter(self.x_train, self.y_train)
        plt.plot(self.x_train, self.y_predict)
        plt.title(f'MSE = {self.MSE}')
        plt.show()

    def calculateMSE(self):
        self.MSE = 0.5 * np.mean((np.array(self.y_train - self.y_predict))**2)
        

        
    



if __name__ == '__main__':
    pass
    


    
