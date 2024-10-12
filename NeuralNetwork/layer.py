
import numpy as np
from neuron import Neuron
from LogisticRegression.logisticRegression import *




class Layer:

    def __init__(self, x_train, y_train, numberOfNode, activationName = 'Sigmoid', randomActivation = 'False', randomScaler = 'False', randomOptimizer = 'False', lambdaRegularization = 1 , batchSize = 1):

        
        self.layer = np.empty(numberOfNode, Neuron)
        self.x_train = x_train
        self.y_train = y_train
        self.m, self.n = x_train.shape
        self.activationName = activationName
        self.lambdaRegularization = lambdaRegularization

        self.settings(randomActivation, randomScaler, randomOptimizer, batchSize)

        pass

    def settings(self, randomActivation, randomScaler, randomOptimizer, batchSize):
        self.randomActivation = randomActivation
        self.batchSize = batchSize
        self.randomScaler = randomScaler
        self.randomOptimizer = randomOptimizer
        pass

    def runNeuran(self):
        for i in len(self.layer):
            # Determine the number of rows to select 
            sizeXY = int(self.batchSize * self.m)

            # Generate random indices to select
            indices = np.random.choice(self.m, sizeXY, replace=False)

            # Use the same indices to select rows from both x and y
            x_selected = self.x_train[indices]
            y_selected = self.y_train[indices]
            activationName = self.activationName
            scalerName = 'MinMax'
            optimizerName = 'GradientDescent'
            if self.randomActivation:
                dictClass = Activation._derived
                activationName = np.random.choice(list(dictClass.values()))
            if self.randomOptimizer:
                dictClass = Optimizer._derived
                optimizerName = np.random.choice(list(dictClass.values()))
            if self.randomScaler:
                dictClass = Scale._derived
                scalerName = np.random.choice(list(dictClass.values()))



            self.layer[i] = Neuron(x_train=x_selected, y_train= y_selected, classifierTreshhold=0.5, costfunctionName= activationName,scalerName=scalerName, optimizerName=optimizerName, lambdaRegularization=self.lambdaRegularization, ifDetail = False)


    def getWeights(self):
        return self.weights, self.bias
    

    def setWeights(self, weights, bias):
        self.weights = weights 
        self.bias = bias






    def setInitialWeight(self):
        pass

    pass