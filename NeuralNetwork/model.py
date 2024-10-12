
import math
from layer import *


class Sequence:
    def __init__(self, arrayOflayers, randomHiddenLayerOrder = 'false'):

        pass    
    pass

    def forward():
        pass

    def backward(self):
        n_sections = np.round(self.m / self.batch_size)
    
        # Shuffle the rows randomly while keeping X and y aligned
        shuffled_indices = np.random.permutation(self.m)
        X_shuffled = self.x_train[shuffled_indices]
        y_shuffled = self.y_train[shuffled_indices]
        
        # Split X_shuffled into n sections (ensure each section has roughly equal rows)
        X_sections = np.array_split(X_shuffled, n_sections)
        y_sections = np.array_split(y_shuffled, n_sections)
        pass



class Model:


    def __init__(self, sequence):
        pass

    def compile():


        pass

    def fit(self):

        


        for ep in range(self.epoch):
           pass

        pass

    def predict():




        pass

    def perEpoch(self):
        self.sequence.forward()
        self.sequence.backward()

        pass



    def getlayer(self, layerName):
        return self.sequence[layerName]
    

    def summary(self):
        pass


    pass