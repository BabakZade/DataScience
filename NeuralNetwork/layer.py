
import numpy as np
from neuron import Neuron
from activation import *




class Layer:

    def __init__(self, units = 1, activationName = 'Sigmoid', randomActivation = True):

        self.units = units
        self.layer = np.empty(units, Neuron)
        self.activationName = activationName
        self.randomActivation = randomActivation
        self.m = 0 
        self.n = 0
        for i in range(units):            
            if randomActivation:
                print(Activation.__subclasses__())
                dictClass = Activation.__subclasses__()
                activationCl = np.random.choice(dictClass)
                activationName = activationCl.__name__
            self.layer[i] = Neuron(activationName)
    
    def initialize(self, a_in):    
        if self.m != a_in.shape[0]:
            self.m = a_in.shape[0]
            self.a_out = np.zeros(shape=(self.m, self.units), dtype= float)
            self.da_l = np.zeros(shape=(self.m, self.units), dtype= float)
            self.z_out = np.zeros(shape=(self.m, self.units), dtype= float)
            self.z_prime = np.zeros(shape=(self.m, self.units), dtype= float)
            self.dz = np.zeros(shape=(self.m, self.units), dtype= float)
         

        if self.n != a_in.shape[1]:      
            self.n = a_in.shape[1]  
            self.weights = np.zeros(shape=(self.units, self.n), dtype= float)
            self.bias = np.zeros(shape=(self.units), dtype= float)
            self.dw = np.zeros(shape=(self.units, self.n), dtype= float)
            self.db = np.zeros(shape=(self.units), dtype= float)
    



    def dense(self, a_in):
        self.a_in = a_in
        for i in range(self.units):
            self.z_out[:,i], self.a_out[:, i] = self.layer[i].activate(self.a_in, self.weights[i,:], self.bias[i])



    def getWeights(self):
        return self.weights, self.bias
    

    def setWeights(self, weights = None, bias = None):
        if weights is not None:
            self.weights = weights 
            self.bias = bias
        else:
            for i in range(len(self.layer)):
                self.weights[i,:] = self.layer[i].activation.weightinitial(self.n)
                self.bias[i] = self.layer[i].activation.weightinitial(1)

    
    def set_backward(self, da):
        
        for n in range(self.units):
            self.dw[n,:], self.db[n], self.dz[:,n] = self.layer[n].gradian(da[:,n])
            
        
        


    
        

            


