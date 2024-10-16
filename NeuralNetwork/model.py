
import math
from layer import *




class Sequence:
    def __init__(self, x_train, y_train, sequence):
        self.sequence = sequence
        output_layer = Layer(units=y_train.shape[1], activationName = self.sequence[-1].activationName, randomActivation=self.sequence[-1].randomActivation)
        self.sequence.append(output_layer)
        self.m, self.n = x_train.shape
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = 0
        
            
    

    def forward(self, xInput, old_sequence = None):
        a_in = xInput
        for l in range(len(self.sequence)):
            self.sequence[l].initialize(a_in)
            self.updateWB(l_index = l, old_sequence= old_sequence)
            
            self.sequence[l].dense(a_in)
            a_in = self.sequence[l].a_out

        # activation[-1] => m * l, sum(activation[-1], over l: axis = 1)
        # last layer for multiple label
        sum_row = np.sum(self.sequence[-1].a_out, axis = 1, keepdims=True)
        self.sequence[-1].a_out =  self.sequence[-1].a_out / sum_row




    def calculeLost(self, yOutput):
        m, n = yOutput.shape # total instance and total lables
        self.cost = 0

        # Clip predictions to avoid log(0) errors
        self.sequence[-1].a_out = np.clip(self.sequence[-1].a_out, 1e-15, 1 - 1e-15)

        # Compute the binary cross-entropy loss using vectorized operations
        self.cost = -(1 / (m * n)) * np.sum(yOutput * np.log(self.sequence[-1].a_out) + (1 - yOutput) * np.log(1 - self.sequence[-1].a_out))




    def backward(self, yOutput):
        m = yOutput.shape[0]
        # activation is m * c
        # yOutput is m * c
        # in last layer
        da_l = self.sequence[-1].a_out - yOutput
        m = yOutput.shape[0]  # Number of training examples

        for l in reversed(range(len(self.sequence))):
            # Step 1: Compute dZ[l] = dA[l] * activation_prime(Z[l])
            # Step 2: Compute dW[l] = (1/m) * dZ[l] * A[l-1]^T
            # Step 3: Compute db[l] = (1/m) * sum(dZ[l])
            self.sequence[l].set_backward(da_l)

            # Step 4: Compute dA[l-1] = W[l]^T * dZ[l] (for the previous layer, except for the first layer)
            if l != 0:
                da_l = np.dot(self.sequence[l].weights.T, self.sequence[l].dz)

            # Step 5: Update W[l] and b[l] using gradients
            self.sequence[l].weights -= self.alpha * self.sequence[l].dw
            self.sequence[l].bias -= self.alpha * self.sequence[l].db

            

    def updateWB(self, l_index, old_sequence = None):
        # weight update
        if old_sequence != None:            
            tmpw, tmpb = old_sequence[l_index].getWeights()
            self.sequence[l_index].setWeights(tmpw, tmpb)
        else:               
            self.sequence[l_index].setWeights()
        pass
        



    def propagate(self):
        
        n_sections = int( np.round(self.m / self.batch_size))
        # Split X_train into n sections (ensure each section has roughly equal rows)
        X_sections = np.array_split(self.x_train, n_sections)
        y_sections = np.array_split(self.y_train, n_sections)
        # initial forward
        
        for section in range(n_sections):
            if section == 0:
                self.forward(X_sections[section])
            else:
                self.forward(X_sections[section], self.sequence)
            

            self.calculeLost(y_sections[section])
            print(f"cost = {self.cost :.5f} ================ batch {section + 1}/{n_sections}")
            self.backward(y_sections[section])













class Model:

    def __init__(self, x_train, y_train, sequenced_layer):
        self.x_train = x_train.to_numpy()
        self.y_train = self.getMatrixLable(y_train)
        self.sequence = Sequence(self.x_train, self.y_train, sequenced_layer)
        pass

    def compile():


        pass

    def getMatrixLable(self, y_array):
        # create matrix, i.e. c classes and m instance y is matrix of [m*c]
        # labels
        labels = np.unique(y_array)
        output = np.zeros((len(y_array), len(labels)))
        for i in range(len(labels)):
            output[y_array == labels[i], i] = 1
        return output

    def fit(self, epoch, batch_size):
        self.sequence.batch_size = batch_size
        for ep in range(epoch):
            print(f"========================================= epoch {ep + 1}/{epoch}")
            self.stepsPerEpoch()




           

        pass

    def predict():




        pass

    def stepsPerEpoch(self):
        self.sequence.propagate()

        pass



    def getlayer(self, layerName):
        return self.sequence[layerName]
    

    def summary(self):
        pass


    pass