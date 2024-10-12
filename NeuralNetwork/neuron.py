from LogisticRegression.logisticRegression import *


class Neuron(LogisticRegression):

    def __init__(self, x_train, y_train, classifierTreshhold = 0.5, costfunctionName = 'sigmoid', scalerName = 'minmax', optimizerName = 'gd', lambdaRegularization = 0, ifDetail = 'true'):
        super().__init__(x_train, y_train, classifierTreshhold, costfunctionName, scalerName, optimizerName, lambdaRegularization, ifDetail)

        self.activation, self.lable = self.predict(x_test=x_train)

    pass

