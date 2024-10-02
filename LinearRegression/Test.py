
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
matplotlib.style.use('ggplot')
from linearRegression import UniVarModel
from linearRegression import MultiVarModel



mtcars = pd.read_csv("LinearRegression/mtcars.csv")

model = UniVarModel(mtcars['wt'], mtcars['mpg'], True)
model.trainModel()


modelMulti = MultiVarModel(mtcars[['wt','qsec']], mtcars['mpg'], True)
modelMulti.trainModel()
print(modelMulti.weight)


from sklearn import linear_model

# Initialize model
regression_model = linear_model.LinearRegression()

# Train the model using the mtcars data
regression_model.fit(X = pd.DataFrame(mtcars[['wt','qsec']]), 
                     y = mtcars["mpg"])

# Check trained model y-intercept
print(regression_model.intercept_)

# Check trained model coefficients
print(regression_model.coef_)


print(modelMulti.calculateMSE(modelMulti.weight, modelMulti.bias))

print(modelMulti.calculateMSE(regression_model.coef_ , regression_model.intercept_))