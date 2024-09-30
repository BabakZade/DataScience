
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
matplotlib.style.use('ggplot')
from univariate import Model



mtcars = pd.read_csv("LinearRegression/mtcars.csv")
model = Model(mtcars['wt'], mtcars['mpg'], True)
model.trainModel()
model.plotCountorForTraining()





from sklearn import linear_model

# Initialize model
regression_model = linear_model.LinearRegression()

# Train the model using the mtcars data
regression_model.fit(X = pd.DataFrame(mtcars["wt"]), 
                     y = mtcars["mpg"])

# Check trained model y-intercept
print(regression_model.intercept_)

# Check trained model coefficients
print(regression_model.coef_)

