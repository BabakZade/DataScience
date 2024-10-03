
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
