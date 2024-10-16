# TensorFlow 2.x - Multiclass Classification

# Dataset
# IRIS dataset is used in this kernel to demonstrate the multiclass classification using TensorFlow 2.x.
# This dataset has 5 features, out of which 4 features are numeric features and 1 is a categorical feature.

# 1. Import dependent libraries
import pandas as pd
import numpy as np
import os

# Plotting libraries
import matplotlib.pyplot as plt

# SKLearn libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# My neural network
from layer import *
from model import *

# Data file path
FILE_PATH = 'NeuralNetwork/Data/IRIS.csv'

# Dataframe from csv file
iris_data = pd.read_csv(FILE_PATH, header=0)
iris_data.info()

iris_data.head(10)

# 2. Preparing dataset
X = iris_data.loc[:, iris_data.columns != 'species']
y = iris_data.loc[:, ['species']]

y_enc = LabelEncoder().fit_transform(y)
print(y_enc)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3)
print(f"Train shape : {X_train.shape}, Y Train : {y_train.shape}")
print(X_train.shape[1:])



layer_1 = Layer(8, 'Sigmoid')
layer_2 = Layer(6, 'Sigmoid')

model = Model(X_train, y_train, [layer_1, layer_2])
model.fit(epoch= 5, batch_size= 32)
