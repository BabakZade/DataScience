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

# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

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

# Converting the label into a matrix form
y_label = tf.keras.utils.to_categorical(y_enc)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.3)
print(f"Train shape : {X_train.shape}, Y Train : {y_train.shape}")
print(X_train.shape[1:])

# Defining the model
def get_model():
    model = Sequential([
        keras.layers.Input(shape=X_train.shape[1:]),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation='softmax')  # 3 classes for Iris dataset
    ])
    return model

model = get_model()

# Compiling the model
model.compile(optimizer='adam', 
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=1)

# Evaluating the model
model.evaluate(X_test, y_test)

# Predicting on the test set
y_pred_proba = model.predict(X_test)

# Convert the probabilities to class labels
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)  # True class labels

# Calculate precision for each class
def calculate_precision_per_class(y_true, y_pred, num_classes):
    precision_per_class = []
    
    for class_index in range(num_classes):
        # Calculate True Positives (TP) and False Positives (FP) for the current class
        true_positive = np.sum((y_pred == y_true) & (y_pred == class_index))
        false_positive = np.sum((y_pred == class_index) & (y_true != class_index))
        
        # Calculate precision for the current class
        precision = true_positive / (true_positive + false_positive + 1e-10)  # Avoid division by zero
        precision_per_class.append(precision)
        
        print(f"Precision for class {class_index}: {precision:.4f}")
    
    return precision_per_class

# Assuming there are 3 classes (for the Iris dataset)
num_classes = 3
precision_per_class = calculate_precision_per_class(y_true, y_pred, num_classes)

# Calculate macro-averaged precision (average precision across all classes)
macro_precision = np.mean(precision_per_class)
print(f"Macro-Averaged Precision: {macro_precision:.4f}")
