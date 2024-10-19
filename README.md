# DataScience

This repository was started as a personal project to deepen my understanding of data science techniques. The goal is to implement core data science algorithms and models entirely from scratch, without relying on high-level machine learning libraries like `scikit-learn`. This project includes implementations of machine learning algorithms, optimization techniques, and data processing utilities in Python.

## Folder Structure

The repository is organized into several directories, each containing specific components required for building and evaluating machine learning models:

```bash
DataScience/
├── README.md
├── requirements.txt
├── LinearRegression/
│   ├── LinearModel.py          # Implements linear regression
│   ├── __init__.py
│   └── tests/                  # Unit tests for Linear Regression
│       └── test_LinearModel.py
├── LogisticRegression/
│   ├── LogisticModel.py        # Implements logistic regression
│   ├── __init__.py
│   └── tests/                  # Unit tests for Logistic Regression
│       └── test_LogisticModel.py
├── NeuralNetwork/
│   ├── Model.py                # Implements the neural network model
│   ├── Layer.py                # Layer implementation for neural networks
│   ├── Neuron.py               # Neuron implementation
│   ├── __init__.py
│   └── tests/                  # Unit tests for Neural Network models
│       └── test_Model.py
├── Optimizer/
│   ├── GradientDescent.py      # Gradient Descent optimizer
│   ├── Adam.py                 # Adam optimizer
│   ├── RMSprop.py              # RMSProp optimizer
│   ├── __init__.py
│   └── tests/                  # Unit tests for optimizers
│       └── test_Optimizer.py
├── Activation/
│   ├── Sigmoid.py              # Sigmoid activation function
│   ├── ReLU.py                 # ReLU activation function
│   ├── SoftMax.py              # SoftMax activation function
│   ├── Tanh.py                 # Tanh activation function
│   ├── __init__.py
│   └── tests/                  # Unit tests for activation functions
│       └── test_Activation.py
├── Scale/
│   ├── StandardScaler.py       # Implements standard scaling (Z-score normalization)
│   ├── MinMaxScaler.py         # Implements Min-Max scaling
│   ├── __init__.py
│   └── tests/                  # Unit tests for scaling methods
│       └── test_Scale.py
├── Metrics/
│   ├── Metrics.py              # Accuracy, Precision, Recall, F1 Score calculations
│   ├── __init__.py
│   └── tests/                  # Unit tests for metrics
│       └── test_Metrics.py
└── utils/
    ├── DataLoader.py           # Utility functions for data loading and preprocessing
    ├── Visualizer.py           # Utility functions for data visualization
    ├── __init__.py
    └── tests/                  # Unit tests for utility functions
        └── test_utils.py
