import numpy as np

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy as the proportion of correct predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray
        The true class labels.
    y_pred : np.ndarray
        The predicted class labels.
        
    Returns:
    --------
    float
        Accuracy score.
    """
    return np.sum(y_true == y_pred) / len(y_true)


def calculate_precision(y_true, y_pred, num_classes):
    """
    Calculate macro-averaged precision score.
    
    Parameters:
    -----------
    y_true : np.ndarray
        The true class labels.
    y_pred : np.ndarray
        The predicted class labels.
    num_classes : int
        The number of unique classes.
        
    Returns:
    --------
    float
        Macro-averaged precision score.
    """
    precision_per_class = []
    for cls in range(num_classes):
        true_positive = np.sum((y_pred == cls) & (y_true == cls))
        false_positive = np.sum((y_pred == cls) & (y_true != cls))
        precision = true_positive / (true_positive + false_positive + 1e-10)  # Avoid division by zero
        precision_per_class.append(precision)
    
    return np.mean(precision_per_class)  # Macro-average


def calculate_recall(y_true, y_pred, num_classes):
    """
    Calculate macro-averaged recall score.
    
    Parameters:
    -----------
    y_true : np.ndarray
        The true class labels.
    y_pred : np.ndarray
        The predicted class labels.
    num_classes : int
        The number of unique classes.
        
    Returns:
    --------
    float
        Macro-averaged recall score.
    """
    recall_per_class = []
    for cls in range(num_classes):
        true_positive = np.sum((y_pred == cls) & (y_true == cls))
        false_negative = np.sum((y_pred != cls) & (y_true == cls))
        recall = true_positive / (true_positive + false_negative + 1e-10)  # Avoid division by zero
        recall_per_class.append(recall)
    
    return np.mean(recall_per_class)  # Macro-average


def calculate_f1(precision, recall):
    """
    Calculate F1-score from precision and recall values.
    
    Parameters:
    -----------
    precision : float
        The macro-averaged precision score.
    recall : float
        The macro-averaged recall score.
        
    Returns:
    --------
    float
        F1-score.
    """
    return 2 * (precision * recall) / (precision + recall + 1e-10)  # Avoid division by zero
