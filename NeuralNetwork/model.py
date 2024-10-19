import math
from layer import *
from scale import *
from optimizer import *
from metrics import *

class Sequence:
    def __init__(self, x_train, y_train, sequence):
        """
        Initializes the Sequence of layers for the neural network, including the output layer (SoftMax).
        
        Parameters:
        -----------
        x_train : np.ndarray
            Training data (features).
        y_train : np.ndarray
            Training labels (targets).
        sequence : list
            List of layer objects forming the neural network.
        """
        self.sequence = sequence
        
        # Append the output layer which uses the SoftMax activation function
        output_layer = Layer(units=y_train.shape[1], activationName='SoftMax', randomActivation=False)
        self.sequence.append(output_layer)

        self.m, self.n = x_train.shape  # m: number of samples, n: number of features
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = 0
        self.settings()

    def settings(self):
        """
        Configures general settings for the training process, such as the learning rate.
        """
        self.alpha = 0.0001  # Learning rate

    def forward(self, xInput, old_sequence=None):
        """
        Perform the forward pass through each layer in the sequence.
        
        Parameters:
        -----------
        xInput : np.ndarray
            Input data for the forward pass.
        old_sequence : list, optional
            Previous sequence of layers (for updating weights and biases from a previous state).
        """
        a_in = xInput
        for l in range(len(self.sequence)):
            self.sequence[l].initialize(a_in)  # Initialize the layer weights and biases
            self.updateWB(l_index=l, old_sequence=old_sequence)  # Optionally update weights and biases
            self.sequence[l].dense(a_in)  # Perform the forward pass for the current layer
            a_in = self.sequence[l].a_out  # Output of the current layer becomes input for the next

        # Apply SoftMax normalization for the last layer
        sum_row = np.sum(self.sequence[-1].a_out, axis=1, keepdims=True)
        self.sequence[-1].a_out = self.sequence[-1].a_out / sum_row  # Normalize

    def calculeLost(self, yOutput):
        """
        Calculate the loss (binary cross-entropy) for the output.
        
        Parameters:
        -----------
        yOutput : np.ndarray
            The true output labels (target values).
        """
        m, n = yOutput.shape  # m: number of samples, n: number of labels
        self.cost = 0

        # Clip output to avoid log(0) errors in cross-entropy loss calculation
        self.sequence[-1].a_out = np.clip(self.sequence[-1].a_out, 1e-15, 1 - 1e-15)

        # Compute cross-entropy loss
        self.cost = -(1 / (m * n)) * np.sum(
            yOutput * np.log(self.sequence[-1].a_out) + (1 - yOutput) * np.log(1 - self.sequence[-1].a_out)
        )

    def backward(self, yOutput):
        """
        Perform the backward pass through all layers (backpropagation).
        
        Parameters:
        -----------
        yOutput : np.ndarray
            The true output labels (target values).
        """
        m = yOutput.shape[0]  # Number of training examples

        # Calculate error signal for the output layer (da_l = y_hat - y)
        self.sequence[-1].da_l = self.sequence[-1].a_out - yOutput

        # Backpropagate through all layers in reverse order
        for l in reversed(range(len(self.sequence))):
            # Compute gradients for current layer (weights, biases, activations)
            self.sequence[l].set_backward(self.sequence[l].da_l)

            # Compute da for the previous layer, except for the first layer
            if l != 0:
                self.sequence[l - 1].da_l = np.dot(self.sequence[l].dz, self.sequence[l].weights)

            # Update weights and biases using the computed gradients
            self.sequence[l].updateWeight()

    def updateWB(self, l_index, old_sequence=None):
        """
        Update weights and biases from a previous sequence if available.
        
        Parameters:
        -----------
        l_index : int
            Index of the layer being updated.
        old_sequence : list, optional
            Previous sequence of layers for copying weights and biases.
        """
        if old_sequence is not None:
            tmpw, tmpb = old_sequence[l_index].getWeights()
            self.sequence[l_index].setWeights(tmpw, tmpb)
        else:
            self.sequence[l_index].setWeights()

    def propagate(self):
        """
        Split the data into mini-batches, perform forward and backward passes, and calculate metrics for each epoch.
        """
        n_sections = int(np.round(self.m / self.batch_size))  # Number of mini-batches
        X_sections = np.array_split(self.x_train, n_sections)  # Split x_train into mini-batches
        y_sections = np.array_split(self.y_train, n_sections)  # Split y_train into mini-batches

        epoch_loss = 0  # Track total loss for the epoch
        for section in range(n_sections):
            # Forward pass
            if section == 0:
                self.forward(X_sections[section])
            else:
                self.forward(X_sections[section], self.sequence)

            # Calculate the loss for the current mini-batch
            self.calculeLost(y_sections[section])
            print(f"cost = {self.cost :.5f} ================ batch {section + 1}/{n_sections}")
            epoch_loss += self.cost  # Accumulate batch loss

            # Backward pass
            self.backward(y_sections[section])

        epoch_loss /= n_sections  # Average loss over all mini-batches
        print(f"Epoch cost: {epoch_loss :.5f}")

        # Calculate performance metrics on training data
        y_per = self.predict(self.x_train)  # Predictions on the training data
        y_true = np.argmax(self.y_train, axis=1)  # Convert one-hot to categorical labels (true labels)
        y_pred = np.argmax(y_per, axis=1)  # Convert predictions to categorical labels (predicted labels)

        num_classes = self.y_train.shape[1]  # Number of unique classes

        # Calculate and print metrics (accuracy, precision, recall, F1-score)
        accuracy = calculate_accuracy(y_true, y_pred)
        precision = calculate_precision(y_true, y_pred, num_classes)
        recall = calculate_recall(y_true, y_pred, num_classes)
        f1 = calculate_f1(precision, recall)
        print(f"Epoch Metrics: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1-score = {f1:.4f}")

    def predict(self, x_test):
        """
        Perform a forward pass to predict output labels for test data.
        
        Parameters:
        -----------
        x_test : np.ndarray
            Input test data (features).
        
        Returns:
        --------
        np.ndarray
            Softmax-normalized predictions (probabilities).
        """
        a_in = x_test
        for l in range(len(self.sequence)):
            a_out = self.sequence[l].get_a_out(a_in)
            a_in = a_out

        # Apply SoftMax normalization on output layer
        sum_row = np.sum(a_out, axis=1, keepdims=True)
        return a_out / sum_row  # Normalize

class Model:
    def __init__(self, x_train, y_train, sequenced_layer, scalerName="StdZ"):
        """
        Initialize the model with training data, sequence of layers, and scaling method.
        
        Parameters:
        -----------
        x_train : pd.DataFrame
            Training data (features).
        y_train : pd.Series or np.ndarray
            Training labels (targets).
        sequenced_layer : list
            List of layers forming the neural network.
        scalerName : str
            Name of the scaling method (e.g., 'StdZ' for standardization).
        """
        self.scaler = ScaleSelector(scalerName=scalerName).scaler  # Select and apply scaler
        self.x_train = self.scaler.scaleTrain(x_train.to_numpy())  # Scale training data
        self.y_train = self.oneHotEncoding(y_train)  # One-hot encode the labels
        self.sequence = Sequence(self.x_train, self.y_train, sequenced_layer)  # Create the sequence of layers

    def compile():
        pass

    def oneHotEncoding(self, y_array):
        """
        Convert categorical labels into a one-hot encoded matrix.
        
        Parameters:
        -----------
        y_array : np.ndarray
            Input labels.
        
        Returns:
        --------
        np.ndarray
            One-hot encoded labels.
        """
        labels = np.unique(y_array)
        output = np.zeros((len(y_array), len(labels)))
        for i in range(len(labels)):
            output[y_array == labels[i], i] = 1
        return output

    def fit(self, epoch, batch_size):
        """
        Train the model over several epochs using mini-batches.
        
        Parameters:
        -----------
        epoch : int
            Number of epochs to train the model.
        batch_size : int
            Size of mini-batches used for training.
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")
        self.sequence.batch_size = batch_size
        for ep in range(epoch):
            print(f"========================================= epoch {ep + 1}/{epoch}")
            self.stepsPerEpoch()  # Perform training for each epoch

    def predict(self, x_test, y_test):
        """
        Predict labels on test data and calculate performance metrics.
        
        Parameters:
        -----------
        x_test : pd.DataFrame
            Test data (features).
        y_test : pd.Series or np.ndarray
            True labels (targets).
        
        Returns:
        --------
        np.ndarray
            Predicted class labels for the test data.
        """
        x_test = self.scaler.scaleNew(x_test)  # Scale test data
        output = self.sequence.predict(x_test)  # Get predictions

        # Convert predictions and true labels to categorical labels
        predicted_labels = np.argmax(output, axis=1)
        y_test = self.oneHotEncoding(y_test)
        true_labels = np.argmax(y_test, axis=1)

        # Initialize TP, FP, FN counters for each class
        true_positive = np.zeros(len(np.unique(true_labels)))
        false_positive = np.zeros(len(np.unique(true_labels)))
        false_negative = np.zeros(len(np.unique(true_labels)))

        # Calculate True Positives, False Positives, and False Negatives for each class
        for i in range(len(true_labels)):
            if predicted_labels[i] == true_labels[i]:
                true_positive[true_labels[i]] += 1
            else:
                false_positive[predicted_labels[i]] += 1
                false_negative[true_labels[i]] += 1

        # Calculate accuracy
        accuracy = np.sum(true_positive) / len(true_labels)

        # Calculate Precision, Recall, and F1-score for each class
        precision = true_positive / (true_positive + false_positive + 1e-10)
        recall = true_positive / (true_positive + false_negative + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        # Macro-averaged metrics
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1_score)

        # Print metrics
        print(f"Accuracy: {accuracy}")
        print(f"Precision per class: {precision}")
        print(f"Recall per class: {recall}")
        print(f"F1-score per class: {f1_score}")
        print(f"Avg Precision: {avg_precision}")
        print(f"Avg Recall: {avg_recall}")
        print(f"Avg F1-score: {avg_f1}")

        return predicted_labels

    def stepsPerEpoch(self):
        """
        Perform forward and backward propagation for all mini-batches in an epoch.
        """
        self.sequence.propagate()  # Perform one epoch of training
