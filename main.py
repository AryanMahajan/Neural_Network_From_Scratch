import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('train.csv').to_numpy()
np.random.shuffle(data)  # Shuffle the dataset

m, n = data.shape
train_size = 1000  # Validation set size

# Split into validation and training sets
X_val, Y_val = data[:train_size, 1:] / 255.0, data[:train_size, 0]
X_train, Y_train = data[train_size:, 1:] / 255.0, data[train_size:, 0]
m_train = X_train.shape[0]

# Neural Network Functions
def init_params() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1 = np.random.randn(10, 784) * 0.01  # Small initial weights
    b1 = np.zeros((10, 1))  # Bias initialized to 0
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def ReLU(Z: np.ndarray) -> np.ndarray:
    return np.maximum(0, Z)

def softmax(Z: np.ndarray) -> np.ndarray:
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, 
                 W2: np.ndarray, b2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z: np.ndarray) -> np.ndarray:
    return Z > 0

def one_hot(Y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    one_hot_Y = np.zeros((Y.size, num_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def back_prop(X: np.ndarray, Y: np.ndarray, Z1: np.ndarray, A1: np.ndarray, 
              Z2: np.ndarray, A2: np.ndarray, W2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    one_hot_Y = one_hot(Y)
    m = X.shape[1]

    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2 @ A1.T
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T @ dZ2 * ReLU_deriv(Z1)
    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, 
                  dW1: np.ndarray, db1: np.ndarray, dW2: np.ndarray, db2: np.ndarray, 
                  alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A: np.ndarray) -> np.ndarray:
    return np.argmax(A, axis=0)

def get_accuracy(predictions: np.ndarray, Y: np.ndarray) -> float:
    return np.mean(predictions == Y)

def gradient_descent(X: np.ndarray, Y: np.ndarray, alpha: float, 
                     iterations: int, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1, b1, W2, b2 = init_params()
    m = X.shape[1]

    for i in range(iterations):
        # Mini-batch gradient descent
        for j in range(0, m, batch_size):
            X_batch = X[:, j:j + batch_size]
            Y_batch = Y[j:j + batch_size]

            Z1, A1, Z2, A2 = forward_prop(X_batch, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = back_prop(X_batch, Y_batch, Z1, A1, Z2, A2, W2)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y_batch)
            print(f"Iteration {i}: Accuracy = {accuracy * 100:.2f}%")

    return W1, b1, W2, b2

# Train the model
W1, b1, W2, b2 = gradient_descent(X_train.T, Y_train, alpha=0.1, iterations=500)
