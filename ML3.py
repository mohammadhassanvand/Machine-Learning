
import numpy as np
import math


# -------------------------
# 1. Artificial Neuron
# -------------------------

def neuron(inputs, weights, bias):
    z = sum(w * x for w, x in zip(weights, inputs)) + bias
    return 1 if z >= 0 else 0


weights = [3, 2, 1]

print("First Neuron")
print(neuron([1, 0, 1], weights, bias=-3))
print(neuron([0, 0, 1], weights, bias=-3))


# -------------------------
# 2. AND Gate
# -------------------------

print("\nAND Gate")

and_weights = [1, 1]
and_bias = -2

print(neuron([0, 0], and_weights, and_bias))
print(neuron([0, 1], and_weights, and_bias))
print(neuron([1, 0], and_weights, and_bias))
print(neuron([1, 1], and_weights, and_bias))


# -------------------------
# 3. OR Gate
# -------------------------

print("\nOR Gate")

or_weights = [1, 1]
or_bias = -1

print(neuron([0, 0], or_weights, or_bias))
print(neuron([0, 1], or_weights, or_bias))
print(neuron([1, 0], or_weights, or_bias))
print(neuron([1, 1], or_weights, or_bias))


# -------------------------
# 4. XOR Gate Attempt
# -------------------------

print("\nXOR Gate Attempt")

xor_weights = [1, 1]
xor_bias = -1

print(neuron([0, 0], xor_weights, xor_bias))
print(neuron([0, 1], xor_weights, xor_bias))
print(neuron([1, 0], xor_weights, xor_bias))
print(neuron([1, 1], xor_weights, xor_bias))


# -------------------------
# 5. Sigmoid Function
# -------------------------

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


print("\nSigmoid Function")
print(sigmoid(-5))
print(sigmoid(0))
print(sigmoid(5))


# -------------------------
# 6. Training a Neuron
# -------------------------

print("\nTraining a Neuron")

X = np.array([0, 1, 2, 3], dtype=float)
y = np.array([0, 0, 1, 1], dtype=float)

weight = 0.0
bias = 0.0
learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):

    # Forward pass
    z = weight * X + bias
    predictions = 1 / (1 + np.exp(-z))

    # Gradients for binary cross-entropy
    dw = np.mean((predictions - y) * X)
    db = np.mean(predictions - y)

    # Update parameters
    weight -= learning_rate * dw
    bias -= learning_rate * db

print("Weight:", weight)
print("Bias:", bias)

final_predictions = 1 / (
    1 + np.exp(-(weight * X + bias))
)

print("Predictions:", final_predictions)


# -------------------------
# 7. Multilayer Neural Network
# -------------------------

print("\nNeural Network for XOR")

X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y_xor = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

np.random.seed(42)

# Initialize weights and biases
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):

    # Forward pass
    z1 = X_xor @ W1 + b1
    a1 = 1 / (1 + np.exp(-z1))

    z2 = a1 @ W2 + b2
    predictions = 1 / (1 + np.exp(-z2))

    # Backpropagation
    dz2 = predictions - y_xor
    dW2 = a1.T @ dz2 / len(X_xor)
    db2 = np.mean(dz2, axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * a1 * (1 - a1)
    dW1 = X_xor.T @ dz1 / len(X_xor)
    db1 = np.mean(dz1, axis=0, keepdims=True)

    # Update parameters
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

print("Predictions:")
print(predictions.round(3))

print("Classifications:")
print((predictions >= 0.5).astype(int))

# -------------------------
# 7. Multilayer Neural Network
# -------------------------

print("\nNeural Network for XOR")

X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y_xor = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

np.random.seed(42)

# Initialize weights and biases
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):

    # Forward pass
    z1 = X_xor @ W1 + b1
    a1 = 1 / (1 + np.exp(-z1))

    z2 = a1 @ W2 + b2
    predictions = 1 / (1 + np.exp(-z2))

    # Backpropagation
    dz2 = predictions - y_xor
    dW2 = a1.T @ dz2 / len(X_xor)
    db2 = np.mean(dz2, axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * a1 * (1 - a1)
    dW1 = X_xor.T @ dz1 / len(X_xor)
    db1 = np.mean(dz1, axis=0, keepdims=True)

    # Update parameters
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

print("Predictions:")
print(predictions.round(3))

print("Classifications:")
print((predictions >= 0.5).astype(int))


# -------------------------
# 8. Evaluate the Network
# -------------------------

print("\nFinal Evaluation")

z1 = X_xor @ W1 + b1
a1 = 1 / (1 + np.exp(-z1))

z2 = a1 @ W2 + b2
final_predictions = 1 / (1 + np.exp(-z2))

print("Probabilities:")
print(final_predictions.round(3))

print("Classifications:")
print((final_predictions >= 0.5).astype(int))
