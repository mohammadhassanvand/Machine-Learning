import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -------------------------
# 1. Load Dataset
# -------------------------

digits = load_digits()

X = digits.data
y = digits.target

print("Data shape:", X.shape)
print("Labels shape:", y.shape)


# -------------------------
# 2. Split Dataset
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------
# 3. Create Neural Network
# -------------------------

model = MLPClassifier(
    hidden_layer_sizes=(64,),
    max_iter=1000,
    random_state=42
)


# -------------------------
# 4. Train Model
# -------------------------

model.fit(X_train, y_train)


# -------------------------
# 5. Make Predictions
# -------------------------

predictions = model.predict(X_test)

print("\nActual labels:", y_test[:10])
print("Predicted labels:", predictions[:10])


# -------------------------
# 6. Evaluate Model
# -------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# -------------------------
# 7. Inspect Neural Network Structure
# -------------------------

print("\nNetwork Structure")

print("Number of layers:", model.n_layers_)
print("Input features:", model.n_features_in_)
print("Output classes:", model.classes_)


# -------------------------
# 8. Inspect Weight Matrices
# -------------------------

print("\nWeight Matrices")

for i, weights in enumerate(model.coefs_):
    print(f"Layer {i + 1} weights shape:", weights.shape)


# -------------------------
# 9. Inspect Biases
# -------------------------

print("\nBias Shapes")

for i, bias in enumerate(model.intercepts_):
    print(f"Layer {i + 1} bias shape:", bias.shape)

# -------------------------
# 10. Visualize Predictions
# -------------------------

fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap="gray")

    ax.set_title(
        f"Actual: {y_test[i]}\nPredicted: {predictions[i]}"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()

# -------------------------
# 11. Find Wrong Predictions
# -------------------------

wrong_indices = []

for i in range(len(y_test)):
    if predictions[i] != y_test[i]:
        wrong_indices.append(i)


print("\nNumber of wrong predictions:", len(wrong_indices))
print("Wrong prediction indices", wrong_indices)

# -------------------------
# 12. Visualize Wrong Predictions
# -------------------------

if len(wrong_indices) > 0:
    fig, axes = plt.subplots(
        1,
        len(wrong_indices),
        figsize=(15, 3)
    )

    for ax, index in zip(axes, wrong_indices):
        ax.imshow(X_test[index].reshape(8, 8), cmap="gray")

        ax.set_title(
            f"Actual: {y_test[index]}\nPredicted: {predictions[index]}"
        )

        ax.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("No wrong predictions!")

# -------------------------
# 13. Prediction Probabilities
# -------------------------

sample = X_test[0].reshape(1, -1)

probabilities = model.predict_proba(sample)

print("\nPrediction Probabilities:")

for digit, probability in enumerate(probabilities[0]):
    print(f"Digit {digit}: {probability:.4f}")

print("\nPredicted Digit:", model.predict(sample)[0])

# -------------------------
# 14. Compare Hidden Layer Sizes
# -------------------------

hidden_sizes = [16, 32, 64, 128]

print("\nComparing Neural Network Architectures")

for size in hidden_sizes:
    test_model = MLPClassifier(
        hidden_layer_sizes=(size,),
        max_iter=1000,
        random_state=42
    )

    test_model.fit(X_train, y_train)

    test_predictions = test_model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    print(
        f"Hidden neurons: {size:3} | "
        f"Accuracy: {test_accuracy:.4f}"
    )


