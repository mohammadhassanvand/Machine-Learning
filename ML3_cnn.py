
import numpy as np

# -------------------------
# 0. Create Image and Kernel
# -------------------------

image = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 1],
    [7, 8, 9, 2],
    [0, 1, 2, 3]
])

kernel = np.array([
    [1, 0],
    [0, -1]
])

print("Image:")
print(image)

print("\nKernel:")
print(kernel)


# -------------------------
# 1. Convolution
# -------------------------

image_height, image_width = image.shape
kernel_height, kernel_width = kernel.shape

output_height = image_height - kernel_height + 1
output_width = image_width - kernel_width + 1

conv_output = np.zeros((output_height, output_width))

for i in range(output_height):
    for j in range(output_width):

        region = image[
            i:i + kernel_height,
            j:j + kernel_width
        ]

        conv_output[i, j] = np.sum(region * kernel)

print("\nConvolution Output:")
print(conv_output)


# -------------------------
# 2. ReLU Activation
# -------------------------

relu_output = np.maximum(0, conv_output)

print("\nReLU Output:")
print(relu_output)


# -------------------------
# 3. Max Pooling
# -------------------------

pool_size = 2
stride = 1

height, width = relu_output.shape

pool_height = (height - pool_size) // stride + 1
pool_width = (width - pool_size) // stride + 1

pool_output = np.zeros((pool_height, pool_width))

for i in range(pool_height):
    for j in range(pool_width):

        start_row = i * stride
        start_col = j * stride

        region = relu_output[
            start_row:start_row + pool_size,
            start_col:start_col + pool_size
        ]

        pool_output[i, j] = np.max(region)

print("\nMax Pooling Output:")
print(pool_output)


# -------------------------
# 4. Multiple Filters
# -------------------------

kernels = [
    np.array([
        [1, 0],
        [0, -1]
    ]),

    np.array([
        [1, 1],
        [-1, -1]
    ])
]

for index, current_kernel in enumerate(kernels, start=1):

    kh, kw = current_kernel.shape

    feature_height = image.shape[0] - kh + 1
    feature_width = image.shape[1] - kw + 1

    feature_map = np.zeros((feature_height, feature_width))

    for i in range(feature_height):
        for j in range(feature_width):

            region = image[i:i + kh, j:j + kw]

            feature_map[i, j] = np.sum(
                region * current_kernel
            )

    print(f"\nFilter {index}:")
    print(current_kernel)

    print("Feature Map:")
    print(feature_map)

    filter_relu = np.maximum(0, feature_map)

    print("ReLU:")
    print(filter_relu)


# -------------------------
# 5. Reusable Convolution Function
# -------------------------

def convolution(input_image, filter_kernel, stride=1):

    image_height, image_width = input_image.shape
    kernel_height, kernel_width = filter_kernel.shape

    output_height = (
        (image_height - kernel_height) // stride
    ) + 1

    output_width = (
        (image_width - kernel_width) // stride
    ) + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):

            start_row = i * stride
            start_col = j * stride

            region = input_image[
                start_row:start_row + kernel_height,
                start_col:start_col + kernel_width
            ]

            output[i, j] = np.sum(region * filter_kernel)

    return output


result = convolution(image, kernel)

print("\nReusable Convolution:")
print(result)


# -------------------------
# 6. Zero Padding
# -------------------------

padded_image = np.pad(
    image,
    pad_width=1,
    mode="constant",
    constant_values=0
)

print("\nOriginal Image:")
print(image)

print("\nPadded Image:")
print(padded_image)

print("\nOriginal Shape:", image.shape)
print("Padded Shape:", padded_image.shape)

padded_result = convolution(padded_image, kernel)

print("\nConvolution with Zero Padding:")
print(padded_result)

print("Output Shape:", padded_result.shape)


# -------------------------
# 7. Same Padding
# -------------------------

def convolution_same(input_image, filter_kernel):

    kh, kw = filter_kernel.shape

    pad_top = (kh - 1) // 2
    pad_bottom = kh - 1 - pad_top

    pad_left = (kw - 1) // 2
    pad_right = kw - 1 - pad_left

    padded = np.pad(
        input_image,
        (
            (pad_top, pad_bottom),
            (pad_left, pad_right)
        ),
        mode="constant",
        constant_values=0
    )

    return convolution(padded, filter_kernel)


kernel3 = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

same_output = convolution_same(image, kernel3)

print("\nSame Padding Output:")
print(same_output)

print("Output Shape:", same_output.shape)


# -------------------------
# 8. Complete CNN Pipeline
# -------------------------

# Step 1: Convolution with Same Padding
cnn_conv = convolution_same(image, kernel3)


# Step 2: ReLU
cnn_relu = np.maximum(0, cnn_conv)


# Step 3: Max Pooling
pool_size = 2
stride = 1

height, width = cnn_relu.shape

pool_height = (height - pool_size) // stride + 1
pool_width = (width - pool_size) // stride + 1

cnn_pool = np.zeros((pool_height, pool_width))

for i in range(pool_height):
    for j in range(pool_width):

        start_row = i * stride
        start_col = j * stride

        region = cnn_relu[
            start_row:start_row + pool_size,
            start_col:start_col + pool_size
        ]

        cnn_pool[i, j] = np.max(region)


# Step 4: Flatten
flattened = cnn_pool.flatten()


# Step 5: Dense Layer
weights = np.array([
    0.2, 0.3, 0.4,
    0.1, 0.2, 0.1,
    0.3, 0.2, 0.1
])

bias = 0.5

dense_output = np.dot(flattened, weights) + bias


# Step 6: Display Complete Pipeline
print("\n========== Complete CNN Pipeline ==========")

print("\nConvolution:")
print(cnn_conv)

print("\nReLU:")
print(cnn_relu)

print("\nMax Pooling:")
print(cnn_pool)

print("\nFlatten:")
print(flattened)

print("\nFlatten Shape:", flattened.shape)
print("Weights Shape:", weights.shape)

print("\nDense Output:", dense_output)

print("\n========== Program Finished Successfully ==========")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

prediction = sigmoid(dense_output)

print("\nPrediction Probability:", prediction)


import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


# ==========================================
# 1. Settings
# ==========================================

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cifar10_mlp.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("PyTorch version:", torch.__version__)


# ==========================================
# 2. Download and load CIFAR-10
# ==========================================

transform = transforms.ToTensor()

train_data = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_data = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

classes = train_data.classes

print("Training images:", len(train_data))
print("Testing images:", len(test_data))
print("Classes:", classes)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=512,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 3. Build the neural network
# ==========================================

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3072, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

model = model.to(device)

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters())
)


# ==========================================
# 4. Loss function and optimizer
# ==========================================

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 5. Train the model
# ==========================================

print("\nStarting training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        scores = model(images)

        # Calculate loss
        loss = loss_fn(scores, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item() * len(labels)

    average_loss = total_loss / len(train_data)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Loss: {average_loss:.4f}"
    )


# ==========================================
# 6. Evaluate the model
# ==========================================

model.eval()

correct = 0
total = 0

class_correct = [0] * 10
class_total = [0] * 10

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        scores = model(images)
        predictions = scores.argmax(dim=1).cpu()

        correct += (predictions == labels).sum().item()
        total += len(labels)

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

        for label, prediction in zip(labels, predictions):
            class_total[label.item()] += 1

            if label.item() == prediction.item():
                class_correct[label.item()] += 1

accuracy = correct / total

print("\nEvaluation complete.")
print("Correct predictions:", correct, "/", total)
print(f"Test accuracy: {accuracy:.3f} ({accuracy * 100:.2f}%)")


# ==========================================
# 7. Accuracy for each class
# ==========================================

print("\nAccuracy by class:")

for i, class_name in enumerate(classes):
    if class_total[i] > 0:
        class_accuracy = class_correct[i] / class_total[i]

        print(
            f"{class_name:12s}: "
            f"{class_accuracy * 100:.2f}%"
        )


# ==========================================
# 8. Show some predictions
# ==========================================

print("\nSample predictions:")

for i in range(min(10, len(all_labels))):
    actual = classes[all_labels[i]]
    predicted = classes[all_predictions[i]]

    result = "Correct" if actual == predicted else "Wrong"

    print(
        f"{i + 1}. Actual: {actual:12s} "
        f"| Predicted: {predicted:12s} "
        f"| {result}"
    )


# ==========================================
# 9. Save the trained model
# ==========================================

torch.save(model.state_dict(), MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
print("PROJECT COMPLETED")
