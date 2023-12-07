import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from mnist_model import MnistModel
from mnist_dataloader import MnistDataloader

# Set file paths
current_file_dir = os.path.dirname(os.path.abspath(__file__))
training_images_filepath = os.path.join(current_file_dir, 'input', 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = os.path.join(current_file_dir, 'input', 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = os.path.join(current_file_dir, 'input', 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = os.path.join(current_file_dir, 'input', 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# Load the data
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Convert to PyTorch tensors
x_train, y_train, x_test, y_test = map(
    lambda x: torch.tensor(np.array(x)), (x_train, y_train, x_test, y_test)
)

# Normalize the images
x_train = x_train.float() / 255.0
x_test = x_test.float() / 255.0

# Reshape for CNN input
x_train = x_train.unsqueeze(1)  # Add channel dimension
x_test = x_test.unsqueeze(1)

# Create DataLoader instances
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Initialize the model
model = MnistModel()

# Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    for data, target in tqdm(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            total_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(x_test)
    accuracy = correct / len(x_test)
    print(f"Epoch: {epoch}, Test loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

# Save the model
model_save_path = os.path.join(current_file_dir, 'models', 'mnist_model.pth')
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
