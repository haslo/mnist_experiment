# inference.py
import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from mnist_model import MnistModel


def load_model(model_path):
    model = MnistModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    return transform(image)


def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(outputs, 1)
        return predicted.item()


def show_preprocessed_image(image_tensor, title_text):
    plt.figure(figsize=(5, 5))
    plt.imshow(image_tensor.squeeze(), cmap='gray')  # Remove channel dimension and show in grayscale
    if title_text != '':
        plt.title(title_text, fontsize=15)
    plt.show()


# Load the model
model_path = './models/mnist_model.pth'
model = load_model(model_path)

# Load and preprocess the image
image_path = '/Users/haslo/Desktop/input.png'
image_tensor = preprocess_image(image_path)

# Predict and display the result
predicted_digit = predict(model, image_tensor)
print(f'Predicted Digit: {predicted_digit}')
show_preprocessed_image(image_tensor, f'Predicted Digit: {predicted_digit}')
