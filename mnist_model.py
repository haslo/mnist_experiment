import torch.nn as nn
import torch.nn.functional as F

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        # First convolutional layer: 1 input channel (grayscale image), 32 output channels, kernel size 3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional layer: 32 input channels, 64 output channels, kernel size 3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling
        self.max_pool = nn.MaxPool2d(2)
        # Fully connected layer
        self.fc = nn.Linear(64 * 7 * 7, 10)  # 10 output classes for MNIST digits

    def forward(self, x):
        # Apply first convolution, followed by ReLU, then max pooling
        x = self.max_pool(F.relu(self.conv1(x)))
        # Apply second convolution, followed by ReLU, then max pooling
        x = self.max_pool(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        # Fully connected layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
