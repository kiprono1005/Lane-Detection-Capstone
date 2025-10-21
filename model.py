"""
PilotNet CNN Architecture for Autonomous Steering
Based on NVIDIA's End-to-End Learning for Self-Driving Cars
Modified for lane keeping task

Author: Kip Chemweno
Course: ECE 4424 - Machine Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PilotNet(nn.Module):
    """
    Modified PilotNet architecture:
    - 1 normalization layer
    - 5 convolutional layers
    - 3 fully connected layers

    Input: RGB image (3, 66, 200)
    Output: Steering angle (continuous value)
    """

    def __init__(self, dropout_rate: float = 0.5):
        super(PilotNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 1 * 18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

        # Dropout and batch normalization
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(36)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)

    def forward(self, x):
        """Forward pass"""
        # Convolutional layers with ELU activation
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.elu(self.bn4(self.conv4(x)))
        x = F.elu(self.bn5(self.conv5(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        x = F.elu(self.fc3(x))
        x = self.fc4(x)

        return x


class SimplifiedPilotNet(nn.Module):
    """Simplified version without batch normalization."""

    def __init__(self, dropout_rate: float = 0.5):
        super(SimplifiedPilotNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 1 * 18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing PilotNet architecture...")
    model = PilotNet()
    x = torch.randn(4, 3, 66, 200)
    output = model(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Parameters: {count_parameters(model):,}")