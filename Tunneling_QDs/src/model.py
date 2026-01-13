"""
Model Module

CNN architecture definitions for quantum dot stability diagram analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class StabilityDiagramCNN(nn.Module):
    """
    Convolutional Neural Network for analyzing stability diagrams.
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 num_classes: int = 10,
                 dropout_rate: float = 0.5):
        """
        Initialize the CNN.
        
        Args:
            input_channels: Number of input channels (default: 1 for grayscale)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(StabilityDiagramCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Assuming input size of 128x128, after 4 pooling layers: 128/16 = 8
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class AutoencoderCNN(nn.Module):
    """
    Convolutional Autoencoder for stability diagram reconstruction.
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 latent_dim: int = 128):
        """
        Initialize the autoencoder.
        
        Args:
            input_channels: Number of input channels
            latent_dim: Dimension of the latent space
        """
        super(AutoencoderCNN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Latent space (for 128x128 input: 128/16 = 8)
        self.fc_encode = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 8 * 8)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstruction, latent_representation)
        """
        # Encode
        encoded = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)
        latent = self.fc_encode(encoded_flat)
        
        # Decode
        decoded_flat = self.fc_decode(latent)
        decoded = decoded_flat.view(decoded_flat.size(0), 256, 8, 8)
        reconstruction = self.decoder(decoded)
        
        return reconstruction, latent


def get_model(model_type: str = 'classifier', **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('classifier' or 'autoencoder')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model
    """
    if model_type == 'classifier':
        return StabilityDiagramCNN(**kwargs)
    elif model_type == 'autoencoder':
        return AutoencoderCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
