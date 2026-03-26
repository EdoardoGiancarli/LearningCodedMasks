"""
Module with basic mock models architectures for testing `spark`.
Contains:
    * basic CNN model
    * basic autoencoder model
"""

import torch
import torch.nn as nn
from torch.types import Tensor


# _________________________________  BASIC CNN MODEL  _________________________________ #

def get_conv_block(
    in_dims: int,
    out_dims: int,
    kernel_size: int,
    padding: int,
) -> nn.ModuleList:
    """Defines baseline conv block for mock model."""
    block = [
        nn.Conv2d(in_dims, out_dims, kernel_size, padding=padding),
        nn.BatchNorm2d(out_dims),
        nn.ReLU(),
    ]
    return nn.ModuleList(block)


class MockModel(nn.Module):
    """
    Baseline mock CNN model for `spark` API tests.
    """
    def __init__(
        self,
        data_shape: torch.Size,
        out_features: int,
        maxpool: int = 2,
        dropout: float = 0.3, 
    ) -> None:
        super().__init__()
        in_dim = int(data_shape[0])
        in_features = int(data_shape.numel() / pow(maxpool, 2))
        self.net = nn.Sequential(
            *get_conv_block(in_dim, 16, 7, 3),
            *get_conv_block(16, 16, 5, 2),
            *get_conv_block(16, in_dim, 3, 1),
            nn.MaxPool2d(maxpool),
            nn.Flatten(),
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(1024, out_features),
            nn.Softmax(1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        return out




# _________________________________  BASIC AUTOENCODER MODEL  _________________________________ #

class Encoder(nn.Module):
    """Baseline mock encoder model."""
    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.architecture = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, latent_dim, kernel_size=4, stride=1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.architecture(x)
        return out


class Decoder(nn.Module):
    """Baseline mock encoder model."""
    def __init__(self, latent_dim: int, out_channels: int) -> None:
        super().__init__()
        self.architecture = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=5, stride=1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.architecture(x)
        return out


class MockAutoEncoder(nn.Module):
    """
    Baseline mock AutoEncoder model for `spark` API tests and stuff.
    """
    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        embedded = self.encoder(x)
        out = self.decoder(embedded)
        return out


# end