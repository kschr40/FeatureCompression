import torch
import torch.nn as nn
from typing import List

class MultiLayerPerceptron(nn.Module):
    """Multi-layer perceptron with configurable layer sizes.

    The network consists of fully connected layers with sizes specified by n_neurons,
    with ReLU activation between layers.
    """
    def __init__(self, n_neurons: List[int], activation = nn.ReLU, dropout: float = 0.0) -> None:
        """Initialize the MLP model.

        Args:
            n_neurons (List[int]): List specifying number of neurons in each layer
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
        """
        super().__init__()
        
        layers = []
        for i in range(len(n_neurons)-1):
            layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
            if i < len(n_neurons)-2:  # Don't add activation after last layer
                layers.append(activation())
                layers.append(nn.Dropout(p=dropout))
                
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_neurons[0])

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_neurons[-1])
        """
        return self.network(x)
    