import torch
import torch.nn as nn
import standard_hyperparams as hp
from typing import Callable, Dict


class SimpleNet(nn.Module):
    """
    A base class for a one-layer fully connected neural network.
    Allows for specification of activation functions for the layer.
    Attributes:
        fc1 (nn.Linear): The fully connected layer.
        activation (Callable): The activation function for the layer.
    """
    def __init__(self, 
                 input_dim: int,
                 weight_sigma: float
                 ) -> None:
                 
        """
        Initializes the Network.
        Args:
            input_dim (int): The dimensionality of the input (d).
            activation (Callable[[torch.Tensor], torch.Tensor]): The activation function for the layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1)
        self.weight_sigma : float = weight_sigma
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initializes the weights of the network from a centered Gaussian distribution
        with the specified standard deviation.
        Args:
            sigma (float): Standard deviation for the layer's weights.
        """
        weight_sigma = self.weight_sigma
        with torch.no_grad():
            self.fc1.bias.data.zero_()
            self.fc1.weight.data.zero_() # First, set all elements to zero
            self.fc1.weight.data[0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 1).
        """
        # Preactivation of the layer
        h1: torch.Tensor = self.fc1(x)

        return h1
