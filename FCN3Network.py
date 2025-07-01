from opt_einsum import contract
import torch.nn as nn
import torch
from activations import *
import standard_hyperparams as hp
from typing import Tuple, Optional, Callable, Dict

class FCN3NetworkEnsembleLinear(nn.Module):

    def __init__(self, d, n1, n2,P,ensembles=1, weight_initialization_variance=(1.0, 1.0, 1.0)):
        super().__init__()

        self.arch = [d, n1, n2]
        self.d = d
        self.n1 = n1
        self.n2 = n2
        self.W0 = nn.Parameter(torch.normal(mean=0.0, 
                                            std=torch.full((ensembles, n1, d), weight_initialization_variance[0]**0.5)).to(hp.DEVICE),
                                            requires_grad=True) # requires_grad moved here
        self.W1 = nn.Parameter(torch.normal(mean=0.0, 
                                            std=torch.full((ensembles, n2, n1), weight_initialization_variance[1]**0.5)).to(hp.DEVICE),
                                            requires_grad=True) # requires_grad moved here
        self.A = nn.Parameter(torch.normal(mean=0.0, 
                                           std=torch.full((ensembles, n2), weight_initialization_variance[2]**0.5)).to(hp.DEVICE),
                                           requires_grad=True) # requires_grad moved here


    def h1_activation(self, X):
        return contract(
            'ijk,ikl,ul->uij',
            self.W1, self.W0, X,
            backend='torch'
        )

    def h0_activation(self, X):
        return contract(
            'ikl,ul->uik',
            self.W0, X,
            backend='torch'
        )


    def forward(self, X):
        """

        Efficiently computes the outputs of a three layer network
        using opt_einsum

        f : P*d -> P*e*1
        C1_ui = W1_ijk*x_uk
        C2_uij = W2_ijk*C1_uik
        C3_ui = A_ij*C2_uij
        """
        print(X.shape)
        A = self.A.clone()
        W1 = self.W1.clone()
        W0 = self.W0.clone()
        return contract(
            'ij,ijk,ikl,unl->ui',
            A, W1, W0, X,
          backend='torch'
        )

class FCN3Network(nn.Module):
    """
    A base class for a three-layer fully connected neural network.

    Allows for specification of activation functions for each layer.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc3 (nn.Linear): The third fully connected layer (output layer).
        activation1 (Callable): The activation function for the first hidden layer.
        activation2 (Callable): The activation function for the second hidden layer.
    """

    @staticmethod
    def load_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> 'FCN3Network':
        """
        Loads the model from a state dictionary.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state dictionary containing model parameters.

        Returns:
            FCN3Network: An instance of the FCN3Network class.
        """
        model = FCN3Network(
            input_dim=state_dict['fc1.weight'].shape[1],
            hidden_width_1=state_dict['fc1.weight'].shape[0],
            hidden_width_2=state_dict['fc2.weight'].shape[0],
            activation1=state_dict.get('activation1', linear_activation),
            activation2=state_dict.get('activation2', linear_activation)
        )
        model.load_state_dict(state_dict)
        return model



    def __init__(
        self,
        input_dim: int,
        hidden_width_1: int,
        hidden_width_2: int,
        activation1: Callable[[torch.Tensor], torch.Tensor] = linear_activation,
        activation2: Callable[[torch.Tensor], torch.Tensor] = linear_activation
    ) -> None:
        """
        Initializes the Network.

        Args:
            input_dim (int): The dimensionality of the input (d).
            hidden_width_1 (int): The number of neurons in the first hidden layer (N^(0)).
            hidden_width_2 (int): The number of neurons in the second hidden layer (N^(1)).
            activation1 (Callable[[torch.Tensor], torch.Tensor]): The activation function for the first hidden layer.
            activation2 (Callable[[torch.Tensor], torch.Tensor]): The activation function for the second hidden layer.

        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_width_1)
        self.fc2 = nn.Linear(hidden_width_1, hidden_width_2)
        self.fc3 = nn.Linear(hidden_width_2, 1)
        self.activation1 = activation1
        self.activation2 = activation2

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        return super()._save_to_state_dict(destination, prefix, keep_vars)

    @staticmethod
    def model_from_hyperparameters(hyperparameters: Dict[str, float]) -> 'FCN3Network':
        sigma_1 = hyperparameters.get('weight_sigma1', 1.0)
        sigma_2 = hyperparameters.get('weight_sigma2', 1.0)
        sigma_3 = hyperparameters.get('weight_sigma3', 1.0)
        activation1 = hyperparameters.get('activation', linear_activation)
        activation2 = hyperparameters.get('output_activation', linear_activation)
        input_dim :int  = hyperparameters.get('input_dimension', 1)
        hidden_width_1:int= hyperparameters.get('hidden_width_1', 10)
        hidden_width_2 :int = hyperparameters.get('hidden_width_2', 10)
        model = FCN3Network(input_dim,hidden_width_1=hidden_width_1,hidden_width_2=hidden_width_2,activation1=activation1, activation2=activation2)
        model._reset_with_weight_sigma((sigma_1, sigma_2, sigma_3))
        return model


    def _reset_with_weight_sigma(self, weight_sigma : tuple = (1.0,1.0,1.0)) -> None:
        """
        Initializes the weights of the network from centered Gaussian distributions
        with the specified standard deviations.

        Args:
            sigma1 (float): Standard deviation for the first layer's weights.
            sigma2 (float): Standard deviation for the second layer's weights.
            sigma3 (float): Standard deviation for the third layer's weights.
        """
        with torch.no_grad():
            self.fc1.weight.data.normal_(0, weight_sigma[0])
            self.fc2.weight.data.normal_(0, weight_sigma[1])
            self.fc3.weight.data.normal_(0, weight_sigma[2])
            self.fc1.bias.data.zero_()
            self.fc2.bias.data.zero_()
            self.fc3.bias.data.zero_()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 1).
        """
        # Preactivation of the first layer
        self.h1: torch.Tensor = self.fc1(x)
        # Activation of the first layer
        self.a1: torch.Tensor = self.activation1(h1)
        # Preactivation of the second layer
        self.h2: torch.Tensor = self.fc2(a1)
        # Activation of the second layer
        self.a2: torch.Tensor = self.activation2(h2)
        # Preactivation of the output layer
        self.output: torch.Tensor = self.fc3(a2)
        # The output layer has a linear activation by default in this architecture.
        return self.output

if __name__ == '__main__':
    f = FCN3NetworkEnsembleLinear(5,10,10,100,ensembles=5)
    for name, p in f.named_parameters():
        print(name)
