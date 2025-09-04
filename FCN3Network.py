from opt_einsum import contract, contract_path
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



class FCN3NetworkEnsembleErf(nn.Module):

    def __init__(self, d, n1, n2,P,ens=1, weight_initialization_variance=(1.0, 1.0, 1.0), device=hp.DEVICE):
        super().__init__()

        self.arch = [d, n1, n2]
        self.d = d
        self.n1 = n1
        self.n2 = n2
        self.ens = ens
        self.num_samples = P
        self.device = device
        self.W0 = nn.Parameter(torch.normal(mean=0.0,  std=torch.full((ens, n1, d), weight_initialization_variance[0]**0.5)).to(device),
                                            requires_grad=True) # requires_grad moved here

        # self._h1_buffer = nn.Parameter(torch.zeros((P, ensembles, n1)).to(device), requires_grad=False)
        # self._h2_buffer = nn.Parameter(torch.zeros((P, ensembles, n2)).to(device), requires_grad=False)
        # self._f_buffer = nn.Parameter(torch.zeros((P,ensembles)).to(device), requires_grad=False)

        self.W1 = nn.Parameter(torch.normal(mean=0.0, 
                                            std=torch.full((ens, n2, n1), weight_initialization_variance[1]**0.5)).to(device),
                                            requires_grad=True) # requires_grad moved here
        self.A = nn.Parameter(torch.normal(mean=0.0, 
                                           std=torch.full((ens, n2), weight_initialization_variance[2]**0.5)).to(device),
                                           requires_grad=True) # requires_grad moved here
        if self.num_samples is not None:
            self._precompute_einsum_paths_h1(self.num_samples)
            self._precompute_einsum_paths_h0(self.num_samples)
            self._precompute_einsum_paths_f(self.num_samples)

    def _precompute_einsum_paths_f(self, num_samples):
        eq = 'qk,uqk->uq'
        shapes = [
            (self.ens, self.n2),
            (num_samples, self.ens, self.n2)
        ]
        dummy_tensors = [torch.empty(s, device=self.device, dtype=torch.float64) for s in shapes]     
        path, _ = contract_path(eq, *dummy_tensors)
        self.forward_path_f = path

    def _precompute_einsum_paths_h1(self, num_samples):
        eq = 'qjk,uqj->uqk'
        shapes = [
            (self.ens, self.n1, self.n2),
            (num_samples, self.ens, self.n1)
        ]
        dummy_tensors = [torch.empty(s, device=self.device, dtype=torch.float64) for s in shapes]
        path, _ = contract_path(eq, *dummy_tensors)
        self.forward_path_h1 = path

    def _precompute_einsum_paths_h0(self, num_samples):
        eq = 'qkl,ul->uqk'
        shapes = [
            (self.ens, self.n1, self.d),
            (num_samples, self.d)
        ]
        dummy_tensors = [torch.empty(s, device=self.device, dtype=torch.float64) for s in shapes]
        path, _ = contract_path(eq, *dummy_tensors)
        self.forward_path_h0 = path


    def h1_GP_preactivation(self, X):
        h0 = contract(
            'qkl,ul->uqk',
            self.W0, X,
            optimize=self.forward_path_h0 if self.forward_path_h0 is not None else None,
          backend='torch'
        )
        h1_pre = contract(
            'qjk,uqj->uqk',self.W1,h0,
            optimize=self.forward_path_h1 if self.forward_path_h1 is not None else None,
            backend='torch'
        )
        return h1_pre

    def H_GP_eig(self, X,Y):

        h1 = self.h1_GP_preactivation(X)

        h1_kernel = contract('ul, uqk,  vqk, vl->l', Y, h1, h1, Y, backend='torch') / contract('ul, ul->l', Y, Y) 

        ret =  h1_kernel / (self.ens * self.n1 * X.shape[0])

        return ret

    def h1_preactivation(self, X):
        h0 = torch.erf(contract(
            'qkl,ul->uqk',
            self.W0, X,
            optimize=self.forward_path_h0 if self.forward_path_h0 is not None else None,
          backend='torch'
        ))
        h1_pre = contract(
            'qjk,uqj->uqk',self.W1,h0,
            optimize=self.forward_path_h1 if self.forward_path_h1 is not None else None,
            backend='torch'
        )
        return h1_pre

    def h1_activation(self, X):
       
      
        h0 = torch.erf(contract(
            'qkl,ul->uqk',
            self.W0, X,
            optimize=self.forward_path_h0 if self.forward_path_h0 is not None else None,
          backend='torch'
        ))

        h1 = torch.erf(contract(
            'qjk,uqj->uqk',self.W1,h0,
            optimize=self.forward_path_h1 if self.forward_path_h1 is not None else None,
            backend='torch'
        ))
        return h1

    def J_eig(self, X, Y):
        J = contract('qkl,ul->uqk',self.W0, X,optimize=self.forward_path_h0 if self.forward_path_h0 is not None else None,backend='torch')
        J_k = contract('ul, uqk,  vqk, vl->l', Y, J, J, Y, backend='torch') / contract('ul, ul->l', Y, Y)
        lJ =  J_k / (self.ens * self.n1 * X.shape[0]);
        return lJ
    def H_eig(self,X, Y, std=False):


        if std is False:
            h1 = self.h1_preactivation(X)

            h1_kernel = contract('ul, uqk,  vqk, vl->l', Y, h1, h1, Y, backend='torch') / contract('ul, ul->l', Y, Y) 


            ret =  h1_kernel / (self.ens * self.n1 * X.shape[0])
            return ret
        else:
            h1 = self.h1_preactivation(X)

            h1_kernel = contract('ul, uqk,  vqk, vl->kql', Y, h1, h1, Y, backend='torch') / contract('ul, ul->l', Y, Y) / X.shape[0]
            print(h1_kernel.shape)

            ls = torch.mean(h1_kernel, dim=(0,1))
            std = torch.std(h1_kernel, dim=(0,1))
            return ls, std

    def forward(self, X):
        """

        Efficiently computes the outputs of a three layer network
        using opt_einsum

        f : P*d -> P*e*1
        C1_ui = W1_ijk*x_uk
        C2_uij = W2_ijk*C1_uik
        C3_ui = A_ij*C2_uij
        """
    
 

        h0 = torch.erf(contract(
            'qkl,ul->uqk',
            self.W0, X,
            optimize=self.forward_path_h0 if self.forward_path_h0 is not None else None,
          backend='torch'
        ))

        h1 = torch.erf(contract(
            'qjk,uqj->uqk',self.W1,h0,
            optimize=self.forward_path_h1 if self.forward_path_h1 is not None else None,
            backend='torch'
        ))

        f = contract(
            'qk,uqk->uq',
            self.A,h1,
            optimize=self.forward_path_f if self.forward_path_f is not None else None,
            backend='torch'
        ).unsqueeze(1)

        return f

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
