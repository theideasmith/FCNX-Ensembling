# FCN3.py (Revised)
import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset, DataLoader, RandomSampler
from torch.utils.data import TensorDataset, random_split
import math
from activations import *

import datetime

import torch
import torch.nn as nn
from typing import Dict, List, Callable
import standard_hyperparams as hp

from typing import Callable, Dict
import time
from datetime import datetime
from rich.progress import Progress, TaskID
from logger import Logger
from check_mps import check_mps_available_or_fail

from layeroutputtracker import *
from simplenet import *
from datamanager import *
from FCN3Network import FCN3Network

class TrainingInfo:
    def __init__(self):
        self.trainloss = []
        self.testloss = []
        self.times = []
        self.losses = []
        self.epochs = []

    def reset(self):
        self.trainloss = []
        self.testloss = []
        self.times = []

    def update_loss(self, timestep, trainloss : float = None, testloss : float = None):

        self.trainloss.append(trainloss)

        if testloss is not None:
            self.testloss.append(testloss)
        self.times.append(timestep)

    def get_last_loss(self):
        return self.losses[-1] if self.losses else None

    def get_average_loss(self):
        return sum(self.losses) / len(self.losses) if self.losses else 0

    def get_time_steps_per_epoch(self):
        if not self.epochs:
            return {}
        epoch_steps = {}
        last_epoch = -1
        start_step = 0
        for i in range(len(self.epochs)):
            epoch = self.epochs[i]
            current_step = self.times[i]
            if epoch != last_epoch:
                if last_epoch != -1:
                    epoch_steps[last_epoch] = current_step - start_step
                last_epoch = epoch
                start_step = current_step
        epoch_steps[last_epoch] = self.times[len(self.times)-1] - start_step # Account for the last epoch
        return epoch_steps

    def display_summary(self):
        if self.losses:
            print("\n--- Training Summary ---")
            print(f"Total Epochs: {self.epochs[-1]}")
            print(f"Final Loss: {self.losses[-1]:.4f}")
            print(f"Average Loss: {self.get_average_loss():.4f}")
            print("Time Steps per Epoch:")
            for epoch, steps in self.get_time_steps_per_epoch().items():
                print(f"  Epoch {epoch}: {steps} steps")
            print(f"Total Training Time Steps: {self.times[len(self.times)-1]}")
        else:
            print("No training information recorded yet.")

class NetworkTrainer:
    def __init__(self, model: nn.Module,
                 manager: DataManager,
                 batch_size: int = hp.BATCH_SIZE,
                 learning_rate: float = hp.LEARNING_RATE,
                 weight_decay_config: dict = {},
                 num_epochs: int = hp.NUM_EPOCHS
                 ):

        """
        Initializes the NetworkTrainer.

        Args:
            model (nn.Module): The neural network model to be trained.
            data (Dataset): The dataset containing training data.
            batch_size (int): The size of each training batch.
            learning_rate (float): The learning rate for the optimizer.
            weight_decay_config (dict): A dictionary specifying weight decay for each parameter.
            num_epochs (int): The number of epochs for training.
        """
        self.model : nn.Module = model
        self.manager : DataManager = manager
        self.batch_size : int = batch_size
        self.dataloader : DataLoader = DataLoader(manager, batch_size=batch_size, shuffle=True)
        self.train_config : dict = {
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
        }
        self.weight_decay_config : dict = weight_decay_config

        self.training_info : TrainingInfo = TrainingInfo()
        self.current_epoch : int = 0
        self.current_time_step : int = 0
        self.converged : bool = False

    def warp_to_epoch(self, nepch):
        self.current_epoch = warp_to_epoch

    def train(self,
              continue_at_epoch: int = 0,
              current_time_step: int = 0,
              logger: Logger = None, 
              reinitialize : bool = True, interrupt_callback = None,completion_callback = None, epoch_callback = None) -> nn.Module:
        """
        Trains the neural network model using Langevin dynamics.

        The training process involves iterating over the dataset, computing the loss,
        and updating the model parameters using Langevin dynamics.
        """
        self.manager.mode = 'train' # Set the data manager to training mode
        # so that it will access the train data rather than the test data
        self.model.train()  # Set the model to training mode
        # Iterate over the specified number of epochs
        self.current_epoch = continue_at_epoch
        self.current_time_step = current_time_step
        self.current_train_loss = self.get_current_train_loss()

        try:
            while self.current_epoch < self.train_config['num_epochs']:
                for batch in self.dataloader:
                    data, targets = batch
                    self.param_update(data, targets)
                    self.current_time_step += 1
                if (epoch_callback is not None):
                    try:
                        epoch_callback(self)
                    except Exception as e:
                        print(f'An error occurred {e}')

                self.current_epoch += 1
                self.training_info.update_loss(self.current_time_step, self.current_train_loss)
                if self.current_time_step%10==0:
                        logger.epoch_callback(self)
            self.converged = True
        except KeyboardInterrupt:
            if interrupt_callback is not None:
                interrupt_callback(self)
            raise KeyboardInterrupt("Training Interrupted; Quitting")

        if completion_callback is not None:
            completion_callback(self)
        self.training_complete()
        print("Reset training state")
        self.converged = True
        print("Asserting converged")
        return self.model

    def training_complete(self):
        """
        Finalizes the training process and displays the training summary.
        """
        self.manager.mode = 'test'
        self.model.eval()  # Set the model to evaluation mode
        self.current_epoch = 0
        self.current_time_step = 0

    def get_current_train_loss(self):
        self.model.eval() # Set model to evaluation mode
        self.manager.mode = 'test'
        with torch.no_grad():
            data = self.manager.data
            targets = self.manager.targets
            data = data.to(hp.DEVICE)
            targets = targets.to(hp.DEVICE)

            # Clear previous gradients
            self.model.zero_grad()

            # Ensure data and targets are on the correct device
            data = data.to(hp.DEVICE)
            targets = targets.to(hp.DEVICE)

            # Forward pass
            outputs: torch.Tensor = self.model(data)

            # Calculate the Mean Squared Error loss
            loss: torch.Tensor = self.loss_function(outputs, targets)    
            for name, param in self.model.named_parameters():
                loss = loss + (0.5 * torch.einsum('...k,...k->', param, param) * self.weight_decay_config[name] )
            self.model.train()
            self.manager.mode = 'train'

        return loss

    def param_update(self, data: torch.Tensor, targets: torch.Tensor) -> None:
        outputs: torch.Tensor = self.model(data)

        self.model.zero_grad()

        try:

            # Calculate the Mean Squared Error loss
            loss: torch.Tensor = self.loss_function(outputs, targets)
            self.current_train_loss = loss.detach().cpu().numpy()
            loss.backward()
        except Exception as e:
            print(e)
            exit()

        # Update parameters with no gradient tracking
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.weight_update_function(
                    param=param,
                    param_name=name)
        
        self.model.zero_grad()


    def loss_function(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Mean Squared Error (MSE) loss.

        Args:
            outputs (torch.Tensor): The model's predictions.
            targets (torch.Tensor): The true target values.

        Returns:
            torch.Tensor: The computed MSE loss.
        """
        loss = 0.5 * torch.sum(torch.square((outputs - targets)))
        return loss

    def weight_update_function(self,param: torch.Tensor, param_name: str):
        """
        Performs a Langevin update on the parameter with weight decay.

        Args:
            param (torch.Tensor): The parameter to be updated.
            grad (torch.Tensor): The gradient of the loss with respect to the parameter.
            step_size (float): The learning rate for the Langevin update.
            noise_std (float): The standard deviation of the Gaussian noise.
            weight_decay (float): The weight decay coefficient.
        """
        η = self.train_config['learning_rate']
        return param.data.add_(-η * param.grad - η * 2 * self.weight_decay_config[param_name] * param)

class LangevinTrainer(NetworkTrainer):
    """
    A specialized trainer for Langevin dynamics, inheriting from NetworkTrainer.
    This class is designed to handle the Langevin update process specifically.
    """
    def __init__(self, model: nn.Module,
                 manager: DataManager,
                 batch_size: int = hp.BATCH_SIZE,
                 learning_rate: float = hp.LEARNING_RATE,
                 noise_std: float = hp.NOISE_STD_LANGEVIN,
                 weight_decay_config: dict = {},
                 num_epochs: int = hp.NUM_EPOCHS,
                 on_data: bool = True
                 ):


        super().__init__(model, manager, batch_size, learning_rate, weight_decay_config, num_epochs)
        self.train_config['noise_std'] = noise_std
        self.train_config['on_data'] = on_data

    def weight_update_function(self, param: torch.Tensor, param_name: str) -> torch.Tensor:
        """
        Performs a Langevin update on the parameter with weight decay.

        Args:
            param (torch.Tensor): The parameter to be updated.
            grad (torch.Tensor): The gradient of the loss with respect to the parameter.
            step_size (float): The learning rate for the Langevin update.
            noise_std (float): The standard deviation of the Gaussian noise.
            weight_decay (float): The weight decay coefficient.
        """
        # noise is defined as a random tensor with the same shape as param
        # In langevin dynamics the noise is given with the expression:
        # sqrt(2η) ξ_t
        # where η is the learning rate and ξ_t is Gaussian noise with zero mean and unit variance.
        # So that the total update rule is:
        # θ_{t+1} = θ_t - η ∇L(θ_t) - 2*η*λ*θ_t  - sqrt(2η) ξ_t
        noise : float = torch.randn_like(param) * self.train_config['noise_std']
        η = self.train_config['learning_rate']
        gd = param.grad if self.train_config['on_data'] is True else 0.0
        return param.data.add_(-η * gd - η * self.weight_decay_config[param_name] * param + noise) 
