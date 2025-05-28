import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader, RandomSampler
from torch.utils.data import TensorDataset, random_split
from typing import Dict, List, Callable
from typing import Tuple, Optional, Callable

import standard_hyperparams as hp

class RestrictedValue:
    def __init__(self, initial_value, allowed_values):
        self._allowed_values = set(allowed_values)
        self._validate(initial_value)
        self._value = initial_value

    def _validate(self, value):
        if value not in self._allowed_values:
            raise ValueError(
                f"Invalid value: '{value}'. Must be one of: {', '.join(map(repr, self._allowed_values))}"
            )

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._validate(new_value)
        self._value = new_value

    def __repr__(self):
        return f"{self.__class__.__name__}(value={repr(self._value)}, allowed_values={repr(tuple(self._allowed_values))})"

    def __str__(self):
        return str(self._value)

    def __eq__(self, other):
        if isinstance(other, RestrictedValue):
            return self._value == other._value and self._allowed_values == other._allowed_values
        return self._value == other

# Subclass for TrainingMode
class TrainingMode(RestrictedValue):
    ALLOWED = {'train', 'test', 'init'}

    def __init__(self, initial_value='init'):
        super().__init__(initial_value, TrainingMode.ALLOWED)

class DataManager(TensorDataset):
    """
    A base class to hold training data and target outputs.

    Attributes:
        X (torch.Tensor): A matrix of input data, where each row represents a datapoint.
                         Shape: (P, d), where P is the number of datapoints and d is the input dimension.
        Y (torch.Tensor): A matrix of target output data for each input datapoint.
                         Shape: (P, output_dim), where output_dim is the dimension of the output.
    """
    def __init__(self, X : torch.Tensor, Y : torch.Tensor, split : Optional[float] = 0.8):
        """
        Initializes the TrainingData object.

        Args:
            X (torch.Tensor): The input data matrix.
            Y (torch.Tensor): The target output matrix.
        """
        super().__init__(X, Y)  # Initialize TensorDataset with training data

        # Verify split is provided and is a valid value
        if split is None:
            raise ValueError("Split ratio must be provided.")
        if not (0 < split < 1):
            raise ValueError("Split ratio must be between 0 and 1.")

        self._mode : TrainingMode = TrainingMode(initial_value='init')
        self.X = X
        self.Y = Y
        split = 1.0
        self.split : float = split
        if split < 1.0:
            split_data : Tuple[Subset, Subset] = DataManager.partition(self)
        
            self.train_dataset : Subset = split_data[0]
            self.test_dataset : Subset = split_data[1]
        else:
            all_indices = list(range(len(self)))
            full_dataset_as_subset = Subset(self, all_indices)
            self.train_dataset = full_dataset_as_subset
            self.test_dataset = full_dataset_as_subset


    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        if new_mode not in ['train', 'test']:
            raise ValueError("Mode must be 'train' or 'test'")
        self._mode = new_mode

    def get_data(self):
        if self.mode == 'train':
            return self.train_dataset
        elif self.mode == 'test':
            return self.test_dataset
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @property
    def data(self):
        if self.mode == 'train':
            return self.tensors[0][self.train_dataset.indices]
        elif self.mode == 'test':
            return self.tensors[0][self.test_dataset.indices]
        elif self.mode == 'init':
            return self.tensors[1]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @property
    def targets(self):
        if self.mode == 'train':
            return self.tensors[1][self.train_dataset.indices]
        elif self.mode == 'test':
            return self.tensors[1][self.test_dataset.indices]
        elif self.mode == 'init':
            return self.tensors[1]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    @staticmethod
    def partition(manager) -> Tuple[Subset, Subset]:
        """
        Splits the data into training and testing sets.

        Args:
            split (float): The proportion of data to use for training (default is 0.8).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The training and testing datasets.
        """

        # Define the sizes of the training and testing sets
        train_size = int(manager.split * len(manager))  # 80% for training
        test_size = len(manager) - train_size  # Remaining for testing

        # Perform the random split
        subsets = random_split(manager, [train_size, test_size])

        # Unpack the list into your train_dataset and test_dataset variables
        train_dataset: Subset = subsets[0]
        test_dataset: Subset = subsets[1]

        return train_dataset, test_dataset

class HypersphereData(DataManager):
    """
    A subclass of DataManager where the input data points are sampled i.i.d.
    from the surface of a unit hypersphere, and the target outputs are also generated.

    Attributes:
        X (torch.Tensor): The input data matrix with points on the unit hypersphere.
                         Shape: (P, d).
        Y (torch.Tensor): The target output matrix.
                         Shape: (P, 1).
    """
    def __init__(self, 
                 num_points: int, 
                 input_dim: int, 
                 normalized : bool = True,
                 data_noise_std : float = 0.1, 
                 split: float = 0.8):
        """
        Initializes the HypersphereData object by generating data on the unit hypersphere
        and corresponding random target outputs.

        Args:
            num_points (int): The number of data points to generate (P).
            input_dim (int): The dimensionality of each data point (d).
        """
        self.split = split
        self.num_points = num_points
        self.input_dim = input_dim
        self.normalized = normalized
        self.data_noise_std = data_noise_std

        random_data_X: torch.Tensor = HypersphereData.sample_hypersphere(self.num_points, self.input_dim, self.normalized)

        # Generate random target outputs (for demonstration purposes).
        # In a real scenario, these would be based on some underlying function.
        w_rand = torch.randn(self.input_dim, 1)
        w_normalized = w_rand/torch.mean(w_rand)
        self.w_rand_normalized = w_normalized
        
        random_targets : torch.Tensor = torch.matmul(random_data_X, w_normalized) + (torch.randn(self.num_points, 1)*self.data_noise_std)

        super().__init__(random_data_X, random_targets, split)

    @staticmethod
    def sample_hypersphere(num_points : int, input_dimension: int, normalized : bool = True) -> torch.Tensor:
        """
        Samples points uniformly from the surface of a unit hypersphere in d dimensions.

        Args:
            num_points (int): The number of points to sample.
            input_dim (int): The dimensionality of the hypersphere.

        Returns:
            torch.Tensor: A tensor of shape (num_points, input_dim) containing the sampled points.
        """
                # Generate random input vectors from a standard normal distribution.
        raw_data_X: torch.Tensor = torch.randn(num_points, input_dimension)
        raw_data_X = raw_data_X
        # Normalize each vector to have unit length.
        data_X: torch.Tensor = raw_data_X #

        if normalized:
            data_X : torch.Tensor = data_X / torch.norm(raw_data_X, dim=1, keepdim=True)
        return data_X.to(hp.DEVICE)
