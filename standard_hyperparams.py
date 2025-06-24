import torch
from activations import linear_activation
from check_mps import get_device
# --- Hyperparameters ---
INPUT_DIMENSION: int = 50

# Large enough that the weight decay will be nonnegligible. 


HIDDEN_WIDTH_1: int = 1000
HIDDEN_WIDTH_2: int = 1000
CHI = HIDDEN_WIDTH_2
KAPPA = 1.0 / CHI 
TEMPERATURE = KAPPA


FCN3_LAMBDA_1 = TEMPERATURE * INPUT_DIMENSION #weight decay factor
FCN3_LAMBDA_2 = TEMPERATURE * HIDDEN_WIDTH_2
FCN3_LAMBDA_3 = TEMPERATURE * HIDDEN_WIDTH_2 * CHI

TEMPERATURE = KAPPA
FCN3_WEIGHT_SIGMA1: float = 1.0/INPUT_DIMENSION
FCN3_WEIGHT_SIGMA2: float = 1.0/HIDDEN_WIDTH_1
FCN3_WEIGHT_SIGMA3: float = 1.0/(HIDDEN_WIDTH_2*CHI)
NUM_DATA_POINTS: int = 200
BATCH_SIZE: int = NUM_DATA_POINTS
LEARNING_RATE: float = (0.0001) 
NOISE_STD_LANGEVIN: float = (2 * LEARNING_RATE * TEMPERATURE )**0.5
NUM_EPOCHS: int = 500

WEIGHT_DECAY_CONFIG: dict = {
            'fc1.weight': FCN3_LAMBDA_1,
            'fc2.weight': FCN3_LAMBDA_2,
            'fc3.weight': FCN3_LAMBDA_3,
            'fc1.bias': 0.0,
            'fc2.bias': 0.0,
            'fc3.bias': 0.0,
}

FCN3_WEIGHT_SIGMA = (
    FCN3_WEIGHT_SIGMA1,
    FCN3_WEIGHT_SIGMA2,
    FCN3_WEIGHT_SIGMA3
)

TEST_TRAIN_SPLIT :int  = 1.0
TARGET_NOISE : int = 0.001

# --- Initialize Network ---
# Example: Linear activation network
HPS : dict = {
    'input_dimension': INPUT_DIMENSION,
    'hidden_width_1': HIDDEN_WIDTH_1,
    'hidden_width_2': HIDDEN_WIDTH_2,
    'activation': linear_activation,
    'output_activation': linear_activation,
    'weight_sigma1': FCN3_WEIGHT_SIGMA1,
    'weight_sigma2': FCN3_WEIGHT_SIGMA2,
    'weight_sigma3': FCN3_WEIGHT_SIGMA3,
}
from least_utilized_device import get_least_utilized_gpu_device
DEVICE = None
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("MPS device found. Using MPS.")
elif torch.cuda.is_available():
    DEVICE = get_least_utilized_gpu_device()
    print("CUDA device found. Using CUDA.")
else: 
    DEVICE = 'cpu'


def to_device(data):
    """
    Moves data (tensors or models) to the global device.

    Args:
        data (torch.Tensor or torch.nn.Module): The data to move.

    Returns:
        torch.Tensor or torch.nn.Module: The data on the selected device.
    """
    return data.to(DEVICE)

# Define a global device variable
