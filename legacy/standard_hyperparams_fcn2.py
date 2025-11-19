from standard_hyperparams import HIDDEN_WIDTH_1
import torch
from activations import linear_activation
from check_mps import get_device
# --- Hyperparameters ---
INPUT_DIMENSION: int = 50
FCN2_HIDDEN_WIDTH: int = 1000
CHI = FCN2_HIDDEN_WIDTH

KAPPA = 1.0 / CHI

# Large enough that the weight decay will be nonnegligible.
TEMPERATURE = 2 * KAPPA



FCN2_LAMBDA_1 = TEMPERATURE * INPUT_DIMENSION #weight decay factor
FCN2_LAMBDA_2 = TEMPERATURE * FCN2_HIDDEN_WIDTH * CHI

FCN2_WEIGHT_SIGMA1: float = 1.0/INPUT_DIMENSION
FCN2_WEIGHT_SIGMA2: float = 1.0/(FCN2_HIDDEN_WIDTH*CHI)
NUM_DATA_POINTS: int = 200
BATCH_SIZE: int = 30
LEARNING_RATE: float = (0.0015)
NOISE_STD_LANGEVIN: float = (2 * LEARNING_RATE * KAPPA )**0.5
NUM_EPOCHS: int = 10000

WEIGHT_DECAY_CONFIG: dict = {
            'fc1.weight': FCN2_LAMBDA_1,
            'fc2.weight': FCN2_LAMBDA_2,
            'fc1.bias': 0.0,
            'fc2.bias': 0.0,
}

FCN2_WEIGHT_SIGMA = (
    FCN2_WEIGHT_SIGMA1,
    FCN2_WEIGHT_SIGMA2,
)

TEST_TRAIN_SPLIT :float  = 0.8
TARGET_NOISE : float= 0.001

# --- Initialize Network ---
# Example: Linear activation network
HPS : dict = {
    'input_dimension': INPUT_DIMENSION,
    'hidden_width_1': FCN2_HIDDEN_WIDTH,
    'activation': linear_activation,
    'output_activation': linear_activation,
    'weight_sigma1': FCN2_WEIGHT_SIGMA1,
    'weight_sigma2': FCN2_WEIGHT_SIGMA2,
}

def get_least_utilized_gpu_device():
    """
    Identifies and returns the PyTorch device (e.g., 'cuda:0', 'cuda:1')
    that currently has the least allocated memory.

    Returns:
        torch.device: The PyTorch device object for the least utilized GPU,
                      or 'cpu' if no GPUs are available.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No CUDA devices found. Using CPU.")
        return torch.device("cpu")

    least_memory_allocated = float('inf')
    least_utilized_device = None

    print(f"Checking {num_gpus} GPU(s) for utilization...")

    for i in range(num_gpus):
        device = torch.device(f"cuda:{i}")
        # Set the current device to query its memory
        torch.cuda.set_device(device)
        
        # Get allocated memory (bytes)
        # memory_allocated() returns the total bytes that PyTorch has allocated on the GPU.
        # memory_reserved() returns the total bytes that PyTorch has reserved (cached) on the GPU.
        # allocated_bytes = torch.cuda.memory_allocated(device)
        # reserved_bytes = torch.cuda.memory_reserved(device)
        
        # For simplicity, we'll use memory_allocated as a primary metric.
        # You might consider memory_reserved or a combination for more nuanced decisions.
        current_allocated_memory = torch.cuda.memory_allocated(device)
        
        print(f"  Device {i} ({torch.cuda.get_device_name(i)}): {current_allocated_memory / (1024**2):.2f} MB allocated")

        if current_allocated_memory < least_memory_allocated:
            least_memory_allocated = current_allocated_memory
            least_utilized_device = device

    if least_utilized_device:
        print(f"\nSelected device: {least_utilized_device} (Least allocated memory: {least_memory_allocated / (1024**2):.2f} MB)")
        return least_utilized_device
    else:
        # Fallback, should not happen if num_gpus > 0
        print("Could not determine least utilized GPU. Falling back to CPU.")
        return torch.device("cpu")

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

print(get_device())
