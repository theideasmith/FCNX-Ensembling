import torch
import sys


def get_device():
    """
    Determines the appropriate device for PyTorch computations (MPS, CUDA, or CPU).

    Returns:
        torch.device: The selected device.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found. Using MPS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device found. Using CUDA.")
    else:
        device = torch.device("cpu")
        print("No GPU found. Using CPU.")
    return device

def check_mps_available_or_fail():
    """
    Checks if MPS is available and built, and raises an exception if not.

    Raises:
        Exception: If MPS is not available or not built.
    """
    if not torch.backends.mps.is_available():
        raise Exception("MPS is not available on this machine.")
    if not torch.backends.mps.is_built():
        raise Exception("PyTorch was not built with MPS support.")
    print("MPS is available and built. Ready to use.")
    return torch.device("mps") # Returns the mps device

def try_initialize_mps():
    """
    Attempts to initialize MPS and returns the device if successful, None otherwise.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS is available and built.  Using MPS.")
        return torch.device("mps")
    else:
        print("MPS is not available or built.  Using CPU.")
        return None # Returns None
