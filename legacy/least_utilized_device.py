import torch
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