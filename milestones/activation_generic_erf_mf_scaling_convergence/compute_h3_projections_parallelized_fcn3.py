import torch
import torch.multiprocessing as mp
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

# Import FCN3NetworkActivationGeneric from the lib directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric


def process_batches(rank, world_size, model_path, d, N, P, chi, ens, weight_var, 
                    n_samples, batch_size, device_id, result_queue):
    """
    Worker function that processes a subset of batches
    """
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    # Load model in each process
    state_dict = torch.load(model_path, map_location=device)
    model = FCN3NetworkActivationGeneric(
        d=d, n1=N, n2=N, P=P, ens=ens,
        activation="erf",
        weight_initialization_variance=weight_var,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Calculate which batches this worker processes
    n_batches = (n_samples + batch_size - 1) // batch_size
    batches_per_worker = (n_batches + world_size - 1) // world_size
    start_batch = rank * batches_per_worker
    end_batch = min(start_batch + batches_per_worker, n_batches)
    
    local_projections_sum = None
    local_n_total = 0
    
    with torch.no_grad():
        for batch_idx in range(start_batch, end_batch):
            current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
            
            # Set seed for reproducibility (optional)
            torch.manual_seed(42 + batch_idx)
            
            X = torch.randn(current_batch_size, d, device=device)
            h1 = model.h1_preactivation(X)  # (batch_size, ens, n1)
            x0 = X[:, 1]
            He3 = (x0 ** 3 - 3.0 * x0).view(-1, 1, 1)
            proj = h1 * He3
            proj_sum = proj.sum(dim=0)  # (ens, n1)
            
            if local_projections_sum is None:
                local_projections_sum = proj_sum
            else:
                local_projections_sum += proj_sum
            
            local_n_total += current_batch_size
            
            if (batch_idx - start_batch) % 10 == 0:
                print(f"Worker {rank}: Processed batch {batch_idx+1}/{n_batches}")
    
    # Move to CPU before putting in queue
    result_queue.put({
        'projections_sum': local_projections_sum.cpu(),
        'n_total': local_n_total,
        'rank': rank
    })
    print(f"Worker {rank} completed: {local_n_total} samples")


def main():
    d = 150
    n_samples = 100_000_000
    batch_size = 10_000
    
    # Determine number of workers
    n_gpus = torch.cuda.device_count()
    world_size = max(1, n_gpus)  # Use number of GPUs, or 1 if no GPU
    
    # If CPU only, you can use more workers
    if not torch.cuda.is_available():
        world_size = mp.cpu_count() // 2  # Use half of CPU cores
    
    print(f"Using {world_size} workers")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model parameters
    model_path = Path("/home/akiva/FCNX-Ensembling/milestones/activation_generic_erf_mf_scaling_convergence/results/d150_P3000_N1600_chi50_kappa0.1/seed42/model.pt")
    state_dict = torch.load(model_path, map_location='cpu')  # Load to CPU first
    ens = int(state_dict['W0'].shape[0])
    P = 3000
    N = 1600
    chi = 50
    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
    
    # Create result queue
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    
    # Launch workers
    processes = []
    for rank in range(world_size):
        device_id = rank % max(1, n_gpus) if torch.cuda.is_available() else 0
        print()
        p = mp.Process(
            target=process_batches,
            args=(rank, world_size, model_path, d, N, P, chi, ens, weight_var,
                  n_samples, batch_size, device_id, result_queue)
        )
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    for _ in range(world_size):
        results.append(result_queue.get())
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Aggregate results
    total_projections_sum = None
    total_n_samples = 0
    
    for result in results:
        if total_projections_sum is None:
            total_projections_sum = result['projections_sum']
        else:
            total_projections_sum += result['projections_sum']
        total_n_samples += result['n_total']
    
    print(f"\nTotal samples processed: {total_n_samples}")
    
    # Calculate final variance
    mean_proj = total_projections_sum / total_n_samples
    var = torch.var(mean_proj - torch.mean(mean_proj)).item()
    
    print(f"Final variance of mean projections over (ens, n1): {var:.3e}")
    
    # Note: For running variance during computation, you'd need a more complex 
    # architecture with periodic synchronization between workers
    # Here we just compute the final result


if __name__ == "__main__":
    main()