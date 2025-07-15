import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from opt_einsum import contract, contract_path
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ logs
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import sys
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import os
from datetime import datetime
import argparse
import json
import sys
sys.path.insert(0, '/home/akiva/FCNX-Ensembling')
from GPKit import gpr_dot_product_explicit
import socket
import tempfile
import shutil
import atexit
import uuid
import math as mt

# 2. Generate Data
input_size = 20
hidden_size = 1500
output_size = 1
num_samples = 20
epochs = 10_000_000  # You might increase this with a decaying LR
chi = hidden_size
k = 1.0
t0 = 2 * k
t = t0 / chi  # Temperature for Langevin (used in noise)

# --- Learning Rate Schedule Parameters ---
T = epochs * 0.8
lrA = 1e-7 / num_samples
lrB = (1.0 / 3) * lrA / num_samples 
beta = mt.log(lrA / lrB) / T
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Set seeds as constants
DATA_SEED = 613
MODEL_SEED = 26
LANGEVIN_SEED = 480

# Set the default dtype to float64
torch.set_default_dtype(torch.float64)

class FCN3NetworkEnsembleLinear(nn.Module):
    def __init__(self, d, n1, n2, ens=1, weight_initialization_variance=(1.0, 1.0, 1.0)):
        super().__init__()
        self.arch = [d, n1, n2]
        self.d = d
        self.n1 = n1
        self.n2 = n2
        self.ens = ens
        
        self.W0 = nn.Parameter(torch.normal(mean=0.0,
                                           std=torch.full((ens, n1, d), weight_initialization_variance[0]**0.5, device=device)),
                               requires_grad=True)
        self.W1 = nn.Parameter(torch.normal(mean=0.0,
                                           std=torch.full((ens, n2, n1), weight_initialization_variance[1]**0.5, device=device)),
                               requires_grad=True)
        self.A = nn.Parameter(torch.normal(mean=0.0,
                                          std=torch.full((ens, 1, n2), weight_initialization_variance[2]**0.5, device=device)),
                              requires_grad=True)
        
        # Pre-compute einsum paths for better performance
        self._precompute_einsum_paths()
        
        # Pre-allocate single noise buffer for all parameters
        self.noise_buffer = torch.empty(1, device=device, dtype=torch.float64)

    def _precompute_einsum_paths(self):
        """Pre-compute einsum paths for repeated operations"""
        eq = 'eij,ejk,ekl,ul->uie'
        shapes = [
            (self.ens, 1, self.n2),
            (self.ens, self.n2, self.n1),
            (self.ens, self.n1, self.d),
            (num_samples, self.d)
        ]
        dummy_tensors = [torch.empty(s, device=device, dtype=torch.float64) for s in shapes]
        path, _ = contract_path(eq, *dummy_tensors)
        self.forward_path = path

    def h1_activation(self, X):
        return contract(
            'ijk,ikl,unl->uij',
            self.W1, self.W0, X,
            backend='torch'
        )

    def h0_activation(self, X):
        return contract(
            'ikl,unl->uik',
            self.W0, X,
            backend='torch'
        )

    def forward(self, X):
        """
        Efficiently computes the outputs of a three layer network using opt_einsum
        f : P*d -> P*e*1
        """
        return contract(
            'eij,ejk,ekl,ul->uie',
            self.A, self.W1, self.W0, X,
            backend='torch',
            optimize=self.forward_path
        )

# Custom loss function (slightly faster than MSE)
def custom_mse_loss(outputs, targets):
    diff = outputs - targets
    return 0.5 * torch.sum(diff * diff)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or resume FCN3 ensemble model.')
    parser.add_argument('--modeldesc', type=str, default=None, help='Model description directory to resume from')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (no logging or saving)')
    parser.add_argument('--epochs', type=int, default=2_000_000, help='Number of training epochs (default: 2,000,000)')
    args = parser.parse_args()

    debug = args.debug
    epochs = args.epochs
    modeldesc = ''
    save_dir = getattr(args, 'modeldesc', '')
    
    # Optimized logging intervals
    log_interval = 5000
    detailed_log_interval = log_interval * 4
    eigenvalue_log_interval = log_interval * 10
    
    # Create or use custom folder with P, D, N, epochs, lrA, and timestamp
    if not debug:
        if args.modeldesc is not None:
            modeldesc = args.modeldesc
            save_dir = os.path.join("/home/akiva/gpnettrain", modeldesc)
            runs_dir = os.path.join(save_dir, "runs")
            os.makedirs(runs_dir, exist_ok=True)
            resume = True
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            modeldesc = f"P_{num_samples}_D_{input_size}_N_{hidden_size}_epochs_{epochs}_lrA_{lrA:.2e}_time_{timestamp}"
            save_dir = os.path.join("/home/akiva/gpnettrain", modeldesc)
            os.makedirs(save_dir, exist_ok=True)
            runs_dir = os.path.join(save_dir, "runs")
            os.makedirs(runs_dir, exist_ok=True)
            resume = False
        # Initialize TensorBoard writer
        writer_cm = SummaryWriter(log_dir=runs_dir)
        state_path = os.path.join(save_dir, 'state.json')
        model_path = os.path.join(save_dir, 'model.pth')
    else:
        if args.modeldesc is not None:
            resume = True
        # Create a random temp directory for debug TensorBoard logs
        debug_tmp_dir = os.path.join('/home/akiva/gpnettrain/debugtmp', str(uuid.uuid4()))
        os.makedirs(debug_tmp_dir, exist_ok=True)
        writer_cm = SummaryWriter(log_dir=debug_tmp_dir)
        def cleanup_debug_tmp():
            try:
                shutil.rmtree(debug_tmp_dir)
                print(f"Deleted debug TensorBoard log directory: {debug_tmp_dir}")
            except Exception as e:
                print(f"Could not delete debug TensorBoard log directory: {e}")
        atexit.register(cleanup_debug_tmp)
        # Also handle KeyboardInterrupt
        import signal
        def handle_sigint(sig, frame):
            cleanup_debug_tmp()
            exit(0)
        signal.signal(signal.SIGINT, handle_sigint)
    
    # Launch TensorBoard on an unused port (always, regardless of debug)
    try:
        from tensorboard import program
        # Find an unused port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
        logdir = runs_dir if not debug else debug_tmp_dir
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', logdir, '--port', str(port)])
        url = tb.launch()
        print(f"TensorBoard started at {url}")
    except Exception as e:
        print(f"Could not launch TensorBoard: {e}")
    
    state_path = os.path.join(save_dir, 'state.json')
    model_path = os.path.join(save_dir, 'model.pth')

    print(f"Torch device: {device}")

    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'N/A'}")

    # Seed for data
    torch.manual_seed(DATA_SEED)
    # Create data on GPU directly and pin memory
    X = torch.randn((num_samples, input_size), dtype=torch.float64, device=device)
    Y = X[:, 0].unsqueeze(-1)

    # Seed for model
    torch.manual_seed(MODEL_SEED)
    ens = 100
    model = FCN3NetworkEnsembleLinear(input_size, hidden_size, hidden_size,
                                     ens=ens,
                                     weight_initialization_variance=(1/input_size, 1.0/hidden_size, 1.0/(hidden_size * chi)))
    model.to(device)

    # Compile model if available (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Model compiled successfully")
        except Exception as e:
            print(f"Could not compile model: {e}")

    losses = []
    
    # Move weight decay to GPU - THIS IS CRITICAL FOR PERFORMANCE
    weight_decay = torch.tensor([input_size, hidden_size, hidden_size*chi], dtype=torch.float64, device=device) * t

    # Pre-allocate noise buffer for Langevin dynamics
    noise_buffer = torch.empty(1, device=device, dtype=torch.float64)
    
    # Resume logic
    epoch = 0
    loaded_model = False
    if resume:
        if os.path.exists(model_path):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                checkpoint = torch.load(model_path, map_location=device)
                model_keys = model.state_dict().keys()

                # Check if the model's keys indicate an optimized wrapper (e.g., '_orig_mod' prefix)
                # We can check if any of the checkpoint keys, when prefixed, match a model key
                needs_adaptation = False
                for k in checkpoint.keys():
                    if f"_orig_mod.{k}" in model_keys:
                        needs_adaptation = True
                        break

                if needs_adaptation:
                    print("Adapting state_dict keys for optimized model...")
                    new_state_dict = {}
                    for k, v in checkpoint.items():
                        new_key = f"_orig_mod.{k}"
                        new_state_dict[new_key] = v
                    breakpoint()
                    model.load_state_dict(new_state_dict)
                else:
                    print("State_dict keys already aligned. Loading directly...")
                    model.load_state_dict(checkpoint)
                model.to(device)
            print(f"Loaded model from {model_path}")
            loaded_model = True
        if os.path.exists(state_path):
            print(f'State exists, loading state from: {state_path}')
            with open(state_path, 'r') as f:
                state = json.load(f)
                epoch = state.get('epoch', 0)
                print(f"Resuming from epoch {epoch}")
        else:
            print("No state.json found, starting from epoch 0")
    elif not debug:
        print(f"Starting new training in {save_dir}")
    else:
        print("Running in debug mode: no logging or saving.")
    
    if not loaded_model and not debug and resume:
        print("No model checkpoint found, starting from scratch.")
    print(f"Beginning training at epoch {epoch}")

    # Main training loop
    with tqdm(total=epochs, desc="Training", unit="epoch", initial=epoch) as pbar:
        while epoch < epochs:
            # Update learning rate
            if epoch < epochs * 0.8:
                current_base_learning_rate = lrA
            else:
                current_base_learning_rate = lrA / 3
            
            effective_learning_rate_for_update = current_base_learning_rate 
            noise_scale = (2 * effective_learning_rate_for_update * t)**0.5

            # Clear gradients
            model.zero_grad()

            # Forward pass
            try:
            
                outputs = model(X)
                loss = custom_mse_loss(outputs, Y.unsqueeze(-1))
                
                # Check for valid loss
                if not torch.isfinite(loss):
                    print(f"Warning: Invalid loss at epoch {epoch}: {loss.item()}")
                    pbar.update(1)
                    epoch += 1
                    continue

                losses.append(loss.item())

                # Backward pass
        
                loss.backward()

                # Manual Langevin dynamics update
                with torch.no_grad():
                    for i, param in enumerate(model.parameters()):
                        if param.grad is not None:
                            # Gradient step already handled by scaler.step() or loss.backward()
                            # Add Langevin noise
                            noise_buffer.resize_(param.shape).normal_(0, noise_scale)
                            param.data.add_(noise_buffer)
                            # Weight decay
                            param.data.add_(param.data, alpha=-(weight_decay[i]) * effective_learning_rate_for_update)

            except Exception as e:
                print(f"Error in training loop at epoch {epoch}: {e}")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = exc_tb.tb_frame.f_code.co_filename
                line_number = exc_tb.tb_lineno
                print(f"An exception occurred:")
                print(f"  Type: {exc_type.__name__}")
                print(f"  Message: {e}")
                print(f"  File: {fname}")
                print(f"  Line: {line_number}")
                pbar.update(1)
                epoch += 1
                continue

            # Optimized logging
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                avg_loss = loss.item() / (ens * num_samples)
                
                if not debug and writer_cm is not None:
                    # Basic metrics every log_interval
                    writer_cm.add_scalars('Metrics', {
                        'Loss': avg_loss,
                        'Learning_Rate': current_base_learning_rate
                    }, epoch)
                    
                    # More expensive computations less frequently
                    if (epoch + 1) % detailed_log_interval == 0:
                        # --- Cosine Similarity with GPR ---
                        with torch.no_grad():
                            # Compute GPR prediction (dot product kernel)
                            gpr_pred = gpr_dot_product_explicit(X, Y.squeeze(), X, 1.0)
                            # Model output: average over ensemble dimension if present
                            model_out = model(X)
                            if model_out.ndim == 3:
                                model_out = model_out.mean(dim=2)  # average over ensemble
                            model_out_flat = model_out.flatten()
                            gpr_pred_flat = gpr_pred.flatten()
                            # Cosine similarity
                            cos_sim = torch.dot(model_out_flat, gpr_pred_flat) / (torch.norm(model_out_flat) * torch.norm(gpr_pred_flat) + 1e-12)
                            writer_cm.add_scalar('CosineSimilarity/Model_vs_GPR', cos_sim.item(), epoch)
                    
                    # Eigenvalue computation even less frequently
                    if (epoch + 1) % eigenvalue_log_interval == 0:
                        with torch.no_grad():
                            W0 = model.W0.permute(*range(model.W0.ndim - 1, -1, -1))
                            W1 = model.W1.permute(*range(model.W1.ndim - 1, -1, -1))
                            
                            covW0W1 = contract('kje,ije,nme,kme->in', W1, W0, W0, W1, backend='torch') / (hidden_size * ens)
                            
                            # Only compute diagonal if that's all we need
                            lH = torch.diagonal(covW0W1, dim1=-2, dim2=-1).squeeze()
                            scalarsH = {}
                            
                            for i in range(min(lH.shape[0], 10)):  # Limit to first 10 eigenvalues
                                scalarsH[str(i)] = lH[i]
                            
                            writer_cm.add_scalars(f'Eigenvalues/lambda_H(W1)', scalarsH, epoch)
                    
                    # Save model and state less frequently
                    if (epoch + 1) % detailed_log_interval == 0:
                        model_filename = f"model.pth"
                        torch.save(model.state_dict(), os.path.join(save_dir, model_filename))
                        
                        with open(os.path.join(save_dir, f"losses.txt"), "a") as f: 
                            f.write(f"{epoch},{loss.item()}\n")
                        
                        # Save state.json with current epoch
                        with open(state_path, 'w') as f:
                            json.dump({'epoch': epoch}, f)

            # Update tqdm progress bar
            pbar.update(1)
            pbar.set_postfix({"MSE Loss": f"{avg_loss:.6f}" if (epoch + 1) % log_interval == 0 else f"{loss.item() / (ens * num_samples):.6f}", 
                             "Lr": f"{current_base_learning_rate:.2e}"})
            epoch += 1

    # Close TensorBoard writer
    if not debug and writer_cm is not None:
        writer_cm.close()
    elif debug and writer_cm is not None:
        writer_cm.close()