import torch
import numpy as np
from pathlib import Path
import os
import sys
import argparse
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric

def compute_K_kernel(f, q, P, n):
    # f: (P, q, n1)
    # Returns K: (q, P, P)
    # For each ensemble member, compute K = f_uqi @ f_vqi^T / (P * n1)

    # Compute kernel per ensemble: K_i[u,v] = sum_m f_inf[u,i,m] * f_inf[v,i,m]
    # Result: (ens, P, P)
    hh_inf_i = torch.einsum('uqm,vqm->quv', f, f) / (n)
    return hh_inf_i

class FCN3EigenLogger:
    def __init__(self, runs_dir, seed, P=None, N=None, chi=None, lr=None):
        # Each seed gets its own subdir, but all are visible in TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(runs_dir, f"seed_{seed}_P_{P}_N_{N}_chi_{chi}_lr_{lr}"))

    def log(self, net, X, Y, step, loss):
        # net: FCN3NetworkEnsembleLinear
        # X: (P, d)
        # Y: (P, 1) or (P,)
        # Compute K kernel and eigenvalues for each ensemble member

        if step % 10000 == 0:
            with torch.no_grad():
                P = 3000
                large_X = torch.randn(P, X.shape[1], device=X.device)
                q = net.ens
                n2 = net.n2
                f = net.h1_preactivation(large_X)  # (P, q, n2)
                Ks = compute_K_kernel(f, q, P, n2)
                largest_eigs = []
                degenerate_eigs = []
                for i in range(Ks.shape[0]):
                    K = Ks[i]
                    eigvals = torch.linalg.eigvalsh(K) / P
                    degenerate_eigs.append(eigvals.sort(descending=True)[0][1:large_X.shape[1]].mean().item())
                    largest = eigvals.max().item()
                    largest_eigs.append(largest)
                # Also log the average largest eigenvalue
                avg_largest = sum(largest_eigs) / len(largest_eigs)
                self.writer.add_scalar("eigval/largest_avg", avg_largest, step)
                self.writer.add_scalar("eigval/degenerate_avg", sum(degenerate_eigs)/len(degenerate_eigs), step)
                print("Eigenvalues: largest_avg =", avg_largest,
                      ", degenerate_avg =", sum(degenerate_eigs)/len(degenerate_eigs))
        self.writer.add_scalar("loss/loss", loss, step)

    def close(self):
        self.writer.close()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Training parameters
d = 20
n1 = 100
n2 = 100
P_list = [100, 400, 1000]
num_epochs = 50000000  # 3 million epochs as in previous scripts
lr = 1e-6
seeds = [1, 2, 3]  # Four dataset seeds

# Data generation (random regression task)
def generate_data(P, d, seed):
    torch.manual_seed(seed)
    X = torch.randn(P, d, device=DEVICE)
    y = X[:, 0].to(device=DEVICE)  # (P, 1)
    return X, y

def train_fcn3(P, kappa, chi, seed, save_dir, ens=5, dry_run=False):
    X, y = generate_data(P, d, seed)

    model = FCN3NetworkActivationGeneric(d=d, n1=n1, n2=n2, P=P, ens=ens, activation="erf",
                 weight_initialization_variance=(1/d, 1/n1, 1/(n2*chi)), device=DEVICE)

    temperature = 2.0 * kappa / chi
    dt = lr
    losses = []
    max_epochs = 3 if dry_run else num_epochs
    logger = FCN3EigenLogger(runs_dir="runs/fcn3_eig", seed=seed, P=P, N=n2, chi = chi, lr=lr)
    # Reshape y to be P x ens
    y = y.unsqueeze(-1).expand(-1, ens)  # (P, ens)
    for epoch in range(max_epochs):
        model.zero_grad()
        y_pred = model(X)  # shape: (P, ens)

        # Use sum reduction for all ensemble members
        loss = ((y_pred - y) ** 2)

        loss = loss.mean()


        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                
                if param.grad is not None:
                    # Weight decay: 1/fan-in for each layer
                    if name.endswith('A'):
                        decay = (n2*chi*temperature) * param
                    elif name.endswith('W1'):
                        decay = (n1) * param * temperature
                    elif name.endswith('W0'):
                        decay = (d) * param * temperature
                    else:
                        decay = 0.0
                    noise = torch.randn_like(param) * (2 * temperature * dt) ** 0.5
                    param.add_(-dt * (param.grad + decay) + noise)
        if not dry_run and epoch % 1000 == 0:
            losses.append(loss.item())
            print(f"P={P} seed={seed} epoch={epoch} loss={loss.item():.6f}")

            torch.save(model.state_dict(), os.path.join(save_dir, f"model_seed{seed}.pt"))
            logger.log(model, X, y, epoch, loss.item())


    if not dry_run:
        # Save model and losses
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_seed{seed}.pt"))
        np.save(os.path.join(save_dir, f"losses_seed{seed}.npy"), np.array(losses))
        print(f"Saved model and losses for P={P}, seed={seed}.")

def run_train_subprocess(P, kappa, chi, seed, save_dir, dry_run=False):
    import subprocess
    import sys
    cmd = [sys.executable, __file__, '--train',
           '--P', str(P), '--kappa', str(kappa), '--chi', str(chi), '--seed', str(seed), '--save_dir', save_dir]
    if dry_run:
        cmd.append('--dry-run')
    return subprocess.Popen(cmd, preexec_fn=os.setsid)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--P', type=int)
    parser.add_argument('--kappa', type=float)
    parser.add_argument('--chi', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.train:
        train_fcn3(args.P, args.kappa, args.chi, args.seed, args.save_dir, dry_run=args.dry_run)
        return

    # Parent process: launch all jobs in parallel (dry run mode)
    jobs = []
    dry_run = False  # Set to True for dry run, False for full run
    for P in P_list:
        kappa =  P / 200
        chi = n2  # mean field scaling
        for seed in seeds:
            save_dir = f"./models/fcn3_d{d}_n1{n1}_n2{n2}_P{P}_kappa{kappa}_chi{chi}_seed{seed}"
            os.mkdir('./models') if not os.path.exists('./models') else None
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs('./runs/fcn3_eig') if not os.path.exists('./runs/fcn3_eig') else None
            p = run_train_subprocess(P, kappa, chi, seed, save_dir, dry_run=dry_run)
            jobs.append(p)

    import signal
    def kill_all_children(signum, frame):
        print("Parent received signal, killing all children...")
        for p in jobs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        sys.exit(1)

    signal.signal(signal.SIGINT, kill_all_children)
    signal.signal(signal.SIGTERM, kill_all_children)

    for p in jobs:
        p.wait()

if __name__ == "__main__":
    main()
