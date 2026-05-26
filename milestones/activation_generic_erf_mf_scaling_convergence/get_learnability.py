import sys
import torch
import numpy as np
from pathlib import Path
import torch.multiprocessing as mp
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric

parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', type=str, default=None, help='Path to model file to compute learnability')
parser.add_argument('--num_workers', type=int, default=None, help='Number of workers for multiprocessing')
args = parser.parse_args()



def worker(proc_id, model_state_dict, d, n1, n2, P, ens, device, batch_size, num_batches, queue):
    torch.cuda.set_device(device)
    model = FCN3NetworkActivationGeneric(
        d=d, n1=n1, n2=n2, P=P, ens=ens, activation="erf",
        weight_initialization_variance=(1.0/d, 1.0/n1, 1.0/(n1*n2))
    ).to(device)
    model.load_state_dict({k: v.squeeze(0) if v.ndim > (3 if 'W' in k else 2) else v for k, v in model_state_dict.items()}, strict=False)
    model.eval()
    h1_sum, h3_sum = 0.0, 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            X_batch = torch.randn(batch_size, d, device=device)
            out = model(X_batch)
            x0 = X_batch[:, 0]
            h3_comp = (x0**3 - 3*x0) / np.sqrt(6)
            h1_sum += (out * x0.unsqueeze(-1)).sum().item() if out.ndim > 1 else (out * x0).sum().item()
            h3_sum += (out * h3_comp.unsqueeze(-1)).sum().item() if out.ndim > 1 else (out * h3_comp).sum().item()
    queue.put((h1_sum, h3_sum))

def compute_learnability(model_path, num_workers):
    sd = torch.load(model_path, map_location="cpu")
    d = sd['W0'].shape[-1]
    n1 = sd['W0'].shape[-2]
    n2 = sd['W1'].shape[-2]
    P = sd['W0'].shape[0] if sd['W0'].ndim == 3 else sd['W0'].shape[1]
    ens = P
    P_total, batch_size = 1000000, 25000
    num_workers_local = min(num_workers, torch.cuda.device_count() or 1)
    num_batches = P_total // batch_size // num_workers_local
    queue = mp.Queue()
    procs = []
    for i in range(num_workers_local):
        device = f"cuda:{i % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
        p = mp.Process(target=worker, args=(i, sd, d, n1, n2, P, ens, device, batch_size, num_batches, queue))
        p.start()
        procs.append(p)
    h1_sum, h3_sum = 0.0, 0.0
    for _ in range(num_workers_local):
        h1, h3 = queue.get()
        h1_sum += h1
        h3_sum += h3
    for p in procs:
        p.join()
    h1_learn = h1_sum / (P_total // num_workers_local * num_workers_local)
    h3_learn = h3_sum / (P_total // num_workers_local * num_workers_local)
    print(f"Learnability He1: {h1_learn}")
    print(f"Learnability He3: {h3_learn}")
    return h1_learn, h3_learn

def scan_and_plot_learnability():
    # Scan both result folders for models, as in analysis_vs_d.py
    results_dirs = [
        Path(__file__).parent / "d_scan_erf_results",
        Path(__file__).parent / "d_scan_erf_results_Pd32"
    ]
    all_dirs = []
    for rdir in results_dirs:
        if rdir.exists():
            all_dirs.extend(sorted(list(rdir.glob("d*/*seed*"))))
    data = []
    for m_dir in all_dirs:
        pt_path = m_dir / "model_final.pt" if (m_dir / "model_final.pt").exists() else m_dir / "model.pt"
        if not pt_path.exists():
            continue
        sd = torch.load(pt_path, map_location="cpu")
        d = sd['W0'].shape[-1]
        n1 = sd['W0'].shape[-2]
        n2 = sd['W1'].shape[-2]
        P = sd['W0'].shape[0] if sd['W0'].ndim == 3 else sd['W0'].shape[1]
        ens = P
        # Use single worker for each model for simplicity
        h1, h3 = compute_learnability(pt_path, num_workers=1)
        label = r"$P \propto d^{1.5}$" if "Pd32" in str(m_dir) else r"$P \propto d$"
        data.append({"d": d, "h1": h1, "h3": h3, "label": label})
    # Group by label
    groups = defaultdict(list)
    for row in data:
        groups[row["label"]].append(row)
    # Plot
    plt.figure(figsize=(10, 7))
    for i, (label, rows) in enumerate(groups.items()):
        dvals = np.array([r["d"] for r in rows])
        h1vals = np.array([r["h1"] for r in rows])
        sort_idx = np.argsort(dvals)
        plt.plot(dvals[sort_idx], h1vals[sort_idx], marker='o', label=label)
    plt.xscale('log')
    plt.xlabel("d (input dimension)")
    plt.ylabel("Learnability He1")
    plt.title("Learnability vs d")
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig("learnability_vs_d.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    if args.modelpath:
        model_path = Path(args.modelpath)
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            sys.exit(1)
        num_workers = args.num_workers if args.num_workers is not None else (torch.cuda.device_count() or 1)
        compute_learnability(model_path, num_workers)
    else:
        scan_and_plot_learnability()
