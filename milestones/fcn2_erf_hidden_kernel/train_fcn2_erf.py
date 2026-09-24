#!/usr/bin/env python3
"""
Training script for 2-layer erf network with eigenvalue tracking.

Trains a single 2-layer network: Input(d) -> Hidden(n1) with erf -> Output(1)
Tracks eigenvalues of the pre-activation kernel H over training.

Usage:
    python train_fcn2_erf.py --d 10 --P 30 --N 100 --epochs 10000000 --device cuda:0
"""

import argparse
import math
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import traceback
from torch.utils.tensorboard import SummaryWriter

# Set default dtype
torch.set_default_dtype(torch.float32)

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric


def custom_mse_loss(outputs, targets):
    """MSE loss summed over all samples and ensembles."""
    diff = outputs - targets
    return torch.sum(diff * diff)


def _he3(vx):
    """Normalized He3: (x^3 - 3x) / sqrt(6)."""
    return (vx ** 3 - 3.0 * vx) / (6.0 ** 0.5)


def mean_mse(model, x, y):
    """Mean squared error over samples (and ensembles if present)."""
    f = model(x)
    return torch.mean((f - y) ** 2).item()


def cubic_learnability(model, x, y):
    """Residualized cubic Hermite learnability L3 = E[f_res He3] / E[y He3].

    Removes the linear He1 component of f before projecting onto He3, matching
    the journal Langevin ``he3_ratio`` definition.
    """
    f = model(x)
    he1 = x[:, :1]
    he3 = _he3(x[:, 0]).unsqueeze(1)
    inner_f_he1 = torch.mean(f * he1)
    inner_y_he3 = torch.mean(y * he3)
    f_res = f - inner_f_he1 * he1
    num = torch.mean(f_res * he3)
    if inner_y_he3.abs() <= 1e-8:
        return 0.0
    return (num / inner_y_he3).item()


def _resolve_warm_start_state_dict(warm_start_path, device):
    """Load a state_dict from a checkpoint.pt or raw model.pt path."""
    path = Path(warm_start_path)
    if not path.exists():
        raise FileNotFoundError(f"Warm-start path not found: {path}")
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and any(k.startswith("W0") or k == "W0" or k == "A" for k in payload):
        return payload
    raise ValueError(f"Unrecognized warm-start checkpoint format at {path}")


def load_warm_start_into_model(model, warm_start_path, device):
    """Copy overlapping weights from a possibly different (d, N) checkpoint."""
    src = _resolve_warm_start_state_dict(warm_start_path, device)
    dst = model.state_dict()

    def _tensor(key_candidates):
        for key in key_candidates:
            if key in src:
                return src[key]
        return None

    w0_src = _tensor(["W0", "module.W0"])
    a_src = _tensor(["A", "module.A"])
    copied = []

    with torch.no_grad():
        if w0_src is not None and "W0" in dst:
            w0_dst = dst["W0"]
            ens = min(w0_src.shape[0], w0_dst.shape[0])
            n1 = min(w0_src.shape[1], w0_dst.shape[1])
            d_dim = min(w0_src.shape[2], w0_dst.shape[2])
            w0_dst[:ens, :n1, :d_dim].copy_(w0_src[:ens, :n1, :d_dim].to(w0_dst.dtype))
            copied.append(f"W0[:{ens},:{n1},:{d_dim}]")
        if a_src is not None and "A" in dst:
            a_dst = dst["A"]
            ens = min(a_src.shape[0], a_dst.shape[0])
            n1 = min(a_src.shape[1], a_dst.shape[1])
            a_dst[:ens, :n1].copy_(a_src[:ens, :n1].to(a_dst.dtype))
            copied.append(f"A[:{ens},:{n1}]")

    model.load_state_dict(dst)
    print(f"Warm-started from {warm_start_path} ({', '.join(copied) if copied else 'no overlapping tensors'})")
    return model


def _pred_record(epoch, targets, outputs):
    """Create a serializable record of predictions and alignment stats."""
    y_true = targets.squeeze(-1).detach().cpu().numpy()
    y_pred = outputs.mean(dim=1).detach().cpu().numpy()
    y_mean = y_true.mean()
    y_pred_mean = y_pred.mean()
    var_y = np.mean((y_true - y_mean) ** 2)
    if var_y == 0:
        slope = float('nan')
        intercept = float('nan')
    else:
        cov = np.mean((y_true - y_mean) * (y_pred - y_pred_mean))
        slope = cov / var_y
        intercept = y_pred_mean - slope * y_mean
    return {
        "epoch": int(epoch),
        "y_true": y_true.tolist(),
        "y_pred_mean": y_pred.tolist(),
        "slope": float(slope),
        "intercept": float(intercept),
    }



def _schedule_wall_breakpoints(effective_epochs):
    """Wall-clock breakpoints for effective fractions 0.5, 0.7, 0.9, 1.0.

    Storage/resume use true (wall) epochs. The schedule stretches wall time so
    ∫ (lr/lr0) d(wall) equals `effective_epochs`, with lr = lr0/{1,3,8,9}.
    """
    E = float(effective_epochs)
    w1 = 0.5 * E                 # end of lr0/1  (effective 0.5)
    w2 = w1 + 0.2 * E * 3        # end of lr0/3  (effective 0.7)
    w3 = w2 + 0.2 * E * 8        # end of lr0/8  (effective 0.9)
    w4 = w3 + 0.1 * E * 9        # end of lr0/9  (effective 1.0)
    return w1, w2, w3, w4


def _schedule_wall_epochs_for_effective(effective_epochs):
    """Total true/wall epochs so ∫ (lr/lr0) d(wall) = effective_epochs."""
    *_, w4 = _schedule_wall_breakpoints(effective_epochs)
    return int(math.ceil(w4))


def _schedule_lr_divisor_from_wall(wall_epoch, effective_epochs):
    """Map true wall epoch → LR divisor (1, 3, 8, or 9)."""
    w1, w2, w3, _w4 = _schedule_wall_breakpoints(effective_epochs)
    w = float(wall_epoch)
    if w < w1:
        return 1.0
    if w < w2:
        return 3.0
    if w < w3:
        return 8.0
    return 9.0


def _parse_schedule_divisors(spec):
    """Parse '2,3,5' → (2.0, 3.0, 5.0). Empty / None → None."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        divs = tuple(float(x) for x in spec)
    else:
        text = str(spec).strip()
        if not text:
            return None
        divs = tuple(float(x) for x in text.split(","))
    if not divs or any(d <= 0 for d in divs):
        raise ValueError(f"schedule divisors must be positive, got {spec!r}")
    return divs


def _equal_wall_lr_divisor(wall_epoch, wall_epochs, divisors):
    """Equal wall-time phases: phase i uses lr0/divisors[i], no stretch."""
    n = len(divisors)
    if wall_epochs <= 0:
        return float(divisors[-1])
    # epoch in 1..wall_epochs during training; treat 0 as phase 0
    w = max(0.0, float(wall_epoch))
    frac = min(w / float(wall_epochs), 0.999999)
    idx = min(int(frac * n), n - 1)
    return float(divisors[idx])


def _schedule_effective_from_wall(wall_epoch, effective_epochs):
    """Map true wall epoch → lr0-equivalent effective progress (logging only)."""
    E = float(effective_epochs)
    w1, w2, w3, w4 = _schedule_wall_breakpoints(effective_epochs)
    w = float(wall_epoch)
    if w <= 0:
        return 0.0
    if w <= w1:
        return w
    if w <= w2:
        return 0.5 * E + (w - w1) / 3.0
    if w <= w3:
        return 0.7 * E + (w - w2) / 8.0
    if w <= w4:
        return 0.9 * E + (w - w3) / 9.0
    return E + (w - w4) / 9.0


def _load_training_state_from_dir(model, source_dir, device, pred_vs_true_path=None):
    """Load model weights + epoch/history from source_dir. Returns start_epoch and histories."""
    source_dir = Path(source_dir)
    checkpoint_path = source_dir / "checkpoint.pt"
    model_checkpoint = source_dir / "model.pt"
    start_epoch = 0
    eigenvalues_over_time = {}
    losses = {}
    loss_stds = {}
    test_mse_over_time = {}
    cubic_learnability_over_time = {}
    pred_vs_true = {}
    if pred_vs_true_path is None:
        pred_vs_true_path = source_dir / "pred_vs_true.json"

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
    elif model_checkpoint.exists():
        print(f"Loading model from {model_checkpoint}")
        state_dict = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"No checkpoint.pt or model.pt in {source_dir}")

    eigenvalues_path = source_dir / "eigenvalues_over_time.json"
    losses_path = source_dir / "losses.json"
    source_pred_path = source_dir / "pred_vs_true.json"

    if eigenvalues_path.exists():
        with open(eigenvalues_path, "r") as f:
            eigenvalues_over_time = json.load(f)
        if start_epoch == 0 and eigenvalues_over_time:
            start_epoch = max(int(k) for k in eigenvalues_over_time.keys())
            print(f"Resuming from epoch {start_epoch}")
    if losses_path.exists():
        with open(losses_path, "r") as f:
            loss_data = json.load(f)
            losses = {int(k): v for k, v in loss_data.get("losses", {}).items()}
            loss_stds = {int(k): v for k, v in loss_data.get("loss_stds", {}).items()}
            test_mse_over_time = {
                int(k): v for k, v in loss_data.get("test_mse", {}).items()
            }
            cubic_learnability_over_time = {
                int(k): v for k, v in loss_data.get("cubic_learnability", {}).items()
            }
    if source_pred_path.exists():
        with open(source_pred_path, "r") as f:
            pred_vs_true = {int(k): v for k, v in json.load(f).items()}

    return (
        start_epoch,
        eigenvalues_over_time,
        losses,
        loss_stds,
        test_mse_over_time,
        cubic_learnability_over_time,
        pred_vs_true,
    )


def init_w0_target_from_lwt(model, lwt, seed=None):
    """Resample teacher-axis columns of W0 ~ N(0, lwt); leave perp and A unchanged.

    Theory's target second moment is lWT = E[w_0^2]. For the VGA point used here
    muW≈0 so a Gaussian with variance lWT matches the mixture second moment.
    Langevin prior / weight decay stay isotropic (s0/d); this only seeds W0.
    """
    lwt = float(lwt)
    if lwt <= 0:
        raise ValueError(f"--init-lwt must be positive, got {lwt}")
    std = lwt ** 0.5
    with torch.no_grad():
        w0 = model.W0  # (ens, n1, d)
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(int(seed))
            draw = torch.randn(w0.shape[:-1], generator=g) * std
            w0[..., 0].copy_(draw.to(device=w0.device, dtype=w0.dtype))
        else:
            w0[..., 0].normal_(0.0, std)
        emp = w0[..., 0].pow(2).mean().item()
    print(
        f"  init-lwt={lwt:g}: resampled W0[..., 0] ~ N(0, {lwt:g}) "
        f"(std={std:.4g}, empirical ⟨w0²⟩={emp:.4g})"
    )

def train_fcn2(d, P, N, eps=0.03, epochs=10_000_000, log_interval=100_000, ens=50,
               device_str="cuda:0", base_lr=1e-5, temperature=0.02, chi=1.0,
               s0=1.0, run_dir=None, writer=None, dataset_seed=42,
               activation="erf", classic=False, batch_size=None, warm_start=None,
               resume_from=None, schedule=False, schedule_divisors=None,
               extra_epochs=0, output_dir=None,
               snapshot_a_interval=None, snapshot_a_burnin=0, init_lwt=None):
    """Train 2-layer erf network and track H eigenvalues.
    
    Args:
        d: Input dimension
        P: Number of training samples
        N: Hidden layer width
        epochs: Epoch budget. With stretched --schedule this is effective (lr0-equivalent);
            with --schedule-divisors or no schedule, this is true wall epochs.
        log_interval: Log eigenvalues every N epochs
        device_str: Device string
        lr: Learning rate
        temperature: Base temperature for weight decay and Langevin noise
        chi: Scaling factor; effective temperature = temperature / chi
        s0: Scaling factor for the read-in weight variance; sigmaW0 is derived as s0 / d
        run_dir: Directory to save checkpoints and results
        writer: TensorBoard writer
        dataset_seed: Random seed for data generation
        batch_size: Minibatch size for Langevin. None or >=P means full batch.
            Gradients are scaled by P/B; lr stays base_lr/P.
        warm_start: Optional checkpoint/model path for weight initialization when
            this run has no existing checkpoint.
        resume_from: Optional prior run directory. Used only when run_dir has no
            checkpoint yet: loads weights, epoch, and history from that directory,
            then continues training with this run's base_lr into run_dir.
        schedule: If True (and schedule_divisors is None), stretch wall epochs so
            ∫(lr/lr0)=--epochs and set lr to lr0/3, lr0/8, lr0/9 at effective 0.5/0.7/0.9.
        schedule_divisors: If set (e.g. (2,3,5)), use equal wall-time phases with
            lr=lr0/div; wall budget equals --epochs (no stretch). Overrides --schedule.
        extra_epochs: Additional true/wall epochs after the scheduled (or --epochs)
            budget, held at the last learning rate.
        output_dir: Optional explicit output directory (overrides auto naming).
        snapshot_a_interval: If set, save A (and W0) every this many wall epochs under
            run_dir/A_snapshots/ after snapshot_a_burnin steps past resume.
        snapshot_a_burnin: Extra wall epochs after resume before the first A snapshot.
        init_lwt: If set, on a fresh run (no checkpoint / resume / warm-start) resample
            W0 teacher-axis entries ~ N(0, init_lwt). Prior/wd unchanged.
        
    Returns:
        (final_eigenvalues, eigenvalues_over_time, run_dir)
    """
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    sigmaW0_value = s0 / d
    equal_divs = _parse_schedule_divisors(schedule_divisors)
    use_stretched_schedule = bool(schedule) and equal_divs is None
    use_equal_schedule = equal_divs is not None

    # Setup directory
    if run_dir is None and output_dir is not None:
        run_dir = Path(output_dir)
    if run_dir is None:
        if classic:
            run_dir = Path(__file__).parent / f"d{d}_P{P}_N{N}_chi_{chi}_lr_{base_lr}_T_{temperature}_seed_{dataset_seed}_eps_{eps}"
        else:
            run_dir = Path(__file__).parent / f"d{d}_P{P}_N{N}_chi_{chi}_lr_{base_lr}_T_{temperature}_seed_{dataset_seed}_eps_{eps}_s0_{s0}_sigmaW0_{sigmaW0_value}"
        if use_equal_schedule:
            tag = "_".join(str(int(x)) if float(x).is_integer() else f"{x:g}" for x in equal_divs)
            run_dir = Path(str(run_dir) + f"_schedule_{tag}")
        elif use_stretched_schedule:
            run_dir = Path(str(run_dir) + "_schedule")
        if init_lwt is not None:
            run_dir = Path(str(run_dir) + f"_initlWT_{float(init_lwt):g}")
    run_dir = Path(run_dir)
    run_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nTraining 2-layer erf network:")
    print(f"  d={d}, P={P}, N={N}")


    effective_temperature = temperature / chi 
    lr = base_lr / P
    if batch_size is None or batch_size <= 0 or batch_size >= P:
        B = P
        use_minibatch = False
    else:
        B = int(batch_size)
        use_minibatch = True
    grad_scale = P / B


    print(f"  lr={lr:.6e}, T={temperature:.6f}, chi={chi:.6f}, T_eff={effective_temperature:.6f}")
    print(f"  batch_size={B} ({'minibatch, grad scale P/B=' + f'{grad_scale:.4g}' if use_minibatch else 'full batch'})")
    if extra_epochs:
        extra_epochs = int(extra_epochs)
    else:
        extra_epochs = 0
    if use_equal_schedule:
        nph = len(equal_divs)
        phase = epochs / nph
        parts = ", ".join(
            f"lr0/{d:g} for wall [{int(i*phase)}, {int((i+1)*phase)})"
            for i, d in enumerate(equal_divs)
        )
        print(
            f"  schedule=equal-wall divisors={equal_divs}: {parts}; "
            f"wall budget = --epochs={epochs} (no stretch)"
        )
        if extra_epochs:
            print(
                f"  extra-epochs={extra_epochs}: after schedule, hold final lr "
                f"(lr0/{equal_divs[-1]:g}) for {extra_epochs} more wall steps"
            )
    elif use_stretched_schedule:
        wall_budget = _schedule_wall_epochs_for_effective(epochs)
        print(
            f"  schedule=ON (stretched): lr → lr0/3 @50%, lr0/8 @70%, lr0/9 @90% effective; "
            f"--epochs={epochs} is effective budget (~{wall_budget} wall steps from scratch)"
        )
        if extra_epochs:
            print(
                f"  extra-epochs={extra_epochs}: after the schedule, hold final lr (lr0/9) "
                f"for {extra_epochs} more wall steps (wall end={wall_budget + extra_epochs})"
            )
    elif extra_epochs:
        print(f"  extra-epochs={extra_epochs}: continue {extra_epochs} wall steps past --epochs")
    print(f"  Output: {run_dir}")
    print(f"  Device: {device}")
    
    # Generate data: Y = X[:, 0] (first dimension as target)
    torch.manual_seed(dataset_seed)
    X = torch.randn(P, d, device=device)
    z = X[:, 0].unsqueeze(-1)  # (P, 1)
    Y = z + eps * _he3(z)  # (P, 1)
    # Model
    ens = ens  # ensemble size
    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation=activation,
        weight_initialization_variance=(sigmaW0_value, 1/(N * chi))
    ).to(device)
    
    # Prefer checkpoint in run_dir unless resume_from is further ahead (e.g. a
    # fresh epoch-0 stub in a new-lr dir should not block loading an old-lr run).
    checkpoint_path = run_dir / "checkpoint.pt"
    model_checkpoint = run_dir / "model.pt"
    start_epoch = 0
    eigenvalues_over_time = {}
    losses = {}
    loss_stds = {}
    test_mse_over_time = {}
    cubic_learnability_over_time = {}
    pred_vs_true = {}
    pred_vs_true_path = run_dir / "pred_vs_true.json"

    def _peek_epoch(path):
        path = Path(path)
        ckpt = path / "checkpoint.pt"
        if not ckpt.exists():
            eigs = path / "eigenvalues_over_time.json"
            if eigs.exists():
                with open(eigs, "r") as f:
                    data = json.load(f)
                return max((int(k) for k in data.keys()), default=None) if data else None
            return None
        try:
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(ckpt, map_location="cpu")
        return int(data.get("epoch", 0))

    dest_ep = _peek_epoch(run_dir) if (checkpoint_path.exists() or model_checkpoint.exists()) else None
    resume_ep = _peek_epoch(resume_from) if resume_from is not None else None
    use_resume_from = (
        resume_from is not None
        and resume_ep is not None
        and (dest_ep is None or resume_ep > dest_ep)
    )

    if use_resume_from:
        resume_path = Path(resume_from)
        if dest_ep is not None:
            print(
                f"Dest checkpoint epoch {dest_ep} < resume-from epoch {resume_ep}; "
                f"loading from {resume_path}"
            )
        else:
            print(f"No usable checkpoint in {run_dir}; resuming state from {resume_path}")
        (
            start_epoch,
            eigenvalues_over_time,
            losses,
            loss_stds,
            test_mse_over_time,
            cubic_learnability_over_time,
            pred_vs_true,
        ) = _load_training_state_from_dir(model, resume_path, device, pred_vs_true_path)
        if eigenvalues_over_time:
            with open(run_dir / "eigenvalues_over_time.json", "w") as f:
                json.dump(eigenvalues_over_time, f)
        if losses or loss_stds or test_mse_over_time or cubic_learnability_over_time:
            with open(run_dir / "losses.json", "w") as f:
                json.dump(
                    {
                        "losses": {str(k): v for k, v in losses.items()},
                        "loss_stds": {str(k): v for k, v in loss_stds.items()},
                        "test_mse": {str(k): v for k, v in test_mse_over_time.items()},
                        "cubic_learnability": {
                            str(k): v for k, v in cubic_learnability_over_time.items()
                        },
                    },
                    f,
                )
        if pred_vs_true:
            with open(pred_vs_true_path, "w") as f:
                json.dump({str(k): v for k, v in pred_vs_true.items()}, f, indent=2)
    elif checkpoint_path.exists() or model_checkpoint.exists():
        (
            start_epoch,
            eigenvalues_over_time,
            losses,
            loss_stds,
            test_mse_over_time,
            cubic_learnability_over_time,
            pred_vs_true,
        ) = _load_training_state_from_dir(model, run_dir, device, pred_vs_true_path)
    elif warm_start is not None:
        load_warm_start_into_model(model, warm_start, device)
        start_epoch = 0
        print("Warm-start applied; training from epoch 0 at the new (d, P, N).")
    elif init_lwt is not None:
        # Fresh run only (checkpoint / resume / warm-start take precedence above).
        init_w0_target_from_lwt(model, init_lwt, seed=dataset_seed + 17)

    if use_stretched_schedule:
        wall_target = _schedule_wall_epochs_for_effective(epochs)
        eff_now = _schedule_effective_from_wall(start_epoch, epochs)
        print(
            f"  schedule wall target={wall_target} (from --epochs={epochs} effective); "
            f"resume wall epoch={start_epoch} → effective≈{eff_now:g}"
        )
        if start_epoch >= wall_target:
            if extra_epochs > 0 and start_epoch < wall_target + extra_epochs:
                print(
                    f"  Past scheduled wall budget; continuing extra wall epochs "
                    f"{start_epoch} → {wall_target + extra_epochs} at final lr (lr0/9)"
                )
            else:
                print("  Already at/past scheduled wall epoch budget; nothing to train.")
    elif use_equal_schedule:
        wall_target = epochs
        print(
            f"  equal-wall schedule target={wall_target}; "
            f"resume wall epoch={start_epoch}; divisors={equal_divs}"
        )
        if start_epoch >= wall_target:
            if extra_epochs > 0 and start_epoch < wall_target + extra_epochs:
                print(
                    f"  Past schedule; continuing extra wall epochs "
                    f"{start_epoch} → {wall_target + extra_epochs} at lr0/{equal_divs[-1]:g}"
                )
            else:
                print("  Already at/past scheduled wall epoch budget; nothing to train.")
    
    model.train()
    
    # Weight decay: lambda_W0 = d * T, lambda_A = N * T
    wd_W0 = (1/sigmaW0_value) * effective_temperature
    wd_A = N * effective_temperature * chi
    

    # Log predictions on the same cadence as eigenvalue / projection logging
    pred_log_interval = log_interval

    snapshot_dir = None
    snapshot_manifest = {"interval": None, "burnin": 0, "epochs": []}
    snapshot_manifest_path = None
    if snapshot_a_interval is not None and int(snapshot_a_interval) > 0:
        snapshot_a_interval = int(snapshot_a_interval)
        snapshot_a_burnin = max(0, int(snapshot_a_burnin))
        snapshot_dir = run_dir / "A_snapshots"
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot_manifest_path = snapshot_dir / "manifest.json"
        if snapshot_manifest_path.exists():
            with open(snapshot_manifest_path, "r") as f:
                snapshot_manifest = json.load(f)
            snapshot_manifest.setdefault("epochs", [])
        snapshot_manifest["interval"] = snapshot_a_interval
        snapshot_manifest["burnin"] = snapshot_a_burnin
        snapshot_manifest["resume_epoch"] = int(start_epoch)
        print(
            f"  A snapshots: every {snapshot_a_interval} wall epochs after "
            f"+{snapshot_a_burnin} burn-in -> {snapshot_dir}"
        )
    
    # Large eval set for eigenvalues / projections / test metrics
    torch.manual_seed(dataset_seed + 1)
    Xinf = torch.randn(3000, d, device=device)
    z_inf = Xinf[:, 0].unsqueeze(-1)
    Yinf = z_inf + eps * _he3(z_inf)
    
    # Compute initial eigenvalues
    if start_epoch == 0 and 0 not in eigenvalues_over_time:
        with torch.no_grad():
            try:
                # Move to CPU for eigenvalue computation
                model_cpu = model.cpu()
                Xinf_cpu = Xinf.cpu()
                eigenvalues = model_cpu.H_eig(Xinf_cpu, Xinf_cpu).cpu().numpy()
                eigenvalues_over_time[0] = eigenvalues.tolist()
                print(f"  Epoch {0:7d} (init): max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
                # Move model back to original device
                model.to(device)
            except Exception as e:
                print(f"  Warning: Could not compute initial eigenvalues: {e}")
                model.to(device)
    print("Initial loss")
    with torch.no_grad():
        output = model(X)
        diff = output - Y
        per_ensemble_loss = torch.sum(diff * diff, dim=0)
        loss = per_ensemble_loss.sum()
        loss_avg = loss.item() / ens
        loss_std = per_ensemble_loss.std().item()
        print(f"  loss={loss_avg:.6e}±{loss_std:.6e}")
        try:
            te_mse0 = mean_mse(model, Xinf, Yinf)
            he3_L0 = cubic_learnability(model, Xinf, Yinf)
            test_mse_over_time[0] = float(te_mse0)
            cubic_learnability_over_time[0] = float(he3_L0)
            print(f"  Test metrics - MSE={te_mse0:.6e}, cubic_L3={he3_L0:.4f}")
        except Exception as e:
            print(f"  Warning: Could not compute initial test metrics: {e}")
            te_mse0 = None
            he3_L0 = None
        # Log to TensorBoard (epoch 0 only; thereafter at log_interval)
        if writer is not None:
            writer.add_scalar('loss/sum_total', loss.item(), 0)
            writer.add_scalar('loss/mean', loss_avg, 0)
            writer.add_scalar('loss/std', loss_std, 0)
            if te_mse0 is not None:
                writer.add_scalar('metrics/test_mse', te_mse0, 0)
            if he3_L0 is not None:
                writer.add_scalar('metrics/cubic_learnability', he3_L0, 0)
        if 0 % pred_log_interval == 0 and 0 not in pred_vs_true:
            rec = _pred_record(0, Y, output)
            pred_vs_true[0] = rec
            with open(pred_vs_true_path, "w") as f:
                json.dump(pred_vs_true, f, indent=2)
    last_output = output

    # Training loop. Storage/resume use true wall epochs.
    # --schedule: stretched wall ≈ 3.6 * --epochs; --schedule-divisors: wall = --epochs.
    if use_stretched_schedule:
        end_epoch = _schedule_wall_epochs_for_effective(epochs) + extra_epochs
    else:
        end_epoch = epochs + extra_epochs

    lr_divisor = 1.0
    for epoch in range(start_epoch, end_epoch + 1):
        if epoch > 0:
            if use_equal_schedule:
                lr_divisor = _equal_wall_lr_divisor(epoch, epochs, equal_divs)
            elif use_stretched_schedule:
                lr_divisor = _schedule_lr_divisor_from_wall(epoch, epochs)
            else:
                lr_divisor = 1.0
            lr = (base_lr / lr_divisor) / P
            # 2. Re-calculate noise scale based on the smoothly changing LR
            torch.manual_seed(epoch + dataset_seed)  # For reproducibility of noise
            noise_scale = np.sqrt(2.0 * lr * effective_temperature)

            if use_minibatch:
                idx = torch.randint(0, P, (B,), device=device)
                Xb = X[idx]
                Yb = Y[idx]
                scale = grad_scale
            else:
                Xb = X
                Yb = Y
                scale = 1.0

            # Forward pass
            output = model(Xb)  # (B, ens)
            
            # Loss per ensemble (scaled to full-sum estimator when minibatching)
            diff = output - Yb  # (B, ens)
            per_ensemble_loss = scale * torch.sum(diff * diff, dim=0)  # (ens,)
            loss = per_ensemble_loss.sum() 
            
            loss_avg = loss.item() / ens
            loss_std = per_ensemble_loss.std().item()
            
            # Backward
            model.zero_grad()
            loss.backward()
            
            # Langevin update
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    
                    # Weight decay
                    if 'W0' == name:
                        wd = wd_W0
                    elif 'A' == name:
                        wd = wd_A
                    else:
                        wd = 0
                    
                    # Gradient + weight decay + noise
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(-lr * param.grad)
                    param.add_(-lr * wd * param.data)
                    param.add_(noise)
                last_output = output

        # Late Langevin snapshots of A (and W0) for posterior averaging
        if (
            snapshot_dir is not None
            and epoch > 0
            and (epoch - start_epoch) >= snapshot_a_burnin
            and (epoch - start_epoch - snapshot_a_burnin) % snapshot_a_interval == 0
        ):
            with torch.no_grad():
                snap_path = snapshot_dir / f"epoch_{epoch:08d}.pt"
                torch.save(
                    {
                        "epoch": int(epoch),
                        "A": model.A.detach().cpu().clone(),
                        "W0": model.W0.detach().cpu().clone(),
                    },
                    snap_path,
                )
            if epoch not in snapshot_manifest["epochs"]:
                snapshot_manifest["epochs"].append(int(epoch))
                with open(snapshot_manifest_path, "w") as f:
                    json.dump(snapshot_manifest, f, indent=2)
            if epoch % max(snapshot_a_interval, 1) == 0 or epoch == start_epoch + snapshot_a_burnin:
                print(f"  Saved A snapshot: {snap_path.name}")
        
        # Logging and checkpointing (TensorBoard only on this cadence)
        if epoch % log_interval == 0:
            if epoch > 0 and writer is not None:
                writer.add_scalar('loss/sum_total', loss.item(), epoch)
                writer.add_scalar('learning_rate/lr', lr, epoch)
                if use_minibatch:
                    writer.add_scalar('training/batch_size', B, epoch)
                    writer.add_scalar('training/grad_scale', scale, epoch)
                if use_stretched_schedule or use_equal_schedule:
                    writer.add_scalar('learning_rate/schedule_divisor', lr_divisor, epoch)
                    if use_stretched_schedule:
                        writer.add_scalar(
                            'learning_rate/effective_epoch',
                            _schedule_effective_from_wall(epoch, epochs),
                            epoch,
                        )

            with torch.no_grad():
                # Test MSE + residualized cubic learnability on fixed eval set
                te_mse = None
                he3_L = None
                try:
                    te_mse = mean_mse(model, Xinf, Yinf)
                    he3_L = cubic_learnability(model, Xinf, Yinf)
                    test_mse_over_time[epoch] = float(te_mse)
                    cubic_learnability_over_time[epoch] = float(he3_L)
                    if writer is not None:
                        writer.add_scalar('metrics/test_mse', te_mse, epoch)
                        writer.add_scalar('metrics/cubic_learnability', he3_L, epoch)
                    print(
                        f"  Test metrics - MSE={te_mse:.6e}, cubic_L3={he3_L:.4f}"
                    )
                except Exception as e:
                    print(f"  Warning: Could not compute test metrics at epoch {epoch}: {e}")

                # Compute eigenvalues
                try:
                    # Move to CPU for eigenvalue computation
                    model_cpu = model.cpu()
                    Xinf_cpu = Xinf.cpu()
                    eigenvalues = model_cpu.H_eig(Xinf_cpu, Xinf_cpu).cpu().numpy()
                    eigenvalues_over_time[epoch] = eigenvalues.tolist()
                    # Move model back to original device
                    model.to(device)
                    
                    # Log to TensorBoard
                    if writer is not None:
                        # # Individual eigenvalues
                        # eigenvalue_dict = {f'eig_{i}': float(eigenvalues[i]) 
                        #                  for i in range(len(eigenvalues))}
                        # writer.add_scalars('eigenvalues/all', eigenvalue_dict, epoch)
                        
                        # Eigenvalue statistics
                        writer.add_scalar('eigenvalues/perp_mean', float(eigenvalues[1:].mean()), epoch)
                        writer.add_scalar('eigenvalues/max', float(eigenvalues.max()), epoch)
                        writer.add_scalar('eigenvalues/min', float(eigenvalues.min()), epoch)
                        writer.add_scalar('eigenvalues/std', float(eigenvalues.std()), epoch)
                        
                        # Eigenvalue histogram
                        writer.add_histogram('eigenvalues/distribution', eigenvalues, epoch)
                        
                except Exception as e:
                    print(f"  Warning: Could not compute eigenvalues at epoch {epoch}: {e}")
                    eigenvalues = None
                    model.to(device)
                
                # Compute h0_activation projections
                try:
                    import traceback
                    model_cpu = model.cpu()
                    Xinf_cpu = Xinf.cpu()
                    
                    # Get hidden layer activations: shape (P, ens, n1)
                    h0_act = model_cpu.h0_activation(Xinf_cpu)
                    P_dim = h0_act.shape[0]
                    
                    # Compute projection directions
                    x0_target = Xinf_cpu[:, 0]
                    x0_target_normed = x0_target 
                    x1_perp = Xinf_cpu[:, 1] if d > 1 else torch.randn_like(Xinf_cpu[:, 0])
                    x1_perp_normed = x1_perp / x1_perp.norm() 
                    
                    # Hermite cubic polynomials for target and perp: (x^3 - 3x)/sqrt(6)
                    h3_target = (x0_target**3 - 3.0 * x0_target) 
                    h3_target_normed = h3_target 
                    h3_perp = (x1_perp**3 - 3.0 * x1_perp)
                    h3_perp_normed = h3_perp 
                    
                    # Project activations onto target/perp directions per ensemble
                    # h0_act shape: (P, ens, n1)
                    proj_lin_target = torch.einsum('pqn,p->qn', h0_act, x0_target_normed) / P_dim
                    proj_lin_perp = torch.einsum('pqn,p->qn', h0_act, x1_perp_normed) / P_dim
                    proj_cubic_target = torch.einsum('pqn,p->qn', h0_act, h3_target_normed)  / P_dim
                    proj_cubic_perp = torch.einsum('pqn,p->qn', h0_act, h3_perp_normed) / P_dim
                    
                    # Compute variances
                    var_lin_target = float(torch.var(proj_lin_target).item())
                    var_lin_perp = float(torch.var(proj_lin_perp).item())
                    var_cubic_target = float(torch.var(proj_cubic_target).item())
                    var_cubic_perp = float(torch.var(proj_cubic_perp).item())
                    
                    # Log variances to TensorBoard
                    if writer is not None:
                        writer.add_scalar('Projections/He1_target_var', var_lin_target, epoch)
                        writer.add_scalar('Projections/He1_perp_var', var_lin_perp, epoch)
                        writer.add_scalar('Projections/He3_target_var', var_cubic_target, epoch)
                        writer.add_scalar('Projections/He3_perp_var', var_cubic_perp, epoch)
                        
                        # Log histograms to TensorBoard
                        writer.add_histogram('Projections/He1_target', proj_lin_target, epoch)
                        writer.add_histogram('Projections/He1_perp', proj_lin_perp, epoch)
                        writer.add_histogram('Projections/He3_target', proj_cubic_target, epoch)
                        writer.add_histogram('Projections/He3_perp', proj_cubic_perp, epoch)
                        
                    print(f"  Projections - He1_target: {var_lin_target:.3g}, He1_perp: {var_lin_perp:.3g}, He3_target: {var_cubic_target:.3g}, He3_perp: {var_cubic_perp:.3g}")
                    
                    model.to(device)
                except Exception as e:
                    traceback.print_exc()
                    print(f"  Warning: Could not compute/log projections at epoch {epoch}: {e}")
                    model.to(device)
                
                # Log loss and eigenvalues
                if epoch > 0:
                    
                    losses[epoch] = float(loss_avg)
                    loss_stds[epoch] = float(loss_std)
                    
                    if writer is not None:
                        writer.add_scalar('loss/mean', loss_avg, epoch)
                        writer.add_scalar('loss/std', loss_std, epoch)
                    
                    if eigenvalues is not None:
                        print(f"  Epoch {epoch:7d}: loss={loss_avg:.6e}±{loss_std:.6e}, "
                              f"max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
                    else:
                        print(f"  Epoch {epoch:7d}: loss={loss_avg:.6e}±{loss_std:.6e}")
                else:
                    if eigenvalues is not None:
                        print(f"  Epoch {epoch:7d} (init): max_eig={eigenvalues.max():.6f}, "
                              f"mean_eig={eigenvalues.mean():.6f}")

                # Save checkpoint
                if epoch % (2 * log_interval) == 0:
                    # Save model state
                    torch.save(model.state_dict(), run_dir / "model.pt")
                    
                    # Save full checkpoint with metadata
                    checkpoint = {
                        'epoch': epoch,  # true wall epoch (resume key)
                        'model_state_dict': model.state_dict(),
                        'config': {
                            'd': d, 'P': P, 'N': N, 'ens': ens,
                            'lr': float(lr), 'temperature': float(temperature),
                            'chi': float(chi), 'effective_temperature': float(effective_temperature),
                            's0': float(s0), 'sigmaW0': float(sigmaW0_value),
                            'schedule': bool(use_stretched_schedule),
                            'schedule_divisors': list(equal_divs) if equal_divs else None,
                            'base_lr': float(base_lr),
                            'effective_epochs_target': int(epochs),
                            'effective_epoch': (
                                float(_schedule_effective_from_wall(epoch, epochs))
                                if use_stretched_schedule else float(epoch)
                            ),
                            'lr_divisor': float(lr_divisor) if epoch > 0 else 1.0,
                        },
                        'loss': float(loss_avg) if epoch > 0 else None,
                        'loss_std': float(loss_std) if epoch > 0 else None,
                        'test_mse': float(te_mse) if te_mse is not None else None,
                        'cubic_learnability': float(he3_L) if he3_L is not None else None,
                    }
                    if eigenvalues is not None:
                        checkpoint['eigenvalues'] = eigenvalues.tolist()
                    torch.save(checkpoint, run_dir / "checkpoint.pt")
                    
                    # Save eigenvalues and losses
                    with open(run_dir / "eigenvalues_over_time.json", "w") as f:
                        json.dump(eigenvalues_over_time, f, indent=2)
                    
                    with open(run_dir / "losses.json", "w") as f:
                        json.dump(
                            {
                                "losses": losses,
                                "loss_stds": loss_stds,
                                "test_mse": test_mse_over_time,
                                "cubic_learnability": cubic_learnability_over_time,
                            },
                            f,
                            indent=2,
                        )
                    with open(pred_vs_true_path, "w") as f:
                        json.dump(pred_vs_true, f, indent=2)

        if epoch % pred_log_interval == 0 and epoch not in pred_vs_true:
            with torch.no_grad():
                if use_minibatch:
                    full_output = model(X)
                else:
                    full_output = last_output
                rec = _pred_record(epoch, Y, full_output)
                pred_vs_true[epoch] = rec
                with open(pred_vs_true_path, "w") as f:
                    json.dump(pred_vs_true, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), run_dir / "model_final.pt")

    # Final flush of metrics histories (covers last log_interval if not on 2*log cadence)
    with open(run_dir / "losses.json", "w") as f:
        json.dump(
            {
                "losses": losses,
                "loss_stds": loss_stds,
                "test_mse": test_mse_over_time,
                "cubic_learnability": cubic_learnability_over_time,
            },
            f,
            indent=2,
        )
    with open(run_dir / "eigenvalues_over_time.json", "w") as f:
        json.dump(eigenvalues_over_time, f, indent=2)
    
    # Save config
    config = {
        "d": d, "P": P, "N": N, "eps": eps,
        "lr": float(lr), "temperature": float(temperature),
        "chi": float(chi), "effective_temperature": float(temperature / chi),
        "s0": float(s0), "sigmaW0": float(sigmaW0_value),
        "epochs": epochs, "ens": ens,
        "batch_size": int(B),
        "minibatch": bool(use_minibatch),
        "warm_start": str(warm_start) if warm_start is not None else None,
        "schedule": bool(use_stretched_schedule),
        "schedule_divisors": list(equal_divs) if equal_divs else None,
        "base_lr": float(base_lr),
        "effective_epochs_target": int(epochs),
        "wall_epochs_target": int(end_epoch),
        "extra_epochs": int(extra_epochs),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Final eigenvalues
    final_eigenvalues = None
    if end_epoch in eigenvalues_over_time:
        final_eigenvalues = np.array(eigenvalues_over_time[end_epoch])
    
    return final_eigenvalues, eigenvalues_over_time, run_dir


def plot_eigenvalues_over_time(run_dir):
    """Plot eigenvalue evolution over training."""
    eigenvalues_path = run_dir / "eigenvalues_over_time.json"
    if not eigenvalues_path.exists():
        print(f"No eigenvalues file found at {eigenvalues_path}")
        return
    
    with open(eigenvalues_path, "r") as f:
        eig_data = json.load(f)
    
    epochs = sorted([int(k) for k in eig_data.keys()])
    eigenvalues = np.array([eig_data[str(e)] for e in epochs])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each eigenvalue over time
    for i in range(eigenvalues.shape[1]):
        ax.plot(epochs, eigenvalues[:, i], alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"H Eigenvalues over Training\n{run_dir.name}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(run_dir / "eigenvalues_over_time.png", dpi=150)
    plt.close(fig)
    print(f"Saved eigenvalue plot to {run_dir / 'eigenvalues_over_time.png'}")


def main():
    parser = argparse.ArgumentParser(description='Train 2-layer erf network')
    parser.add_argument('--d', type=int, default=10, help='Input dimension')
    parser.add_argument('--P', type=int, default=30, help='Number of samples')
    parser.add_argument('--N', type=int, default=256, help='Hidden layer width')
    parser.add_argument('--epochs', type=int, default=10_000_000, help='Number of epochs')
    parser.add_argument('--log-interval', type=int, default=100_000, help='Logging interval for eigenvalues, projections, predictions, and checkpoints')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Base temperature for Langevin')
    parser.add_argument('--chi', type=float, default=1.0, help='Scale factor; effective temperature = temperature/chi')
    parser.add_argument('--s0', type=float, default=1.0, help='Scale factor for W0 variance; sigmaW0 is computed as s0/d')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--dataset-seed', type=int, default=42, help='Random seed for dataset generation')
    parser.add_argument('--ens', type=int, default=10, help='Ensemble size')
    parser.add_argument('--eps', type=float, default=0.03, help='Epsilon for cubic nonlinearity')
    parser.add_argument('--classic', action='store_true', help='Use classic s0=1.0 naming')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Minibatch size for Langevin. Omit or set >=P for full batch. '
             'Gradients use (P/B) scaling; lr stays base_lr/P.',
    )
    parser.add_argument(
        '--warm-start',
        type=str,
        default=None,
        help='Path to checkpoint.pt or model.pt for weight init when this run has no checkpoint.',
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Prior run directory to load checkpoint/history from when this run_dir has none. '
             'Continues with --lr into the (possibly new) classic/s0 run directory.',
    )
    parser.add_argument(
        '--schedule',
        action='store_true',
        help='LR schedule on stretched wall time: lr0/3, lr0/8, lr0/9 after effective '
             'fractions 0.5, 0.7, 0.9. --epochs is the lr0-equivalent budget; checkpoints '
             'and resume use true wall epochs. Appends _schedule to the run directory name. '
             'Ignored if --schedule-divisors is set.',
    )
    parser.add_argument(
        '--schedule-divisors',
        type=str,
        default=None,
        help='Equal wall-time LR schedule, no stretch. Comma-separated divisors, e.g. '
             '"2,3,5" → lr0/2, lr0/3, lr0/5 on equal thirds of --epochs (wall = --epochs). '
             'Overrides --schedule. Appends _schedule_2_3_5 to auto run dir name.',
    )
    parser.add_argument(
        '--extra-epochs',
        type=int,
        default=0,
        help='Additional true/wall epochs after the schedule/--epochs budget, held at the '
             'final learning rate. Resume continues from checkpoint.',
    )
    parser.add_argument('--output-dir', type=str, default=None, help='Directory for model checkpoints and run artifacts')
    parser.add_argument('--tensorboard-dir', type=str, default=None, help='Directory for TensorBoard event files')
    parser.add_argument(
        '--snapshot-A-interval',
        type=int,
        default=None,
        help='If set, save A (and W0) every N wall epochs under output-dir/A_snapshots/ '
             'after --snapshot-A-burnin steps past resume. For late posterior averages.',
    )
    parser.add_argument(
        '--snapshot-A-burnin',
        type=int,
        default=0,
        help='Wall epochs after resume before the first A snapshot (default 0).',
    )
    parser.add_argument(
        '--init-lwt',
        type=float,
        default=None,
        help='If set on a fresh run, resample W0 teacher-axis ~ N(0, init_lwt) '
             '(theory target second moment). Perp weights and A keep default init; '
             'Langevin prior/wd stay isotropic s0/d. Ignored if resuming/warm-starting.',
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Training 2-Layer ERF Network")
    print("="*60)
    
    # Setup TensorBoard
    sigmaW0_tag = args.s0 / args.d
    if args.tensorboard_dir is not None:
        tensorboard_dir = Path(args.tensorboard_dir)
    elif args.classic:
        tensorboard_dir = Path(__file__).parent / "minigrokkingruns" / f"d{args.d}_P{args.P}_N{args.N}_chi_{args.chi}_seed_{args.dataset_seed}_lr_{args.lr}_T_{args.temperature}_eps_{args.eps}"
    else:
        tensorboard_dir = Path(__file__).parent / "minigrokkingruns" / f"d{args.d}_P{args.P}_N{args.N}_chi_{args.chi}_seed_{args.dataset_seed}_lr_{args.lr}_T_{args.temperature}_eps_{args.eps}_s0_{args.s0}_sigmaW0_{sigmaW0_tag}"
    if args.tensorboard_dir is None:
        if args.schedule_divisors:
            tag = "_".join(
                str(int(float(x))) if float(x).is_integer() else f"{float(x):g}"
                for x in args.schedule_divisors.split(",")
            )
            tensorboard_dir = Path(str(tensorboard_dir) + f"_schedule_{tag}")
        elif args.schedule:
            tensorboard_dir = Path(str(tensorboard_dir) + "_schedule")
    tensorboard_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"TensorBoard logging to: {tensorboard_dir}")
    
    # Train
    final_eigs, eigs_over_time, run_dir = train_fcn2(
        d=args.d, P=args.P, N=args.N,eps=args.eps,
        epochs=args.epochs, log_interval=args.log_interval,
        device_str=args.device, base_lr=args.lr, temperature=args.temperature, chi=args.chi,
        s0=args.s0, writer=writer, dataset_seed=args.dataset_seed, ens=args.ens, classic=args.classic,
        batch_size=args.batch_size, warm_start=args.warm_start, resume_from=args.resume_from,
        schedule=args.schedule, schedule_divisors=args.schedule_divisors,
        extra_epochs=args.extra_epochs, output_dir=args.output_dir,
        snapshot_a_interval=args.snapshot_A_interval,
        snapshot_a_burnin=args.snapshot_A_burnin,
        init_lwt=args.init_lwt,
    )
    
    writer.close()
    
    # Plot
    plot_eigenvalues_over_time(run_dir)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
