import os
import sys
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path


# Hyperparameters
D = 100
CHI = 700
N = 700

# P → base_lr used for *continued* training (appears in the new run-dir name).
P_LR = {
    100: 0.005,
    400: 0.005,
    700: 0.005,
    1000: 0.005,
    4000: 0.005,
    6000: 0.001,
    8000: 0.001,
    10000: 0.001,
}

# P → base_lr of the *existing* classic run dirs to load checkpoints from.
# When P_LR[P] != P_LR_SOURCE[P], training writes to a new lr-tagged dir but
# bootstraps weights/epoch/history via --resume-from the source dir.
P_LR_SOURCE = {
    100: 0.005,
    400: 0.005,
    700: 0.005,
    1000: 0.005,
    4000: 0.05,
    6000: 0.05,
    8000: 0.05,
    10000: 0.05,
}

P_VALUES = sorted(P_LR)

TEMPERATURE = 0.1
ENSEMBLE_SIZE = 10
# Past the original 5M / ~15M checkpoints so cancelled jobs keep training on resume.
EPOCHS = 20_000_000
EPS = 0.074
DEVICE = "cuda:1"

SEEDS = [0]

MAX_PARALLEL_JOBS = 4

# Classic naming (no _s0_ / _sigmaW0_ suffixes). Matches in-flight run dirs.
USE_CLASSIC_NAMING = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_fcn2_erf.py")


def lr_for(p_val):
    return P_LR[int(p_val)]


def source_lr_for(p_val):
    return P_LR_SOURCE[int(p_val)]


def log_base_dir_for(lr):
    """Per-lr launcher log folder. Prefer legacy _s0_1.0 dir if it already exists."""
    classic = os.path.join(
        SCRIPT_DIR,
        f"launcher_logs_red_robin_sample_complexity_D{D}_chi{CHI}_eps{EPS}"
        f"_T{TEMPERATURE}_lr{lr}_ens{ENSEMBLE_SIZE}",
    )
    legacy = classic + "_s0_1.0"
    return legacy if os.path.isdir(legacy) else classic


def classic_run_dir(p_val, seed, lr=None):
    """Model/checkpoint directory used with --classic (no s0 in the name)."""
    if lr is None:
        lr = lr_for(p_val)
    return Path(SCRIPT_DIR) / (
        f"d{D}_P{p_val}_N{N}_chi_{float(CHI)}_lr_{lr}_T_{TEMPERATURE}"
        f"_seed_{seed}_eps_{EPS}"
    )


def checkpoint_epoch(run_dir):
    """Return checkpoint epoch under run_dir if present, else None."""
    run_dir = Path(run_dir)
    ckpt = run_dir / "checkpoint.pt"
    if not ckpt.exists():
        return None
    try:
        import torch

        try:
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(ckpt, map_location="cpu")
        return int(data.get("epoch", 0))
    except Exception as exc:
        print(f"  [warn] could not read {ckpt}: {exc}")
        return None


def make_cmd(p_val, seed):
    """Build the train_fcn2_erf.py command for a single P and seed."""
    lr = lr_for(p_val)
    src_lr = source_lr_for(p_val)
    dest_dir = classic_run_dir(p_val, seed, lr=lr)
    src_dir = classic_run_dir(p_val, seed, lr=src_lr)

    cmd = [
        sys.executable,
        "-u",
        TRAIN_SCRIPT,
        "--d",
        str(D),
        "--P",
        str(int(p_val)),
        "--chi",
        str(CHI),
        "--temperature",
        str(TEMPERATURE),
        "--N",
        str(N),
        "--lr",
        str(lr),
        "--device",
        DEVICE,
        "--epochs",
        str(EPOCHS),
        "--dataset-seed",
        str(seed),
        "--ens",
        str(ENSEMBLE_SIZE),
        "--eps",
        str(EPS),
    ]
    if USE_CLASSIC_NAMING:
        cmd.append("--classic")

    # Pass --resume-from whenever the source checkpoint is strictly ahead of dest
    # (covers new-lr dirs that only have an epoch-0 stub).
    if src_dir.resolve() != dest_dir.resolve():
        ep_src = checkpoint_epoch(src_dir)
        ep_dest = checkpoint_epoch(dest_dir)
        if ep_src is not None and (ep_dest is None or ep_src > ep_dest):
            cmd.extend(["--resume-from", str(src_dir)])

    return cmd


def effective_start_epoch(p_val, seed):
    """Epoch we will resume from: max of dest (new lr) and source (old lr)."""
    new_dir = classic_run_dir(p_val, seed, lr=lr_for(p_val))
    src_dir = classic_run_dir(p_val, seed, lr=source_lr_for(p_val))
    ep_new = checkpoint_epoch(new_dir)
    ep_src = checkpoint_epoch(src_dir)

    if ep_new is None and ep_src is None:
        return None, new_dir, "none"
    if ep_src is not None and (ep_new is None or ep_src > ep_new):
        return ep_src, src_dir, "source"
    return ep_new, new_dir, "dest"


def build_p_order(p_values):
    """Red-robin ordering of P values: alternately take from the low and high
    end of the sorted list, so the P-spread is covered as early as possible."""
    pending = deque(p_values)
    order = []
    take_left = True
    while pending:
        order.append(pending.popleft() if take_left else pending.pop())
        take_left = not take_left
    return order


def launch(p_val, seed, running):
    lr = lr_for(p_val)
    src_lr = source_lr_for(p_val)
    cmd = make_cmd(p_val, seed)
    dest_dir = classic_run_dir(p_val, seed, lr=lr)
    ep, from_dir, origin = effective_start_epoch(p_val, seed)
    log_dir = log_base_dir_for(lr)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"d{D}_P{p_val}_N{N}_chi{CHI}_T{TEMPERATURE}_seed{seed}_eps{EPS}.log",
    )

    # Append so cancel/restart preserves prior log history.
    with open(log_file, "a", encoding="utf-8") as log_f:
        log_f.write("\n")
        log_f.write("=" * 60 + "\n")
        log_f.write(
            f"Launcher relaunch {datetime.now().isoformat(timespec='seconds')}\n"
        )
        log_f.write(f"  dest_run_dir={dest_dir} (train lr={lr})\n")
        log_f.write(f"  source_lr={src_lr}\n")
        if ep is None:
            log_f.write("  checkpoint: none (fresh start)\n")
        else:
            log_f.write(
                f"  checkpoint: epoch {ep} from {origin} ({from_dir.name}) -> target {EPOCHS}\n"
            )
        log_f.write(f"  cmd: {' '.join(cmd)}\n")
        log_f.write("=" * 60 + "\n")
        log_f.flush()
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    running.append(
        {
            "proc": proc,
            "P": p_val,
            "chi": CHI,
            "N": N,
            "seed": seed,
            "lr": lr,
            "log": log_file,
        }
    )
    if ep is None:
        status = "fresh"
    elif origin == "source":
        status = f"resume@{ep} from lr={src_lr}"
    else:
        status = f"resume@{ep}"
    print(
        f"Launched P={p_val} lr={lr} (d={D}, chi={CHI}, N={N}), seed={seed} [{status}] "
        f"-> {dest_dir.name} | log {log_file}"
    )
    time.sleep(0.5)


def try_launch_one(new_p_queue, seed_queue, running, skipped):
    """Fill one free slot. A brand-new P always wins over advancing an
    already-started P to its next seed. Skips jobs already at/ past EPOCHS."""
    if len(running) >= MAX_PARALLEL_JOBS:
        return False

    def maybe_skip(p_val, seed):
        ep, from_dir, origin = effective_start_epoch(p_val, seed)
        if ep is not None and ep >= EPOCHS:
            skipped.append({"P": p_val, "seed": seed, "epoch": ep})
            print(
                f"Skipping P={p_val} seed={seed}: already at epoch {ep} >= {EPOCHS} "
                f"({origin}: {from_dir.name})"
            )
            return True
        return False

    if new_p_queue:
        p_val = new_p_queue.popleft()
        if maybe_skip(p_val, SEEDS[0]):
            if len(SEEDS) > 1:
                seed_queue.append((p_val, 1))
            return True
        launch(p_val, SEEDS[0], running)
        if len(SEEDS) > 1:
            seed_queue.append((p_val, 1))
        return True

    if seed_queue:
        p_val, next_idx = seed_queue.popleft()
        if maybe_skip(p_val, SEEDS[next_idx]):
            if next_idx + 1 < len(SEEDS):
                seed_queue.append((p_val, next_idx + 1))
            return True
        launch(p_val, SEEDS[next_idx], running)
        if next_idx + 1 < len(SEEDS):
            seed_queue.append((p_val, next_idx + 1))
        return True

    return False


def main():
    if not os.path.exists(TRAIN_SCRIPT):
        raise FileNotFoundError(f"Could not find training script: {TRAIN_SCRIPT}")

    missing = [p for p in P_LR if p not in P_LR_SOURCE]
    if missing:
        raise KeyError(f"P_LR_SOURCE missing entries for P={missing}")

    p_values = list(P_VALUES)

    print("=" * 42)
    print("Training FCN2 ERF Network - Sample Complexity Sweep (Cubic Task)")
    print("=" * 42)
    print("Hyperparameters:")
    print(f"  D={D}, chi={CHI}, N={N}")
    print(f"  P values ({len(p_values)} points): {p_values}")
    print(f"  P → train lr (new):     {dict(sorted(P_LR.items()))}")
    print(f"  P → checkpoint lr (old): {dict(sorted(P_LR_SOURCE.items()))}")
    print(f"  naming: {'classic (no _s0_/_sigmaW0_ in run dir)' if USE_CLASSIC_NAMING else 's0-tagged'}")
    print(f"  T={TEMPERATURE}, ens={ENSEMBLE_SIZE}")
    print(f"  epochs={EPOCHS}, eps={EPS}, device={DEVICE}")
    print(f"  seeds: {SEEDS} (new P always takes priority over next seed of an existing P)")
    print(f"  log dirs: {[log_base_dir_for(lr) for lr in sorted(set(P_LR.values()))]}")
    print()
    print("Resume plan:")
    for p_val in p_values:
        lr = lr_for(p_val)
        src_lr = source_lr_for(p_val)
        for seed in SEEDS:
            ep, from_dir, origin = effective_start_epoch(p_val, seed)
            dest = classic_run_dir(p_val, seed, lr=lr)
            if ep is None:
                print(f"  P={p_val} seed={seed}: fresh -> {dest.name}")
            elif origin == "source":
                print(
                    f"  P={p_val} seed={seed}: load epoch {ep} from lr={src_lr} "
                    f"({from_dir.name}) -> train lr={lr} ({dest.name})"
                )
            else:
                print(f"  P={p_val} seed={seed}: continue epoch {ep} in {dest.name}")
    print()

    new_p_queue = deque(build_p_order(p_values))
    seed_queue = deque()  # (p_val, next_seed_idx) for P's that have started but have seeds left
    running = []
    completed = []
    skipped = []

    # Fill as many slots as we can up front.
    while try_launch_one(new_p_queue, seed_queue, running, skipped):
        pass

    while new_p_queue or seed_queue or running:
        for job in running[:]:
            ret = job["proc"].poll()
            if ret is not None:
                running.remove(job)
                completed.append(job)
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(
                    f"Completed P={job['P']} lr={job['lr']} (chi={job['chi']}, N={job['N']}), "
                    f"seed={job['seed']} [{status}]"
                )

        # Backfill freed slots, new-P first, then queued next-seed jobs.
        while try_launch_one(new_p_queue, seed_queue, running, skipped):
            pass

        time.sleep(2)

    print("=" * 42)
    print(f"All jobs completed. {len(completed)} jobs run, {len(skipped)} skipped.")
    print("=" * 42)


if __name__ == "__main__":
    main()
