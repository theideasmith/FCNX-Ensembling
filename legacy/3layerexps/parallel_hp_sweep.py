import subprocess
import os
from datetime import datetime
import itertools
from multiprocessing import Pool
import argparse
import yaml

# Add argument for YAML config file
parser = argparse.ArgumentParser(description='Parallel hyperparameter sweep using YAML config.')
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
args = parser.parse_args()

# Load config from YAML file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Extract parameters from config
Ps = config.get('Ps', [30])
Ns = config.get('Ns', [50, 200, 600, 1000])
Ds = config.get('Ds', [3])
ens = config.get('ens', 2)
nepochs = config.get('nepochs', 100_000_000)
lrGP = config.get('lrGP', 1e-4)
lrMF = config.get('lrMF', 1e-4)
num_processes = config.get('num_processes', 10)
refactored_script = config.get('script', 'net_einsum_parallel.py')
lr_schedule = config.get('lr_schedule', False)
rate_decay = config.get('rate_decay', 434294.0)
kappas = config.get('kappa', [1.0])


# Compose save path
if 'savepath' in config:
    SAVEPATH = config['savepath']
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVEPATH = f"/home/akiva/gpnettrain/fcn3_MF_SLOWLR_{timestamp}"

print("Ps:")
print(Ps)
print("Ns:")
print(Ns)
print("Ds:")
print(Ds)
print(f'Save to: {SAVEPATH}')
print("Starting hyperparameter sweep...")
print("-" * 50)

def run_training_command(params):
    """Function to execute a single training command."""
    p, n, d, chi, enum, kappa, save_path, script_path = params
    print(f"Running training for p: {p}, n: {n}, d: {d}, chi: {chi}, ens: {enum}, epochs: {nepochs}, save_path: {save_path}")
    try:
        command = [
            'python', script_path,
            '--P', str(p),
            '--N', str(n),
            '--D', str(d),
            '--chi', str(chi),
            '--lr0', str(lrGP if chi == 1 else lrMF),
            '--epochs', str(nepochs),
            '--to', save_path,
            '--ens', str(enum),
            '--lr_schedule',
            '--rate_decay', str(rate_decay),
            '--kappa', str(kappa),
        ]
        print(' '.join(command))

        result = subprocess.run(command, check=True, capture_output=True, text=True)

        print(f"Successfully completed P={p}, N={n}, D={d}, ens={enum}")
        print("--- Output from net.py ---")
        print(result.stdout)
        if result.stderr:
            print("--- Errors/Warnings from net.py ---")
            print(result.stderr)
        print("-" * 50)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during training for P={p}, N={n}, D={d}, ens={enum}:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Exit Code: {e.returncode}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        print("-" * 50)
        return False
    except FileNotFoundError:
        print(f"Error: The script '{script_path}' was not found.")
        print("Please ensure 'net.py' is in the same directory as this script.")
        return False
    except Exception as ex:
        print(f"An unexpected error occurred for P={p}, N={n}, D={d}, ens={enum}: {ex}")
        print("-" * 50)
        return False

if __name__ == '__main__':
    all_commands_params = []

    for p, n, d, kappa in zip(Ps, Ns, Ds, kappas):
        chi = 1.0
        if config.get('chi', 1.0) == 'use_N_as_chi':
            chi = n
        all_commands_params.append((p, n, d, chi, ens, kappa, os.path.join(SAVEPATH, f'd_{d}_N_{n}_P_{p}_chi_{chi}_kappa_{kappa}'), refactored_script))
    for i, c in enumerate(all_commands_params):
        print(c)
        if i > 10:
            break
    print(f"Using {num_processes} parallel processes.")

    with Pool(processes=num_processes) as pool:
        results = pool.map(run_training_command, all_commands_params)

    print("Hyperparameter sweep finished.")
    if all(results):
        print("All training commands completed successfully.")
    else:
        print("Some training commands failed. Check the logs above for details.")
