import subprocess
import os
from datetime import datetime
import itertools
from multiprocessing import Pool

# Define the base values and sweep range
P0 = 50  # base P
kappa0 = 0.5  # base kappa
I_list = [1, 4, 8, 16, 40]  # sweep values for i
N0 = 600
d= 30

lrMF = 0.25 * 1e-3
ens = 5
print("I_list:")
print(I_list)
print("N0:")
print(N0)
print("d:")
print(d)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVEPATH = f"/home/akiva/gpnettrain/fcn3_MF_PKappa_proportional_scaling_{timestamp}"
# Path to the refactored net.py script
refactored_script = 'net_einsum_parallel.py'
print(f'Save to: {SAVEPATH}')``
print("Starting hyperparameter sweep (EK, P/kappa constant)...")
print("-" * 50)

def run_training_command(params):
    """Function to execute a single training command."""
    p, n, d, chi, enum, kappa, save_path, script_path = params
    nepochs =  100_000_000
    lr0 = lrMF
    print(f"Running training for p: {p}, n: {n}, d: {d}, chi: {chi}, ens: {enum}, kappa: {kappa}, epochs: {nepochs}, save_path: {save_path}")
    command = [
        'python', script_path,
        '--P', str(p),
        '--N', str(n),
        '--D', str(d),
        '--chi', str(chi),
        '--epochs', str(nepochs),
        '--to', save_path,
        '--ens', str(enum),
        '--lr0', str(lr0),
        '--kappa', str(kappa)
    ]
    print(' '.join(command))
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully completed P={p}, N={n}, D={d}, ens={enum}, kappa={kappa}")
        print("--- Output from net.py ---")
        print(result.stdout)
        if result.stderr:
            print("--- Errors/Warnings from net.py ---")
            print(result.stderr)
        print("-" * 50)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during training for P={p}, N={n}, D={d}, ens={enum}, kappa={kappa}:")
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
        print(f"An unexpected error occurred for P={p}, N={n}, D={d}, ens={enum}, kappa={kappa}: {ex}")
        print("-" * 50)
        return False

if __name__ == '__main__':
    all_commands_params = []
    for i in I_list:
        p = int(P0 * i)
        kappa = kappa0 * i
        N = int(N0 * 1)
        chi = N
        all_commands_params.append((p, N, d, chi, ens, kappa, os.path.join(SAVEPATH, f'd_{d}_N_{N}_P_{p}_chi_{chi}_kappa_{kappa}'), refactored_script))
    num_processes = 10
    print(f"Using {num_processes} parallel processes.")
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_training_command, all_commands_params)
    print("Hyperparameter sweep finished.")
    if all(results):
        print("All training commands completed successfully.")
    else:
        print("Some training commands failed. Check the logs above for details.") 