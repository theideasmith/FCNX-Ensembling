import subprocess
import os
from datetime import datetime
import itertools
from multiprocessing import Pool

# Define the hyperparameters lists
Ps = [50]
Ns =[60, 80, 160, 600, 1200]
Ds = [10]
chis = [1]
lrMF = 1e-3
lrs = [1e-4]
kappa = [1.0]
ens = 2
print("Ps:")
print(Ps)
print("Ns:")
print(Ns)
print("Ds:")
print(Ds)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVEPATH = f"/home/akiva/gpnettrain/fcn3_GP_GPR_lr0_1e-4_{timestamp}"
# Path to the refactored net.py script
refactored_script = 'net_einsum_parallel.py'
print(f'Save to: {SAVEPATH}')
print("Starting hyperparameter sweep...")
print("-" * 50)

def run_training_command(params):
    """Function to execute a single training command."""
    p, n, d, chi, enum, k, lr, save_path, script_path = params
    nepochs =  100_000_000

    lr0 = lr
    print(f"Running training for p: {p}, n: {n}, d: {d}, chi: {chi}, ens: {enum}, epochs: {nepochs}, save_path: {save_path}")
    command = [
        'python', script_path,
        '--P', str(p),
        '--N', str(n),
        '--D', str(d),
        '--chi', str(chi),
        '--epochs', str(nepochs),
        '--to', save_path,
        '--ens', str(enum),
     #   '--rate_decay', str(rate_decay),
      #  '--lr_schedule',
        # '--lr_stepped',
        '--lr0', str(lr0),
        '--kappa', str(k)
    ]
    print(' '.join(command))

    try:

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

    for p, n, d, k, chi, lr in itertools.product(Ps, Ns, Ds, kappa, chis, lrs):
        all_commands_params.append((p, n, d, chi, ens, k, lr, os.path.join(SAVEPATH, f'd_{d}_N_{n}_P_{p}_chi_{chi}_kappa_{k}_lr_{lr}'), refactored_script))

    # Determine the number of processes to use. You can adjust this.
    # A common choice is os.cpu_count() or a fixed number.
    num_processes = 10
    print(f"Using {num_processes} parallel processes.")

    with Pool(processes=num_processes) as pool:
        # Use pool.map to execute run_training_command for each set of parameters in parallel
        # The results list will contain True/False based on command success
        results = pool.map(run_training_command, all_commands_params)

    print("Hyperparameter sweep finished.")
    if all(results):
        print("All training commands completed successfully.")
    else:
        print("Some training commands failed. Check the logs above for details.")
