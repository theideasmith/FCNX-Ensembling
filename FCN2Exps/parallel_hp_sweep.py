import subprocess
import os
from datetime import datetime
import itertools
from multiprocessing import Pool

# Define the hyperparameters lists
Ps = [30]
Ns = [40, 130, 200, 800, 1000, 2000, 4000]
Ds = [3]

ens = 30
print("Ps:")
print(Ps)
print("Ns:")
print(Ns)
print("Ds:")
print(Ds)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVEPATH = f"/home/akiva/gpnettrain/fcn2_FL_{timestamp}"
# Path to the refactored net.py script
refactored_script = 'net.py'
print(f'Save to: {SAVEPATH}')
print("Starting hyperparameter sweep...")
print("-" * 50)

def run_training_command(params):
    """Function to execute a single training command."""
    p, n, d, chi, enum, save_path, script_path = params
    nepochs = 500_000 if chi == 1 else 10_000_000

    print(f"Running training for P={p}, N={n}, D={d}, ens={enum}")
    print(f'{p}, {n}, {d}, {chi}')
    try:
        command = [
            'python', script_path,
            '--P', str(p),
            '--N', str(n),
            '--D', str(d),
            '--chi', str(chi),
            '--epochs', str(nepochs),
            '--to', save_path,
            '--ens', str(enum),
            '--off_data',
        ]

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
    for enum in range(ens):
        for p, n, d in itertools.product(Ps, Ns, Ds):
            chi = 1 # as per your original code
            all_commands_params.append((p, n, d, chi, enum, SAVEPATH, refactored_script))

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
