from torch.utils.data import TensorDataset, DataLoader, random_split
import sys
sys.path.insert(0, '/home/akiva/FCNX-Ensembling/lib')

from google.oauth2.service_account import Credentials
import gspread
import torch._dynamo as dynamo

import signal
import math as mt
import uuid
import atexit
import shutil
import tempfile
import socket
import json
import argparse
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard.writer import SummaryWriter
import re
import sys
import logging
import warnings
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from opt_einsum import contract, contract_path
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ logs
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from FCN3Network import FCN3NetworkEnsembleErf
from GPKit import gpr_dot_product_explicit


##################################################
### &*&   GOOGLE SHEETS LOGGING SETUP.   &*&
##################################################
GOOGLE_SHEETS_ENABLED = True 
GOOGLE_SHEETS_CREDENTIALS_FILE = '/home/akiva/google_api_sheets_service_account.json'
GOOGLE_SHEET_NAME = 'FCN3TrainingProgress' 


_gs_client = None
_gs_worksheet = None
_gs_sheet_url_printed = False


def _precreate_gsheet():
    """Create and share the Google Sheet once, and print its URL.

    This replaces the previous import-time side effect and will only be called
    after CLI args are parsed and Google Sheets logging is confirmed enabled.
    """
    if not GOOGLE_SHEETS_ENABLED:
        return
    try:
        scopes = ['https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_file(
            GOOGLE_SHEETS_CREDENTIALS_FILE, scopes=scopes)
        client = gspread.authorize(creds)
        user_email = 'aclscientist@gmail.com'
        try:
            spreadsheet = client.open(GOOGLE_SHEET_NAME)
            print(f"Sheet '{GOOGLE_SHEET_NAME}' already exists.")
            permissions = spreadsheet.list_permissions()
            already_shared = any(p['type'] == 'user' and p.get(
                'emailAddress', '') == user_email for p in permissions)
            if not already_shared:
                spreadsheet.share(user_email, perm_type='user', role='writer')
                print(f"Shared with {user_email}.")
            print(
                f"Sheet URL: https://docs.google.com/spreadsheets/d/{spreadsheet.id}/edit")
        except gspread.exceptions.SpreadsheetNotFound:
            print(
                f"Sheet '{GOOGLE_SHEET_NAME}' not found. Creating new sheet.")
            spreadsheet = client.create(GOOGLE_SHEET_NAME)
            print(
                f"New sheet created: https://docs.google.com/spreadsheets/d/{spreadsheet.id}/edit")
            spreadsheet.share(user_email, perm_type='user', role='writer')
            print(
                f"Sheet URL: https://docs.google.com/spreadsheets/d/{spreadsheet.id}/edit")
    except Exception as e:
        print(f"Google Sheets setup error: {e}")


def _init_gsheets():
    global _gs_client, _gs_worksheet, _gs_sheet_url_printed
    if not GOOGLE_SHEETS_ENABLED:
        return
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = ['https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_file(
            GOOGLE_SHEETS_CREDENTIALS_FILE, scopes=scopes)
        _gs_client = gspread.authorize(creds)
        _gs_worksheet = _gs_client.open(
            GOOGLE_SHEET_NAME).sheet1  # Use the first worksheet
        # Print the Google Sheets link
        if not _gs_sheet_url_printed:
            spreadsheet_id = _gs_worksheet.spreadsheet.id
            sheet_gid = _gs_worksheet.id
            url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit#gid={sheet_gid}"
            print(f"Google Sheets logging enabled. Sheet URL: {url}")
            _gs_sheet_url_printed = True
    except Exception as e:
        print(f"Google Sheets setup error: {e}")
        _gs_worksheet = None


def log_to_gsheets(epoch, gpr_alignment, eigenvalues, folder_name, input_size, hidden_size, ensemble_size, chi, learning_rate, status='RUNNING', current_loss=None):
    folder_name = os.path.basename(folder_name)
    if not GOOGLE_SHEETS_ENABLED:
        return
    global _gs_worksheet
    if _gs_worksheet is None:
        _init_gsheets()
    from datetime import datetime
    header = [
        "FolderName", "Timestamp", "Epoch", "GPRAlignment", "Eigenvalues",
        "InputSize", "HiddenSize", "EnsembleSize", "Chi", "LearningRate", "CurrentLoss", "Status"
    ]
    row = [
        folder_name, datetime.now().isoformat(), epoch, gpr_alignment, str(eigenvalues),
        input_size, hidden_size, ensemble_size, chi, learning_rate, current_loss, status
    ]
    try:
        if _gs_worksheet is not None:
            first_row = _gs_worksheet.row_values(1)
            if first_row != header:
                update_header = header[:]
                if len(first_row) < len(header):
                    first_row += [''] * (len(header) - len(first_row))
                elif len(first_row) > len(header):
                    first_row = first_row[:len(header)]
                if first_row != header:
                    _gs_worksheet.update([header], 'A1:L1')
            folder_names = _gs_worksheet.col_values(1)
            found = False
            for idx, val in enumerate(folder_names[1:], start=2):
                if val == str(folder_name):
                    current_row = _gs_worksheet.row_values(idx)
                    if len(current_row) < len(header):
                        current_row += [None] * \
                            (len(header) - len(current_row))
                    elif len(current_row) > len(header):
                        current_row = current_row[:len(header)]
                    updated_row = [new_val if new_val != str(
                        None) else old_val for new_val, old_val in zip(row, current_row)]
                    _gs_worksheet.update([updated_row], f'A{idx}:L{idx}')
                    found = True
                    _set_status_cell_color(
                        _gs_worksheet, idx, len(header), status)
                    break
            if not found:
                _gs_worksheet.append_row(row)
                last_row = len(_gs_worksheet.get_all_values())
                _set_status_cell_color(
                    _gs_worksheet, last_row, len(header), status)
    except Exception as e:
        print(f'Google Sheets log error: {e}')

# Update _set_status_cell_color to support blue for COMPLETE


def _set_status_cell_color(worksheet, row_idx, col_idx, status):
    # row_idx and col_idx are 1-based
    try:
        sheet_id = worksheet._properties['sheetId']
        color = {
            'RUNNING': {'red': 0.0, 'green': 1.0, 'blue': 0.0},
            'STOPPED': {'red': 1.0, 'green': 0.0, 'blue': 0.0},
            'COMPLETE': {'red': 0.0, 'green': 0.6, 'blue': 1.0}
        }.get(status, {'red': 1.0, 'green': 1.0, 'blue': 1.0})
        worksheet.spreadsheet.batch_update({
            'requests': [{
                'repeatCell': {
                    'range': {
                        'sheetId': sheet_id,
                        'startRowIndex': row_idx - 1,
                        'endRowIndex': row_idx,
                        'startColumnIndex': col_idx - 1,
                        'endColumnIndex': col_idx
                    },
                    'cell': {
                        'userEnteredFormat': {
                            'backgroundColor': color
                        }
                    },
                    'fields': 'userEnteredFormat.backgroundColor'
                }
            }]
        })
    except Exception as e:
        print(f'Google Sheets cell color error: {e}')

# Add a function to delete a row by folder_name


def delete_gsheets_row_by_foldername(folder_name):
    if not GOOGLE_SHEETS_ENABLED:
        return
    global _gs_worksheet
    if _gs_worksheet is None:
        _init_gsheets()
    try:
        if _gs_worksheet is not None:
            folder_names = _gs_worksheet.col_values(1)
            # skip header
            for idx, val in enumerate(folder_names[1:], start=2):
                if val == str(folder_name):
                    _gs_worksheet.delete_rows(idx)
                    print(
                        f"Deleted Google Sheets row for debug folder: {folder_name}")
                    break
    except Exception as e:
        print(f'Google Sheets row delete error: {e}')

    import signal

    def handle_tb_sigint(sig, frame):
        debug = globals().get('debug', False)
        print("KeyboardInterrupt received: saving model before shutdown...")
        # Close tqdm progress bar if available
        try:
            if 'training_pbar' in globals() and training_pbar is not None:
                training_pbar.close()
        except Exception as e:
            print(f"Could not close tqdm progress bar: {e}")
        # Save model and state
        try:
            if 'model' in globals() and 'save_dir' in globals() and not debug:
                model_filename = f"model.pth"
                atomic_save_model(model, os.path.join(
                    save_dir, model_filename))
                with open(os.path.join(save_dir, f"losses.txt"), "a") as f:
                    if 'loss' in locals():
                        f.write(
                            f"{{'interrupt': True, 'epoch': epoch, 'loss': {loss.item()}}}\n")
                atomic_save_json({'epoch': epoch}, state_path)
                print(f"Model and state saved at interrupt (epoch {epoch})")
        except Exception as e:
            print(f"Error saving model/state on interrupt: {e}")
        # Set status to STOPPED in Google Sheets
        try:
            log_to_gsheets(
                epoch=epoch,
                gpr_alignment=None,
                eigenvalues=None,
                folder_name=modeldesc,
                input_size=input_size,
                hidden_size=hidden_size,
                ensemble_size=ens,
                chi=chi,
                learning_rate=current_base_learning_rate,
                status='STOPPED'
            )
        except Exception as e:
            print(f"Error updating Google Sheets status on interrupt: {e}")
        # If in debug mode, delete the Google Sheets row
        if debug:
            try:
                delete_gsheets_row_by_foldername(modeldesc)
            except Exception as e:
                print(f"Error deleting Google Sheets row for debug: {e}")
        exit(0)

# Custom loss function (slightly faster than MSE)
def custom_mse_loss(outputs, targets):
    diff = outputs - targets
    return 0.5 * torch.sum(diff * diff)


def atomic_save_model(model, save_path):
    import torch
    import os
    tmp_path = save_path + ".tmp"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, save_path)


def atomic_save_json(data, save_path):
    import json
    import os
    tmp_path = save_path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f)
    os.replace(tmp_path, save_path)

def gen_eigenvalues_for_logging(model, X_inf, Y1_inf, Y3_inf):
    with torch.no_grad():
    
        lH1 = model.H_eig(X_inf, Y1_inf)
        lH3 = model.H_eig(X_inf, Y3_inf)
        lK1 = model.K_eig(X_inf, Y1_inf)
        lK3 = model.K_eig(X_inf, Y3_inf)
        return lH1, lH3, lK1, lK3

def sift_eigenvalues_for_logging(lH1, lH3, lK1, lK3, kappa, num_samples):

        # Ensure we only index up to available eigenvalues
        d_plot = min(int(lH1.shape[0]), int(lK1.shape[0]), 10)

        scalarsH1 = {}
        scalarsH3 = {}
        scalarsK1 = {}
        scalarsK3 = {}
        muK1 = {}
        muK3 = {}

        # noise_term = kappa / P
        noise_term = float(kappa) / float(num_samples)

        for i in range(d_plot):
            scalarsH1[str(i)] = lH1[i].item() if hasattr(
                lH1[i], 'item') else float(lH1[i])
            scalarsH3[str(i)] = lH3[i].item() if hasattr(
                lH3[i], 'item') else float(lH3[i])

            scalarsK1[str(i)] = lK1[i].item() if hasattr(
                lK1[i], 'item') else float(lK1[i])
            scalarsK3[str(i)] = lK3[i].item() if hasattr(
                lK3[i], 'item') else float(lK3[i])

            # learnability mu = lKi / (lKi + kappa/P)
            li_k1 = lK1[i].item() if hasattr(
                lK1[i], 'item') else float(lK1[i])
            li_k3 = lK3[i].item() if hasattr(
                lK3[i], 'item') else float(lK3[i])
            muK1[str(i)] = li_k1 / (li_k1 + noise_term + 1e-30)
            muK3[str(i)] = li_k3 / (li_k3 + noise_term + 1e-30)

        return scalarsH1, scalarsH3, scalarsK1, scalarsK3, muK1, muK3
    

def nicely_print_eigen_summary(lH1, lH3, lK1, lK3, scalarsH1, scalarsH3, scalarsK1, scalarsK3, muK1, muK3):
    try:

        # Determine how many entries we actually have
        d_print = min(
            len(scalarsH1),
            len(scalarsH3),
            len(scalarsK1),
            len(scalarsK3),
            len(muK1),
            len(muK3)
        )

        print("\nEigenvalues and learnabilities (mu) summary:")
        header = f"{'idx':>3} {'He1':>14} {'He3':>14} {'K_He1':>14} {'K_He3':>14} {'mu_He1':>10} {'mu_He3':>10}"
        print(header)
        print("-" * len(header))

        for i in range(d_print):
            h1 = scalarsH1.get(str(i), float('nan'))
            h3 = scalarsH3.get(str(i), float('nan'))
            k1 = scalarsK1.get(str(i), float('nan'))
            k3 = scalarsK3.get(str(i), float('nan'))
            mu1 = muK1.get(str(i), float('nan'))
            mu3 = muK3.get(str(i), float('nan'))

            # Ensure numeric formatting even if values are tensors
            def _num(x):
                try:
                    return x.item() if hasattr(x, 'item') else float(x)
                except Exception:
                    return float('nan')

            print(
                f"{i:3d} {_num(h1):14.6g} {_num(h3):14.6g} {_num(k1):14.6g} {_num(k3):14.6g} {_num(mu1):10.6g} {_num(mu3):10.6g}")

        # Print a small aggregate summary for He1 (as before), plus basic stats for mus
        if int(lH1.shape[0]) > 1:
            lH10 = lH1[0].item() if hasattr(
                lH1[0], 'item') else float(lH1[0])
            lH1_rest = lH1[1:]
            lH1_rest_mean = lH1_rest.mean().item() if hasattr(
                lH1_rest.mean(), 'item') else float(lH1_rest.mean())
            lH1_rest_std = lH1_rest.std().item() if hasattr(
                lH1_rest.std(), 'item') else float(lH1_rest.std())
            print(
                f"\nInitial He1 eigenvalues: lH1[0]={lH10:.4g}, mean(lH1[1:])={lH1_rest_mean:.4g}, std(lH1[1:])={lH1_rest_std:.4g}")
        else:
            single = lH1[0].item() if hasattr(
                lH1[0], 'item') else float(lH1[0])
            print(
                f"\nInitial He1 eigenvalue: lH1[0]={single:.6g}")

        # Summary stats for mu (learnability)
        try:
            mu_vals = [_num(muK1[str(i)])
                    for i in range(d_print)]
            mu_mean = float(np.mean(mu_vals)) if len(
                mu_vals) > 0 else float('nan')
            mu_std = float(np.std(mu_vals)) if len(
                mu_vals) > 0 else float('nan')
            print(
                f"Learnability (mu) for K_He1: mean={mu_mean:.6g}, std={mu_std:.6g}")
            mu3_vals = [_num(muK3[str(i)])
                        for i in range(d_print)]
            mu3_mean = float(np.mean(mu3_vals)) if len(
                mu3_vals) > 0 else float('nan')
            mu3_std = float(np.std(mu3_vals)) if len(
                mu3_vals) > 0 else float('nan')
            print(
                f"Learnability (mu) for K_He3: mean={mu3_mean:.6g}, std={mu3_std:.6g}\n")
        except Exception:
            pass

    except Exception as e:
        print(f"Could not print eigenvalue summary: {e}")

def log_eigenvalues_to_tensorboard(writer, epoch, scalarsH1, scalarsH3, scalarsK1, scalarsK3, muK1, muK3):
    writer.add_scalars('Eigenvalues/He1', scalarsH1, epoch)
    writer.add_scalars('Eigenvalues/He3', scalarsH3, epoch)
    writer.add_scalars('Eigenvalues/K_He1', scalarsK1, epoch)
    writer.add_scalars('Eigenvalues/K_He3', scalarsK3, epoch)
    writer.add_scalars('Learnability/He1_mu', muK1, epoch)
    writer.add_scalars('Learnability/He3_mu', muK3, epoch)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train or resume FCN3 ensemble model.')
    parser.add_argument('--modeldesc', type=str, default=None,
                        help='Model description directory to resume from')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode (no logging or saving)')
    parser.add_argument('--nolog', action='store_true',
                        help='Disable Google Sheets logging for this run')
    parser.add_argument('--tags', type=str, default='',
                        help='Comma/space-separated tags; include "nolog" to disable Sheets logging; use eiglog[:|=]N to set eigen log interval (default 150 if no N)')
    parser.add_argument('--eiglog', type=int, default=None,
                        help='Log eigenvalues every N epochs; can also use tag eiglog[:|=]N, default 150 when tag has no value')
    parser.add_argument('--epochs', type=int, default=2_000_000,
                        help='Number of training epochs (default: 2,000,000)')
    parser.add_argument('--chi', type=int, default=1,
                        help='Value for chi (default: 200)')
    parser.add_argument('--P', type=int, default=20,
                        help='Number of samples P (default: 400)')
    parser.add_argument('--N', type=int, default=400,
                        help='Hidden size N (default: 200)')
    parser.add_argument('--d', type=int, default=20,
                        help='Input size d (default: 50)')
    parser.add_argument('--lrA', type=float, default=1e-9,
                        help='Learning rate lrA (default: 1e-3/400)')
    parser.add_argument('--ens', type=int, default=100,
                        help='Number of ensembles to train')
    parser.add_argument('--kappa', type=float, default=1.0,
                        help='Data noise parameter kappa')
    parser.add_argument('-static', '--static', action='store_true',
                        help='Keep the learning rate static at lrA')
    parser.add_argument('--alphaT', type=float, default=0.7,
                        help='Fraction (<=1) of epochs before lowering learning rate')
    parser.add_argument('--eps', type=float, default=0.03,
                        help='Small coefficient for He3 component in target')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (default: P)')
    args = parser.parse_args()

    global  debug
    debug = args.debug


    #################################################################
    #  &*& GOOGLE SHEETS LOGGING AND ARGUMENT PARSING.   &*&
    #################################################################
    # Determine if Google Sheets logging should be disabled via flags/tags
    provided_tags = {t for t in [s.strip().lower()
                                 for s in args.tags.replace(',', ' ').split()] if t}
    nolog_requested = args.nolog or ('nolog' in provided_tags) or (
        args.modeldesc is not None and 'nolog' in str(args.modeldesc).lower())
    if nolog_requested or debug:
        # Disable Sheets logging globally if requested or in debug mode
        GOOGLE_SHEETS_ENABLED = False

    ################################################################
    # &*&   Main hyperparameters from arguments.        &*&
    ################################################################

    epochs = args.epochs
    modeldesc = ''
    save_dir = getattr(args, 'modeldesc', '')
    chi = args.chi
    ensembles = getattr(args, 'ens', 100)
    num_samples = args.P
    hidden_size = args.N
    input_size = args.d
    lrA = args.lrA / num_samples
    output_size = 1
    k = getattr(args, 'kappa', 1.0)
    kappa = k
    batch_size = args.batch_size if args.batch_size is not None else num_samples
    t0 = 2 * k
    t = t0 / chi  # Temperature for Langevin (used in noise)
    eps = float(getattr(args, 'eps', 0.03))

    # --- Learning Rate Schedule Parameters ---
    alphaT = max(0.0, min(1.0, getattr(args, 'alphaT', 0.7)))
    T = int(epochs * alphaT)
    lrB = (1.0 / 3) * lrA / num_samples

    # Log intervals
    log_interval = 15_000
    save_interval = 100_000
    eigenvalue_log_interval = 10_000
    gsheets_log_interval = 10_000
    tqdm_log_interval = 100

    # Set seeds as constants
    DATA_SEED = 613
    MODEL_SEED = 26
    LANGEVIN_SEED = 480

    # Set the base learning rate globally
    global current_base_learning_rate
    current_base_learning_rate = lrA

    # Set the default dtype to float64
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Resume decides whether to load from existing checkpoint
    resume = False

    # Conditionally set up logging and saving directories
    # based on whether in debug mode
    if not debug:
        if args.modeldesc is not None:
            modeldesc = os.path.normpath(args.modeldesc)
            save_dir = os.path.join("/home/akiva/fcn3s", modeldesc)
            runs_dir = os.path.join(save_dir, "runs")
            os.makedirs(runs_dir, exist_ok=True)
            resume = True
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            modeldesc = f"erf_cubic_eps_{eps}_P_{num_samples}_D_{input_size}_N_{hidden_size}_epochs_{epochs}_lrA_{lrA:.2e}_time_{timestamp}"
            save_dir = os.path.join("/home/akiva/exp/fcn3erf", modeldesc)
            os.makedirs(save_dir, exist_ok=True)
            runs_dir = os.path.join(save_dir, "runs")
            os.makedirs(runs_dir, exist_ok=True)
            resume = False

        # Initialize TensorBoard writer
        writer_cm = SummaryWriter(log_dir=runs_dir)
        state_path = os.path.join(save_dir, 'state.json')
        model_path = os.path.join(save_dir, 'model.pth')
        # Prepare Google Sheet only if enabled and not debug
        if GOOGLE_SHEETS_ENABLED:
            _precreate_gsheet()
    else:
        if args.modeldesc is not None:
            resume = True
        # Create a random temp directory for debug TensorBoard logs
        debug_tmp_dir = os.path.join(
            '/home/akiva/exp/fcn3erf/debugtmp', str(uuid.uuid4()))
        print(f"Debug mode: logging to {debug_tmp_dir}")
        save_dir = debug_tmp_dir
        modeldesc = debug_tmp_dir  # Set foldername/modeldesc to tmp dir in debug mode
        os.makedirs(debug_tmp_dir, exist_ok=True)
        writer_cm = SummaryWriter(log_dir=debug_tmp_dir)

        def cleanup_debug_tmp():
            try:
                shutil.rmtree(debug_tmp_dir)
                print(
                    f"Deleted debug TensorBoard log directory: {debug_tmp_dir}")
            except Exception as e:
                print(f"Could not delete debug TensorBoard log directory: {e}")
        atexit.register(cleanup_debug_tmp)
        # Also handle KeyboardInterrupt

        def handle_sigint(sig, frame):
            cleanup_debug_tmp()
            exit(0)
        signal.signal(signal.SIGINT, handle_sigint)

    state_path = os.path.join(save_dir, 'state.json')
    model_path = os.path.join(save_dir, 'model.pth')

    print(f"Torch device: {device}")
    if torch.cuda.is_available():
        print(
            f"CUDA device name: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'N/A'}")

    ################################################
    # &*&   Data Generation.              &*&
    ################################################
    torch.manual_seed(DATA_SEED)
    X = torch.randn((num_samples, input_size),
                    dtype=torch.float64, device=device)

    # Target: y(x) = He1(w·x) + eps * He3(w·x), probabilists' Hermite with w = e1
    z = X[:, 0]  # w = e1 along w0 direction
    He1 = z
    He3 = z**3 - 3.0 * z
    Y = (He1 + eps * He3).unsqueeze(-1)

    torch.manual_seed(DATA_SEED)
    X_inf = torch.randn((2000, input_size), dtype=torch.float64, device=device)
    Y1_inf = X_inf  # He1 along each coordinate
    Y3_inf = eps * (X_inf**3 - 3.0 * X_inf)  # He3 along each coordinate

    # ------------------------------------------------------------
            # 1. Put the full data into a TensorDataset (once, before the loop)
    # ------------------------------------------------------------
    full_dataset = TensorDataset(X, Y)          # X: (N, d), Y: (N,)


    #############################################################
    # &*&   Model Initialization.          &*&.
    #############################################################
    torch.manual_seed(MODEL_SEED)
    ens = getattr(args, 'ens', 100)

    model = FCN3NetworkEnsembleErf(input_size, hidden_size, hidden_size,
                                   num_samples,
                                   ens=ens,
                                   weight_initialization_variance=(
                                       1/input_size, 1.0/hidden_size, 1.0/(hidden_size * chi)),
                                   device=device)

    model.to(device)


    ##############################################################
    # &*&   Resume model and state if requested.    &*&
    ###############################################################

    ### Resume model logic: 
    # 1. Check if model checkpoint exists
    # 2. Load checkpoint
    # 3. Check if keys need adaptation for optimized model and adapt keys if necessary
    # 4. Load state dict into model
    # 5. Load training state (epoch) if state.json exists

    # Resume logic
    epoch = 0
    loaded_model = False
    if resume:
        if os.path.exists(model_path):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                print(model_path)

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



    losses = []


    ###############################################################
    ## &*& Initializing Langevin dynamics parameters. &*&
    ###############################################################
    weight_decay = torch.tensor(
        [input_size, hidden_size, hidden_size*chi], dtype=torch.float64, device=device) * t

    # Pre-allocate noise buffer for Langevin dynamics
    noise_buffer = torch.empty(1, device=device, dtype=torch.float64)
    # Dedicated RNG for Langevin noise to ensure deterministic, epoch-indexed randomness
    langevin_gen = torch.Generator(device=device)


    torch.manual_seed(LANGEVIN_SEED)

    try:
        ls= gen_eigenvalues_for_logging(
            model, X_inf, Y1_inf, Y3_inf)
        scalarsH1, scalarsH3, scalarsK1, scalarsK3, muK1, muK3 = sift_eigenvalues_for_logging(
            *ls, kappa, num_samples)
        nicely_print_eigen_summary(
            *ls, scalarsH1, scalarsH3, scalarsK1, scalarsK3, muK1, muK3)

    except Exception as e:
        print('Failed to compute initial eigenvalues:',  e)
        pass

    with tqdm(total=epochs, desc="Training", unit="epoch", initial=epoch) as pbar:
        global training_pbar
        training_pbar = pbar

        while epoch < epochs:

            model.zero_grad()
            effective_learning_rate_for_update = lrA
            current_base_learning_rate = effective_learning_rate_for_update
            if (epoch > T) and (getattr(args, 'static', True)):
                effective_learning_rate_for_update = lrA / 3

            noise_scale = (2 * effective_learning_rate_for_update * t) ** 0.5
            langevin_gen.manual_seed(LANGEVIN_SEED + epoch)

            try:
                outputs = model(X)
                loss = custom_mse_loss(outputs, Y.unsqueeze(-1))
                losses.append(loss.item())
                avg_loss = loss.item() / (ens * num_samples)
                loss.backward()

                with torch.no_grad():
                    for i, param in enumerate(model.parameters()):
                        if param.grad is not None:
                            noise_buffer.resize_(param.shape).normal_(
                                0, noise_scale, generator=langevin_gen)
                            param.add_(noise_buffer)
                            param.add_(
                                param.data, alpha=-(weight_decay[i]).item() * effective_learning_rate_for_update)
                            param.add_(param.grad, alpha=-effective_learning_rate_for_update)

                if not torch.isfinite(loss):
                    print(f"Warning: Invalid loss at epoch {epoch}: {loss.item()}")

                pbar.update(1)
                # Update progress bar with time info
                pbar.set_postfix({
                    "MSE": f"{avg_loss:.6f}",
                    "Lr": f"{current_base_learning_rate:.2e}",
                })

                epoch += 1

            except Exception as e:
                print(f"Error in training loop at epoch {epoch}: {e}")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = exc_tb.tb_frame.f_code.co_filename if exc_tb is not None else 'Unknown'
                line_number = exc_tb.tb_lineno if exc_tb is not None else 'Unknown'
                exc_type_name = exc_type.__name__ if exc_type is not None else 'Unknown'
                print(f"An exception occurred:")
                print(f" Type: {exc_type_name}")
                print(f" Message: {e}")
                print(f" File: {fname}")
                print(f" Line: {line_number}")

                pbar.update(1)
                epoch += 1
                continue

            if (epoch) % log_interval == 0 or epoch == 0:
                with open(state_path, 'w') as f:
                    json.dump({'epoch': epoch}, f)
                    writer_cm.add_scalar('Loss', avg_loss, epoch)
                    writer_cm.add_scalar(
                        'Learning Rate', current_base_learning_rate, epoch)

            eigenvalues = {}
            if ((epoch) % eigenvalue_log_interval == 0 or epoch == 0) and not args.debug:
                try:
                    lH1, lH3, lK1, lK3 = gen_eigenvalues_for_logging(
                            model, X_inf, Y1_inf, Y3_inf)
                    eigenvalues_data = sift_eigenvalues_for_logging(
                            lH1, lH3, lK1, lK3, kappa, num_samples)
                    scalarsH1, scalarsH3, scalarsK1, scalarsK3, muK1, muK3 = eigenvalues_data
                    log_eigenvalues_to_tensorboard(writer_cm, epoch, *eigenvalues_data)

                    eigenvalues = {
                        'He1': scalarsH1,
                        'He3': scalarsH3,
                        'K_He1': scalarsK1,
                        'K_He3': scalarsK3,
                        'mu_He1': muK1,
                        'mu_He3': muK3
                    }

                except Exception as e:
                    print(f"Error computing/logging eigenvalues at epoch {epoch}: {e}")
                    eigenvalues = None
            

            if (epoch) % gsheets_log_interval == 0 or epoch == 0:   
                try:
                    log_to_gsheets(
                        epoch=epoch,
                        gpr_alignment=cos_sim.item() if 'cos_sim' in locals() else None,
                        eigenvalues=eigenvalues,
                        folder_name=modeldesc,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        ensemble_size=ens,
                        chi=chi,
                        learning_rate=current_base_learning_rate,
                        status='RUNNING',
                        current_loss=avg_loss
                    )
                except Exception as gsheets_exc:
                    print(
                        f"[Warning] Google Sheets logging failed at detailed log interval: {gsheets_exc}")

            if (epoch) % save_interval == 0:
                try: 
                    model_filename = f"model.pth"
                    torch.save(model.state_dict(), os.path.join(
                        save_dir, model_filename))

                    with open(os.path.join(save_dir, f"losses.txt"), "a") as f:
                        f.write(f"{epoch},{loss.item()}\n")
                except Exception as save_exc:
                    print(f"[Warning] Error saving model or losses at epoch {epoch}: {save_exc}")

    # Close TensorBoard writer
    if writer_cm is not None:
        writer_cm.close()
    elif debug and writer_cm is not None:
        writer_cm.close()
    if debug:
        try:
            delete_gsheets_row_by_foldername(modeldesc)
        except Exception as e:
            print(f"Error deleting Google Sheets row for debug: {e}")
    else:
        # Set status to COMPLETE in Google Sheets at end of training
        try:
            log_to_gsheets(
                epoch=epoch,
                gpr_alignment=None,
                eigenvalues=None,
                folder_name=modeldesc,
                input_size=input_size,
                hidden_size=hidden_size,
                ensemble_size=ens,
                chi=chi,
                learning_rate=current_base_learning_rate,
                status='COMPLETE',
                current_loss=avg_loss
            )
        except Exception as e:
            print(
                f"Error updating Google Sheets status at end of training: {e}")
