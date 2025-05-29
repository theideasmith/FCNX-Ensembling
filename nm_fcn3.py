"""
Training script for an ensemble of neural networks using a single teacher network.
Includes JsonHandler for data persistence and EnsembleManager for training orchestration.
"""
from json_handler import JsonHandler
import os
import shutil
import argparse
import yaml
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import io
import matplotlib.pyplot as plt
from utils import unix_basename
from ensemble_manager import EnsembleManager
from Covariance import compute_avg_channel_covariance, project_onto_target_functions, transform_eigenvalues
from FCN2Network import FCN2Network
from FCN3 import SimpleNet, FCN3Network, LangevinTrainer, DataManager, Logger
import standard_hyperparams as hp
from datamanager import DataManager, HypersphereData
# Default configuration
DEFAULT_CONFIG = {
    'self_destruct': False,
    'cash_freak': 1000,
    'init_seed': 222,
    'input_dimension': 50,
    'hidden_width1': 200,
    'hidden_width2': 200,
    'num_data_points': 400,
    'batch_size': 400,
    'num_epochs': 500000,
    'num_ensembles': 20,
    'num_datasets': 3
}

def load_config(config_file):
    """Load configuration from a YAML file or return default config."""
    config = DEFAULT_CONFIG.copy()
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
            config.update(yaml_config)
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Error loading YAML config: {e}. Using default configuration.")
    return config

def initialize_globals(config):
    """Set global hyperparameters from config."""
    global SELF_DESTRUCT, CASH_FREAK, INIT_SEED, input_dimension, hidden_width1, hidden_width2
    global chi, kappa, temperature, lambda_1, lambda_2, lambda_3
    global weight_sigma1, weight_sigma2, weight_sigma3, num_data_points
    global batch_size, learning_rate, noise_std_ld, num_epochs, weight_sigma

    SELF_DESTRUCT = config['self_destruct']
    CASH_FREAK = config['cash_freak']
    INIT_SEED = config['init_seed']
    torch.manual_seed(INIT_SEED)

    input_dimension = config['input_dimension']
    hidden_width1 = config['hidden_width1']
    hidden_width2 = config['hidden_width2']
    chi = hidden_width2
    kappa = 1.0 / chi
    temperature = 2 * kappa
    lambda_1 = temperature * input_dimension
    lambda_2 = temperature * hidden_width1
    lambda_3 = temperature * hidden_width2 * chi
    weight_sigma1 = 1.0 / input_dimension
    weight_sigma2 = 1.0 / hidden_width1
    weight_sigma3 = 1.0 / (hidden_width2 * chi)
    num_data_points = config['num_data_points']
    batch_size = config['batch_size']
    learning_rate = 1e-3 / hidden_width2
    noise_std_ld = (2 * learning_rate * temperature) ** 0.5
    num_epochs = config['num_epochs']
    weight_sigma = (weight_sigma1, weight_sigma2, weight_sigma3)

def destruct(deldir):
    """Clean up ensemble directory and raise interrupt if enabled."""
    if SELF_DESTRUCT:
        shutil.rmtree(deldir)
        raise KeyboardInterrupt

def log_matrix_to_tensorboard(writer, tag, matrix, global_step):
    """Log a 2D NumPy matrix to TensorBoard as an image."""
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        print(f"Warning: Invalid matrix for tag '{tag}'. Skipping.")
        return

    min_val, max_val = matrix.min(), matrix.max()
    normalized_matrix = np.zeros_like(matrix, dtype=np.float32)
    if max_val - min_val > 1e-8:
        normalized_matrix = (matrix - min_val) / (max_val - min_val)

    image_tensor = np.expand_dims(normalized_matrix, axis=0)
    writer.add_image(f"{tag}/image", image_tensor, global_step=global_step, dataformats='CHW')

def log_covariance_plot(writer, tag, cov_matrix, step):
    """Log covariance matrix and its diagonal as a plot to TensorBoard."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 20))
    abs_max = np.max(np.abs(cov_matrix))

    im = ax1.imshow(cov_matrix, cmap='viridis', vmin=-abs_max, vmax=abs_max)
    ax1.set_title(f'Covariance of Output Features (Step {step})')
    ax1.set_xlabel('Output Feature Index')
    ax1.set_ylabel('Output Feature Index')
    fig.colorbar(im, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)

    diagonal_values = np.diag(cov_matrix)
    ax2.plot(np.arange(len(diagonal_values)), diagonal_values, marker='o', linestyle='-', color='red')
    ax2.set_title('Main Diagonal of the fc2 Cov Matrix')
    ax2.set_xlabel('Diagonal Element Index')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    ax2.set_xticks(np.arange(len(diagonal_values)))

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    image_np = np.array(Image.open(buf))
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    writer.add_image(f"{tag}/fc2.Weights_Cov", image_tensor, global_step=step)

def epoch_callback_fcn3(trainer, writer, ensemble_manager, model_identifier):
    """Log training metrics and visualizations at specified intervals."""
    if trainer.current_epoch % CASH_FREAK != 0:
        return

    run_id = ensemble_manager.run_identifier
    tag = f'FCN3_Run_{run_id}_Modelnum_{model_identifier}'

    with torch.no_grad():
        writer.add_scalar(f'{tag}/Train_Step', trainer.current_train_loss, trainer.current_time_step)
        H = compute_avg_channel_covariance(trainer.model, trainer.manager.data, layer_name='fc2')
        log_matrix_to_tensorboard(writer, f'{tag}/H', H.cpu().numpy(), trainer.current_epoch)

        X = trainer.manager.data
        d = X.shape[-1]
        w = torch.eye(d).to(hp.DEVICE)
        y = X @ w
        lH = project_onto_target_functions(H, y)
        lK = transform_eigenvalues(lH, 1.0, chi, num_data_points)

        scalarsH = {str(i): float(f"{value_li.cpu().numpy().item():.6g}") for i, value_li in enumerate(lH)}
        scalarsK = {str(i): float(f"{value_li.cpu().numpy().item():.6g}") for i, value_li in enumerate(lK)}
        writer.add_scalars(f'{tag}/eig_lH', scalarsH, trainer.current_epoch)
        writer.add_scalars(f'{tag}/eig_lK', scalarsK, trainer.current_epoch)

        W = trainer.model.fc2.weight.detach().T
        cov_W = torch.einsum('aj,bj->ab', W, W) / hidden_width1
        log_covariance_plot(writer, tag, cov_W.cpu().numpy(), trainer.current_epoch)

    writer.flush()

def activation(x):
    return x

class TrainingCallbacks:
    """Handle training callbacks for interrupt, completion, and epoch events."""
    def __init__(self, ensemble_manager, writer, model_identifier):
        self.ensemble_manager = ensemble_manager
        self.writer = writer
        self.model_identifier = model_identifier

    def interrupt(self, trainer):
        """Handle training interruption."""
        self.ensemble_manager.save_model(
            trainer.model,
            self.model_identifier,
            current_time_step=trainer.current_time_step,
            converged=False,
            epochs_trained=trainer.current_epoch
        )
        self.writer.close()
        self.writer.flush()
        destruct(self.ensemble_manager.ensemble_dir)

    def completion(self, trainer):
        """Handle training completion."""
        self.ensemble_manager.save_model(
            trainer.model,
            self.model_identifier,
            converged=trainer.converged,
            current_time_step=trainer.current_time_step,
            epochs_trained=trainer.current_epoch
        )

    def epoch(self, trainer):
        """Handle epoch logging."""
        epoch_callback_fcn3(trainer, self.writer, self.ensemble_manager, self.model_identifier)

def initialize_ensemble_manager(ensemble_dir, json_handler, desc, config):
    """Initialize EnsembleManager with a unique run identifier."""
    if ensemble_dir:
        basename = unix_basename(ensemble_dir)
        run_identifier = basename.split('ensemble_', 1)[1] if basename.startswith('ensemble_') else basename
        print(f"Using existing ensemble directory: {ensemble_dir}")
    else:
        dt = datetime.now().strftime("%a, %b %d %Y, %I:%M%p")
        run_identifier = f"FCN3_D_{input_dimension}_N_{hidden_width1}_chi_{chi}_T_{temperature}_{dt}"
        print(f"Creating new ensemble directory: ./Menagerie/{run_identifier}/")

    return EnsembleManager(
        run_identifier=run_identifier,
        json_handler=json_handler,
        deletedir=ensemble_dir is None,
        desc=desc,
        config={'num_ensembles': config['num_ensembles'], 'num_datasets': config['num_datasets']}
    )

def initialize_teacher(ensemble_manager):
    """Load or create and save a teacher network."""
    if ensemble_manager.teacher_exists():
        return ensemble_manager.load_cached_teacher()

    teacher = SimpleNet(input_dimension, activation, 1.0).to(hp.DEVICE)
    torch.save(teacher, os.path.join(ensemble_manager.ensemble_dir, 'teacher_network.pth'))
    ensemble_manager.teacher = teacher
    return teacher

def get_model_indices(ensemble_manager, most_recent_model_manifest, num_ensembles):
    """Determine starting dataset and model indices for training."""
    mid = len(ensemble_manager.training_config['manifest']) if most_recent_model_manifest is None else int(most_recent_model_manifest.get('model_identifier'))
    return int((mid - mid % num_ensembles) / num_ensembles), int(mid % num_ensembles), mid

def load_existing_model(most_recent_model_manifest):
    """Load an existing model and its data if available."""
    if not most_recent_model_manifest or not os.path.exists(most_recent_model_manifest['model_path']):
        return None, None, None, 0, 0

    model_data = {
        'model': torch.load(most_recent_model_manifest['model_path'], weights_only=False).to(hp.DEVICE),
        'data': torch.load(most_recent_model_manifest['raw_X_path']).to(hp.DEVICE),
        'target': torch.load(most_recent_model_manifest['raw_Y_path']).to(hp.DEVICE)
    }
    return (
        model_data['model'],
        model_data['data'],
        model_data['target'],
        int(most_recent_model_manifest['epochs_trained']),
        int(most_recent_model_manifest['current_time_step'])
    )

def create_new_model(j, i, ensemble_manager, teacher, raw_X=None, raw_Y=None):
    """Create and configure a new model for training."""
    if raw_X is None or raw_Y is None:
        raw_X = HypersphereData.sample_hypersphere(num_data_points, input_dimension, normalized=False).to(hp.DEVICE)
        raw_Y = teacher(raw_X)
    xpath = ensemble_manager.save_data(raw_X, f"{j}")
    ypath = ensemble_manager.save_targets(raw_Y, f"{j}")

    weight_decay_config = {
        'fc1.weight': lambda_1,
        'fc2.weight': lambda_2,
        'fc3.weight': lambda_3,
        'fc1.bias': 0.0,
        'fc2.bias': 0.0,
        'fc3.bias': 0.0
    }
    hyperparameters = {
        'input_dimension': input_dimension,
        'hidden_width_1': hidden_width1,
        'hidden_width_2': hidden_width2,
        'activation': activation,
        'output_activation': activation,
        'weight_sigma1': weight_sigma1,
        'weight_sigma2': weight_sigma2,
        'weight_sigma3': weight_sigma3
    }
    model = FCN3Network.model_from_hyperparameters(hyperparameters).to(hp.DEVICE)
    model_identifier = f"{j * ensemble_manager.num_ensembles + i}"

    model_architecture_spec = {
        'kind': 'FCN3',
        'input_dim': input_dimension,
        'hidden_width_1': hidden_width1,
        'hidden_width_2': hidden_width2,
        'weight_sigma': (weight_sigma1, weight_sigma2, weight_sigma3),
        'weight_decay': (lambda_1, lambda_2, lambda_3)
    }
    manifest = ensemble_manager.add_model_to_manifest(
        model_identifier,
        model_architecture_spec,
        data_path=xpath,
        targets_path=ypath,
        langevin_noise=noise_std_ld,
        chi=chi,
        temperature=temperature,
        kappa=kappa,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )
    return model, manifest, raw_X, raw_Y, model_identifier

def train_model(trainer, logger, model_identifier, ensemble_manager, writer):
    """Train a model with specified callbacks."""
    callbacks = TrainingCallbacks(ensemble_manager, writer, model_identifier)
    model = trainer.model
    model._reset_with_weight_sigma(weight_sigma)
    trainer.train(
        logger=logger,
        continue_at_epoch=trainer.current_epoch,
        current_time_step=trainer.current_time_step,
        interrupt_callback=callbacks.interrupt,
        completion_callback=callbacks.completion,
        epoch_callback=callbacks.epoch
    )

def train_dataset(j, start_model_num, ensemble_manager, teacher, writer, model, raw_X, raw_Y, most_recent_epoch, most_recent_timestep, num_ensembles):
    """Train models for a single dataset."""
    is_running_new_model = model is None
    if is_running_new_model:
        model, manifest, raw_X, raw_Y, model_identifier = create_new_model(j, start_model_num, ensemble_manager, teacher)
    else:
        manifest = ensemble_manager.most_recent_model()
        model_identifier = f"{j * num_ensembles + start_model_num}"

    for i in range(start_model_num if is_running_new_model else 0, num_ensembles):
        if is_running_new_model and i != start_model_num:
            model, manifest, raw_X, raw_Y, model_identifier = create_new_model(j, i, ensemble_manager, teacher, raw_X, raw_Y)

        trainer = LangevinTrainer(
            model=model,
            batch_size=batch_size,
            learning_rate=learning_rate,
            noise_std=noise_std_ld,
            manager=DataManager(raw_X.detach().to(hp.DEVICE), raw_Y.detach().to(hp.DEVICE), split=0.95),
            weight_decay_config={
                'fc1.weight': lambda_1,
                'fc2.weight': lambda_2,
                'fc3.weight': lambda_3,
                'fc1.bias': 0.0,
                'fc2.bias': 0.0,
                'fc3.bias': 0.0
            },
            num_epochs=manifest.get('num_epochs', num_epochs)
        )

        if not is_running_new_model:
            print(f"Training cached model: {model_identifier} starting at epoch: {most_recent_epoch}")
            trainer.current_epoch = most_recent_epoch
            trainer.current_time_step = most_recent_timestep

        logger = Logger(num_epochs=trainer.train_config['num_epochs'], completed=trainer.current_epoch, description=f"Training YURI {model_identifier}")
        with logger:
            print(f"Training network {i} on dataset {j}")
            train_model(trainer, logger, model_identifier, ensemble_manager, writer)

def main(desc='', ensemble_dir=None, config_file=None):
    """Run the ensemble training and saving process."""
    config = load_config(config_file)
    initialize_globals(config)
    json_handler = JsonHandler()
    ensemble_manager = initialize_ensemble_manager(ensemble_dir, json_handler, desc, config)
    teacher = initialize_teacher(ensemble_manager)

    with SummaryWriter(log_dir=ensemble_manager.tensorboard_dir) as writer:
        try:
            most_recent_model_manifest = ensemble_manager.most_recent_model()
            start_dataset_num, start_model_num, model_identifier = get_model_indices(
                ensemble_manager, most_recent_model_manifest, config['num_ensembles']
            )
            print(f"Running model: {model_identifier} dnum: {start_dataset_num}, mnum: {start_model_num}")

            model, raw_X, raw_Y, most_recent_epoch, most_recent_timestep = load_existing_model(most_recent_model_manifest)

            for j in range(start_dataset_num, config['num_datasets']):
                train_dataset(
                    j, start_model_num, ensemble_manager, teacher, writer,
                    model, raw_X, raw_Y, most_recent_epoch, most_recent_timestep,
                    config['num_ensembles']
                )

        except KeyboardInterrupt:
            print("Training interrupted")
            destruct(ensemble_manager.ensemble_dir)
            writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a file or set an ensemble directory.")
    parser.add_argument('-f', '--file', type=str, help="Specify an ensemble directory to process.")
    parser.add_argument('-c', '--config', type=str, help="Specify a YAML configuration file.")
    args = parser.parse_args()

    ensemble_dir = args.file
    config_file = args.config
    desc = 'FCN3 with correct hyperparameters and weight covariance plotting for GP comparison'

    if ensemble_dir or config_file:
        print(f"Proceeding with ensemble directory: {ensemble_dir}, config file: {config_file}")
        main(ensemble_dir=ensemble_dir, desc=desc, config_file=config_file)
    else:
        print("Running with default settings: no ensemble directory or config file specified.")
        main(desc=desc)
