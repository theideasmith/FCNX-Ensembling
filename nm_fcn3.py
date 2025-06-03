"""
Training script for an ensemble of neural networks using a single teacher network.
Includes JsonHandler for data persistence and EnsembleManager for training orchestration.
"""
import argparse
import io
import os
import shutil
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# Import standard_hyperparams for default device, but minimize other direct uses
import standard_hyperparams as hp
from Covariance import *
from logger import Logger
from FCN3 import DataManager, LangevinTrainer, SimpleNet # Kept for default teacher and trainer
from FCN3Network import FCN3NetworkEnsembleLinear # Kept for default model builder example
from json_handler import JsonHandler
from utils import unix_basename
import torch.autograd as autograd

autograd.set_detect_anomaly(True)


class EnsembleManager:
    """
    Manages the saving and loading of ensemble training results, including models, data, and metadata.
    Also orchestrates the training process, handling configuration, logging, and model management.
    """

    def __init__(self, run_identifier,
                 deletedir=False,
                 menagerie_dir=os.path.abspath('./Menagerie'),
                 json_handler=None,
                 config=None,
                 desc=''):
        """
        Initializes the EnsembleManager.

        Args:
            run_identifier (str): A unique identifier for this training run.
            deletedir (bool): If True, deletes the ensemble directory if it already exists.
            menagerie_dir (str): The base directory for storing ensemble data.
            json_handler (JsonHandler, optional): An instance of JsonHandler.
                                                If None, a default JsonHandler is created.
            config (dict): Configuration parameters for the ensemble, including model-specific settings.
            desc (str): A description for the training run.
        """
        # Load default configuration first
        self.config = self._load_default_config()

        # Update with provided config, prioritizing user-defined values
        if config:
            self.config.update(config)

        self.run_identifier = run_identifier
        self.menagerie_dir = menagerie_dir
        self.ensemble_dir = os.path.join(self.menagerie_dir, f'ensemble_{run_identifier}')


        # Set core parameters from config
        self.SELF_DESTRUCT = self.config.get('self_destruct', False)
        self.CASH_FREAK = self.config.get('cash_freak', 3000)
        self.INIT_SEED = self.config.get('init_seed', 222)
        torch.manual_seed(self.INIT_SEED)

        # General hyperparameters, now drawn from self.config
        self.input_dimension = self.config.get('input_dimension', 50)
        self.hidden_width1 = self.config.get('hidden_width1', 200)
        self.hidden_width2 = self.config.get('hidden_width2', 200) # May not be relevant for all models
        self.num_data_points = self.config.get('num_data_points', 400)
        self.batch_size = self.config.get('batch_size', 400)
        self.num_epochs = self.config.get('num_epochs', 500000)
        self.num_ensembles = self.config.get('num_ensembles', 20) # Renamed for clarity from 'num_ensembles' to 'num_models_per_dataset'
        self.num_datasets = self.config.get('num_datasets', 3) # Renamed from 'num_datasets'

        # Model-specific parameters derived from configuration
        # These now rely on values *passed into* the config, not hardcoded derivations
        self.chi = self.config.get('chi', self.hidden_width2) # Default to hidden_width2 for FCN3
        self.kappa = self.config.get('kappa', 1.0 / self.chi)
        self.temperature = self.config.get('temperature', 2 * self.kappa)
        self.learning_rate = self.config.get('learning_rate', 1e-3 / self.hidden_width2) # Default to FCN3 calc
        self.noise_std_ld = self.config.get('noise_std_ld', (2 * self.learning_rate * self.temperature) ** 0.5)

        # Weight decay and sigma parameters should come directly from config if not general
        self.lambda_1 = self.config.get('lambda_1', self.temperature * self.input_dimension)
        self.lambda_2 = self.config.get('lambda_2', self.temperature * self.hidden_width1)
        self.lambda_3 = self.config.get('lambda_3', self.temperature * self.hidden_width2 * self.chi) # Default to FCN3 calc
        self.weight_sigma1 = self.config.get('weight_sigma1', 1.0 / self.input_dimension)
        self.weight_sigma2 = self.config.get('weight_sigma2', 1.0 / self.hidden_width1)
        self.weight_sigma3 = self.config.get('weight_sigma3', 1.0 / (self.hidden_width2 * self.chi))
        self.weight_sigma = (self.weight_sigma1, self.weight_sigma2, self.weight_sigma3) # For FCN3

        # Model-specific builder and callbacks
        if self.config['model_builder'] is None: self.model_builder = self._default_fcn3_model_builder
        else: self.model_builder = self.config['model_builder']

        if self.config['teacher_builder'] is None: 
            self.teacher_builder = self._default_teacher_builder
        else:
            self.teacher_builder = self.config['teacher_builder']




        if self.config['epoch_callback_func'] is None: 
            self.epoch_callback_func = self._default_epoch_callback
        else:
            self.epoch_callback_func = self.config['epoch_callback_func']

        if os.path.exists(self.ensemble_dir) and deletedir:
            shutil.rmtree(self.ensemble_dir)
        if not os.path.exists(self.ensemble_dir):
            os.makedirs(self.ensemble_dir)
        print(f'Ensemble_Dir: {self.ensemble_dir}')

        self.tensorboard_dir = self.ensemble_dir
        self.json_handler = json_handler if json_handler else JsonHandler(directory=self.ensemble_dir)
        self.training_config_path = os.path.relpath(os.path.join('training_config'))
        print(self.training_config_path)
        self.training_config = {}

        if os.path.exists(self.ensemble_dir) and (not deletedir):
            print("Ensemble_dir exists. Loading configuration.")
            self.training_config = self.json_handler.load_data(os.path.join(self.ensemble_dir, 'training_config'))
            self.num_ensembles = self.training_config['num_ensembles'] # This should be 'num_models_per_dataset' now
            self.num_datasets = self.training_config['num_datsets'] # This should be 'num_datasets' now
            self.teacher = self.load_teacher()
        else:
            self.teacher = None
            self.training_config = {
                'run_identifier': self.run_identifier,
                'num_ensembles': self.num_ensembles,
                'num_datsets': self.num_datasets,
                "description": desc,
                "manifest": []
            }
            self.json_handler.save_data(self.training_config, self.training_config_path)

        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)

        # Store the teacher's full output for covariance calculation
        self.full_teacher_y = None

    def _load_default_config(self):
        """Internal method to load default configuration."""
        x = {
            'self_destruct': False,
            'cash_freak': 1000,
            'init_seed': 222,
            'input_dimension': 50,
            'hidden_width1': 200,
            'hidden_width2': 200,
            'num_data_points': 400,
            'batch_size': 400,
            'num_epochs': 500000,
            'num_ensembles': 20, # Renamed to 'num_models_per_dataset' conceptually
            'num_datasets': 3,
            'teacher_builder': 'fcn3',
            'model_builder': 'fcn3', # Default FCN3 builder for example
            'epoch_callback_func': 'fcn3', # Default epoch callback
            # Add general hyperparameters here if not dynamically calculated
            # 'learning_rate': None, # Will be calculated by FCN3 specific default if None
            # 'chi': None, # Will be calculated by FCN3 specific default if None
            # 'kappa': None, # Will be calculated by FCN3 specific default if None
            # 'temperature': None, # Will be calculated by FCN3 specific default if None
            # 'noise_std_ld': None, # Will be calculated by FCN3 specific default if None
            # 'lambda_1': None,
            # 'lambda_2': None,
            # 'lambda_3': None,
            # 'weight_sigma1': None,
            # 'weight_sigma2': None,
            # 'weight_sigma3': None,
        }

        return x

    def _default_teacher_builder(self, input_dim):

        """Default teacher network builder (e.g., SimpleNet)."""
        return SimpleNet(input_dim).to(hp.DEVICE)

    def _default_fcn3_model_builder(self, model_config):
        """Default FCN3 model builder."""
        return FCN3NetworkEnsembleLinear(
            model_config['input_dimension'],
            model_config['hidden_width1'],
            model_config['hidden_width2'],
            model_config['num_data_points'],
            ensembles=model_config['num_ensembles']
        )

    def _default_epoch_callback(self, trainer, writer, model_identifier, ensemble_manager_instance):
        """
        Default epoch callback for FCN3 model, generalized to accept `ensemble_manager_instance`
        to access its dynamic parameters and logging methods.
        """
        print("Using default epoch callback")
        if trainer.current_epoch % ensemble_manager_instance.CASH_FREAK != 0:
            return

        tag = f'Run_{ensemble_manager_instance.run_identifier}_Modelnum_{model_identifier}'

        with torch.no_grad():
            writer.add_scalar(f'{tag}/Train_Step', trainer.current_train_loss, trainer.current_time_step)
            # Use the configurable layer name
            H = compute_avg_channel_covariance(trainer.model, trainer.manager.data)
            ensemble_manager_instance._log_matrix_to_tensorboard(writer, f'{tag}/H', H.cpu().numpy(), trainer.current_epoch)

            lH = project_onto_target_functions(H, ensemble_manager_instance.full_teacher_y)
            lK = transform_eigenvalues(lH, 1.0, ensemble_manager_instance.chi, ensemble_manager_instance.num_data_points)

            scalarsH = {str(i): float(f"{value_li.cpu().numpy().item():.6g}") for i, value_li in enumerate(lH)}
            scalarsK = {str(i): float(f"{value_li.cpu().numpy().item():.6g}") for i, value_li in enumerate(lK)}
            writer.add_scalars(f'{tag}/eig_lH', scalarsH, trainer.current_epoch)
            writer.add_scalars(f'{tag}/eig_lK', scalarsK, trainer.current_epoch)

             # This part is still specific to FCN3's 'fc2' layer.
             # For a truly modular solution, this would need to be abstracted
             # (e.g., by checking for `fc2` attribute or making the layer dynamic).
             # For now, it remains FCN3-specific in the default callback.
             if hasattr(trainer.model, ensemble_manager_instance.covariance_layer_name):
                 W = trainer.model.W1.detach().T
                 cov_W = torch.einsum('naj,nbj->ab', W, W) / (ensemble_manager_instance.hidden_width1 * ensemble_manager.num_ensembles)
                 ensemble_manager_instance._log_covariance_plot(writer, tag, cov_W.cpu().numpy(), trainer.current_epoch)
             else:
                 print(f"Warning: Model has no layer named '{ensemble_manager_instance.covariance_layer_name}'. Skipping covariance plot.")


        writer.flush()

    def load_config(self, config_file):
        """Load configuration from a YAML file or return current config."""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                self.config.update(yaml_config) # Update the internal config dictionary
                # Re-set instance attributes based on loaded config
                self._update_instance_attributes_from_config()
                print(f"Loaded configuration from {config_file}")
            except Exception as e:
                print(f"Error loading YAML config: {e}. Using current configuration.")
        return self.config

    def _update_instance_attributes_from_config(self):
        """Updates relevant instance attributes from the current self.config."""
        self.SELF_DESTRUCT = self.config.get('self_destruct', False)
        self.CASH_FREAK = self.config.get('cash_freak', 1000)
        self.INIT_SEED = self.config.get('init_seed', 222)
        torch.manual_seed(self.INIT_SEED)

        self.input_dimension = self.config.get('input_dimension', 50)
        self.hidden_width1 = self.config.get('hidden_width1', 200)
        self.hidden_width2 = self.config.get('hidden_width2', 200)
        self.num_data_points = self.config.get('num_data_points', 400)
        self.batch_size = self.config.get('batch_size', 400)
        self.num_epochs = self.config.get('num_epochs', 500000)
        self.num_ensembles = self.config.get('num_ensembles', 20)
        self.num_datasets = self.config.get('num_datasets', 3)

        self.chi = self.config.get('chi', self.hidden_width2)
        self.kappa = self.config.get('kappa', 1.0 / self.chi)
        self.temperature = self.config.get('temperature', 2 * self.kappa)
        self.learning_rate = self.config.get('learning_rate', 1e-3 / self.hidden_width2)
        self.noise_std_ld = self.config.get('noise_std_ld', (2 * self.learning_rate * self.temperature) ** 0.5)

        self.lambda_1 = self.config.get('lambda_1', self.temperature * self.input_dimension)
        self.lambda_2 = self.config.get('lambda_2', self.temperature * self.hidden_width1)
        self.lambda_3 = self.config.get('lambda_3', self.temperature * self.hidden_width2 * self.chi)
        self.weight_sigma1 = self.config.get('weight_sigma1', 1.0 / self.input_dimension)
        self.weight_sigma2 = self.config.get('weight_sigma2', 1.0 / self.hidden_width1)
        self.weight_sigma3 = self.config.get('weight_sigma3', 1.0 / (self.hidden_width2 * self.chi))
        self.weight_sigma = (self.weight_sigma1, self.weight_sigma2, self.weight_sigma3)

        self.model_builder = self.config['model_builder'] # Must be present after load
        self.teacher_builder = self.config.get('teacher_builder', self._default_teacher_builder)
        self.epoch_callback_func = self.config.get('epoch_callback_func', self._default_epoch_callback)


    def load_teacher(self):
        """Loads the teacher network."""
        teacher_path = os.path.join(self.ensemble_dir, 'teacher_network.pth')
        if os.path.exists(teacher_path):
            self.teacher = torch.load(teacher_path, weights_only=False).to(hp.DEVICE)
            return self.teacher
        return None

    def save_teacher(self):
        """Saves the teacher network."""
        if self.teacher is not None:
            torch.save(self.teacher, os.path.join(self.ensemble_dir, 'teacher_network.pth'))

    def teacher_exists(self):
        """Checks if a teacher network exists."""
        return os.path.exists(os.path.join(self.ensemble_dir, 'teacher_network.pth'))

    def add_model_to_manifest(self,
                              model_identifier,
                              model_architecture,
                              data_path='',
                              targets_path='',
                              current_epoch=None,
                              current_time_step=None,
                              converged=False,
                              **kwargs):
        """Adds a model's configuration to the training manifest."""
        model_config = {
            "batch_size": self.batch_size,
            "model_identifier": model_identifier,
            "network_architecture": model_architecture,
            "input_dimension": self.input_dimension,
            "num_epochs": self.num_epochs,
            "num_data_points": self.num_data_points,
            "target_noise": hp.TARGET_NOISE, # Still using hp for this, consider making configurable
            "weight_decay_config": hp.WEIGHT_DECAY_CONFIG, # Still using hp for this, consider making configurable
            "network_dir": os.path.join(self.ensemble_dir, f'network_{model_identifier}'),
            "model_path": os.path.join(self.ensemble_dir, f'network_{model_identifier}', f"network_{model_identifier}.pth"),
            "raw_X_path": data_path,
            "raw_Y_path": targets_path,
            "epochs_trained": 0 if current_epoch is None else current_epoch,
            "current_time_step": 0 if current_time_step is None else current_time_step,
            "converged": converged,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "chi": self.chi,
            "kappa": self.kappa,
        }
        if kwargs:
            model_config.update(kwargs)
        self.training_config['manifest'].append(model_config)
        self.json_handler.save_data(self.training_config, self.training_config_path)
        return model_config

    def save_model(self, model, model_identifier, converged=True, current_time_step=0, epochs_trained=0):
        """Saves a trained model and updates its status in the manifest."""
        network_dir = os.path.join(self.ensemble_dir, f'network_{model_identifier}')
        if not os.path.exists(network_dir):
            os.makedirs(network_dir)

        self.model_update(model_identifier, 'converged', converged)
        self.model_update(model_identifier, 'epochs_trained', epochs_trained)
        self.model_update(model_identifier, 'current_time_step', current_time_step)

        torch.save(model, os.path.join(network_dir, f'network_{model_identifier}.pth'))

    def most_recent_model(self):
        """Obtains the model most recently trained but not completed."""
        for item in reversed(self.training_config['manifest']): # Check from most recent
            if not item.get('converged', True): # Default to True if 'converged' key is missing
                print("Found an incomplete model in the manifest.")
                return item
        print("No recent untrained model exists.")
        return None

    def model_update(self, model_identifier, key, updated_value):
        """Updates a specific key for a model in the manifest."""
        # Find the index of the model by model_identifier
        index = -1
        for i, m in enumerate(self.training_config['manifest']):
            if str(m.get('model_identifier')) == str(model_identifier):
                index = i
                break

        if index != -1:
            try:
                self.training_config['manifest'][index][key] = updated_value
                self.json_handler.save_data(self.training_config, self.training_config_path)
            except FileNotFoundError:
                print(f"Error: The file '{self.training_config_path}' was not found during model update.")
            except json.JSONDecodeError:
                print(f"Error: The file '{self.training_config_path}' does not contain valid JSON.")
        else:
            print(f"Warning: Model with identifier '{model_identifier}' not found in manifest for update.")

    def save_data(self, X, x_identifier):
        """Saves input data X to a file."""
        xpath = os.path.join(self.ensemble_dir, f"raw_X_{x_identifier}.pth")
        torch.save(X, xpath)
        return xpath

    def save_targets(self, Y, y_identifier):
        """Saves target data Y to a file."""
        ypath = os.path.join(self.ensemble_dir, f"raw_Y_{y_identifier}.pth")
        torch.save(Y, ypath)
        return ypath

    def destruct(self):
        """Cleans up ensemble directory and raises interrupt if enabled."""
        if self.SELF_DESTRUCT:
            print(f"Self-destruct enabled: Deleting {self.ensemble_dir}")
            shutil.rmtree(self.ensemble_dir)
            raise KeyboardInterrupt

    def _log_matrix_to_tensorboard(self, writer, tag, matrix, global_step):
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

    def _log_covariance_plot(self, writer, tag, cov_matrix, step):
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

    def _activation(self, x):
        # This can be made configurable if different models use different activations
        return x

    class _TrainingCallbacks:
        """Internal class to handle training callbacks for interrupt, completion, and epoch events."""
        def __init__(self, ensemble_manager_instance, writer, model_identifier):
            self.ensemble_manager = ensemble_manager_instance
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
            self.ensemble_manager.destruct()

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
            """Handle epoch logging using the dynamically set callback."""
            self.ensemble_manager.epoch_callback_func(trainer, self.writer, self.model_identifier, self.ensemble_manager)


    def initialize_teacher(self):
        """Load or create and save a teacher network using the configured teacher_builder."""
        if self.teacher_exists():
            print("Teacher exists")
            return self.load_teacher()

        # Call the teacher_builder function provided in the config
        teacher = self.teacher_builder(self.input_dimension).to(hp.DEVICE)
        print(teacher)
        torch.save(teacher, os.path.join(self.ensemble_dir, 'teacher_network.pth'))
        self.teacher = teacher
        return teacher

    def get_model_indices(self, most_recent_model_manifest):
        """Determine starting dataset and model indices for training."""
        mid = len(self.training_config['manifest']) if most_recent_model_manifest is None else \
            int(most_recent_model_manifest.get('model_identifier'))
        return int((mid - mid % self.num_ensembles) / self.num_ensembles), int(mid % self.num_ensembles), mid

    def load_existing_model(self, most_recent_model_manifest):
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

    def create_new_model(self, j, teacher, raw_X=None, raw_Y=None):
        """Create and configure a new model for training using the configured model_builder."""
        if raw_X is None or raw_Y is None:
            std_dev = 1 / (self.input_dimension**0.5)
            raw_X = torch.randn(self.num_data_points, self.input_dimension) * std_dev
            raw_X = raw_X.to(hp.DEVICE)
            raw_Y = teacher(raw_X).detach() # Teacher generates the targets
            # Store the full teacher output for covariance calculation
            W = torch.eye(self.input_dimension).to(hp.DEVICE) # This is your d*d identity matrix
            self.full_teacher_y = raw_X @ W
        xpath = self.save_data(raw_X, f"{j}")
        ypath = self.save_targets(raw_Y, f"{j}")

        model_identifier = f"{j}"

        # Pass all relevant config parameters to the model_builder
        model_config_for_builder = {
            'input_dimension': self.input_dimension,
            'hidden_width1': self.hidden_width1,
            'hidden_width2': self.hidden_width2,
            'num_data_points': self.num_data_points,
            'num_ensembles': self.num_ensembles,
            'activation': self._activation, # Assuming activation function is general
            'output_activation': self._activation, # Assuming output activation is general
            'weight_sigma1': self.weight_sigma1,
            'weight_sigma2': self.weight_sigma2,
            'weight_sigma3': self.weight_sigma3,
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'lambda_3': self.lambda_3,
        }
        # Model-specific architecture details for manifest
        model_architecture_spec = {
            'kind': self.config.get('model_type', 'Custom'), # Allow defining model type
            'input_dim': self.input_dimension,
            **{k: v for k, v in model_config_for_builder.items() if k not in ['activation', 'output_activation']}
        }


        model = self.model_builder(model_config_for_builder).to(hp.DEVICE)

        manifest = self.add_model_to_manifest(
            model_identifier,
            model_architecture_spec,
            data_path=xpath,
            targets_path=ypath,
            langevin_noise=self.noise_std_ld,
            chi=self.chi,
            temperature=self.temperature,
            kappa=self.kappa,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs
        )
        return model, manifest, raw_X, raw_Y, model_identifier

    def train_model(self, trainer, logger, model_identifier, writer):

        """Trains a model with specified callbacks."""
        callbacks = self._TrainingCallbacks(self, writer, model_identifier)
        model = trainer.model
        # This line needs to be generalized or moved to a model-specific setup function
        if hasattr(model, '_reset_with_weight_sigma'): # For FCN3 and similar models
            model._reset_with_weight_sigma(self.weight_sigma)
        print(hp.DEVICE)

        trainer.train(
            logger=logger,
            continue_at_epoch=trainer.current_epoch,
            current_time_step=trainer.current_time_step,
            interrupt_callback=callbacks.interrupt,
            completion_callback=callbacks.completion,
            epoch_callback=callbacks.epoch,
        )

    def train_dataset(self, j, teacher, writer, model, raw_X, raw_Y, most_recent_epoch, most_recent_timestep):
        """Trains models for a single dataset."""
        is_running_new_model = model is None
        if is_running_new_model:
            model, manifest, raw_X, raw_Y, model_identifier = self.create_new_model(j, teacher)
        else:
            manifest = self.most_recent_model()
            model_identifier = f"{j}"

        trainer = LangevinTrainer( # This trainer is still FCN3 specific. Generalize or allow as config input.
            model=model,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            noise_std=self.noise_std_ld,
            manager=DataManager(raw_X, raw_Y.to(hp.DEVICE), split=0.95), # DataManager also expects raw_X, raw_Y on device
            weight_decay_config={
                'W0': self.lambda_1,
                'W1': self.lambda_2,
                'A': self.lambda_3,
            },
            num_epochs=manifest.get('num_epochs', self.num_epochs)
        )

        if not is_running_new_model:
            print(f"Training cached model: {model_identifier} starting at epoch: {most_recent_epoch}")
            trainer.current_epoch = most_recent_epoch
            trainer.current_time_step = most_recent_timestep
            if i > start_model_num:
                is_running_new_model = True

        logger = Logger(num_epochs=trainer.train_config['num_epochs'], completed=trainer.current_epoch, description=f"Training YURI {model_identifier}")
        with logger:
            print(f"Training parallelized network on dataset {j} for {self.num_epochs} epochs")
            self.train_model(trainer, logger, model_identifier, writer)

    def run_ensemble_training(self, desc='', ensemble_dir=None, config_file=None):
        """Run the ensemble training and saving process."""
        # Load config which will update instance attributes
        self.load_config(config_file)
        teacher = self.initialize_teacher()

        with SummaryWriter(log_dir=self.tensorboard_dir) as writer:
            try:
                most_recent_model_manifest = self.most_recent_model()
                start_dataset_num, start_model_num, model_identifier = self.get_model_indices(
                    most_recent_model_manifest
                )
                print(f"Running model: {model_identifier} dnum: {start_dataset_num}, mnum: {start_model_num}")

                model, raw_X, raw_Y, most_recent_epoch, most_recent_timestep = self.load_existing_model(most_recent_model_manifest)

                for j in range(start_dataset_num, self.num_datasets):
                    self.train_dataset(
                        j, teacher, writer,
                        model, raw_X, raw_Y, most_recent_epoch, most_recent_timestep
                    )
                    model, raw_X, raw_Y, most_recent_epoch, most_recent_timestep = None, None, None, 0, 0
                    start_model_num = 0

            except KeyboardInterrupt:
                print("Training interrupted")
                self.destruct()
                writer.close()


class NetworksLoader:
    """
    An iterable for cycling through trained networks so that they
    don't need to all be loaded into memory at once.
    """
    def __init__(self, ensemble_manager_instance):
        """
        Initializes the NetworksLoader.

        Args:
            ensemble_manager_instance: An instance of the EnsembleManager class.
                                       This instance is expected to have a
                                       'training_config' attribute which contains
                                       a 'manifest' list, and an 'ensemble_dir'
                                       attribute for resolving file paths.
        """
        self.ensemble_manager = ensemble_manager_instance
        self._current_index = 0

        self.manifest = self.ensemble_manager.training_config.get('manifest', [])
        if not self.manifest:
            print("Warning: NetworksLoader initialized with an empty manifest from EnsembleManager. No networks to load.")

    def __iter__(self):
        """
        Returns the iterator object itself.
        Resets the current index to allow for multiple iterations over the networks.
        """
        self._current_index = 0
        return self

    def __next__(self):
        """
        Loads and returns the next network, its associated data, and target from the manifest.
        Raises StopIteration when all networks have been yielded.
        """
        if self._current_index < len(self.manifest):
            model_manifest = self.manifest[self._current_index]
            self._current_index += 1

            model_path = model_manifest.get('model_path')
            raw_x_path = model_manifest.get('raw_X_path')
            raw_y_path = model_manifest.get('raw_Y_path')

            if not model_path or not os.path.exists(model_path):
                print(f"Warning: Model file not found or path missing for manifest entry {self._current_index - 1} (path: {model_path}). Skipping this entry.")
                return self.__next__()

            if not raw_x_path or not os.path.exists(raw_x_path):
                print(f"Warning: Raw X data file not found or path missing for manifest entry {self._current_index - 1} (path: {raw_x_path}). Skipping this entry.")
                return self.__next__()

            if not raw_y_path or not os.path.exists(raw_y_path):
                print(f"Warning: Raw Y data file not found or path missing for manifest entry {self._current_index - 1} (path: {raw_y_path}). Skipping this entry.")
                return self.__next__()

            try:
                model = torch.load(model_path, weights_only=False).to(hp.DEVICE)
                data = torch.load(raw_x_path).to(hp.DEVICE)
                target = torch.load(raw_y_path).to(hp.DEVICE)
            except Exception as e:
                print(f"Error loading network or data for manifest entry {self._current_index - 1} (model: {model_path}): {e}. Skipping this entry.")
                return self.__next__()

            return {
                'model': model,
                'data': data,
                'target': target,
                'model_manifest': model_manifest
            }
        else:
            raise StopIteration


def main():
    """Main function to parse arguments and start the ensemble training."""
    parser = argparse.ArgumentParser(description="Process a file or set an ensemble directory.")
    parser.add_argument('-f', '--file', type=str, help="Specify an ensemble directory to process.")
    parser.add_argument('-c', '--config', type=str, help="Specify a YAML configuration file.")
    parser.add_argument('-d', '--desc', type=str, help="Add a note to describe this ensemble training run")

    args = parser.parse_args()

    ensemble_dir = args.file
    config_file = args.config


    # Load initial config from file to pass to EnsembleManager constructor
    initial_config_from_file = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                initial_config_from_file = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading initial YAML config: {e}. Proceeding without initial config.")

    # Define a default model builder for FCN3 if not provided in config
    def fcn3_model_builder(model_params):
        return FCN3NetworkEnsembleLinear(
            model_params['input_dimension'],
            model_params['hidden_width1'],
            model_params['hidden_width2'],
            model_params['num_data_points'],
            ensembles=model_params['num_ensembles']
        )

    def teacher_builder(input_dim):
            return SimpleNet(input_dim_id).to(hp.DEVICE) # Default to SimpleNet
    
    # Define a default epoch callback for FCN3 if not provided in config
    def fcn3_epoch_callback(trainer, writer, model_identifier, ensemble_manager_instance):
        if trainer.current_epoch % ensemble_manager_instance.CASH_FREAK != 0:
            return

        tag = f'Run_{ensemble_manager_instance.run_identifier}_Modelnum_{model_identifier}'

        with torch.no_grad():
            writer.add_scalar(f'{tag}/Train_Step', trainer.current_train_loss, trainer.current_time_step)
            H = compute_avg_channel_covariance_fcn3(trainer.model, trainer.manager.data)
            ensemble_manager_instance._log_matrix_to_tensorboard(writer, f'{tag}/H', H.cpu().numpy(), trainer.current_epoch)

            # Requires full_teacher_y to be set correctly by EnsembleManager
            lH = project_onto_target_functions(H, ensemble_manager_instance.full_teacher_y)
            lK = transform_eigenvalues(lH, 1.0, ensemble_manager_instance.chi, ensemble_manager_instance.num_data_points)

            scalarsH = {str(i): float(f"{value_li.cpu().numpy().item():.6g}") for i, value_li in enumerate(lH)}
            scalarsK = {str(i): float(f"{value_li.cpu().numpy().item():.6g}") for i, value_li in enumerate(lK)}
            writer.add_scalars(f'{tag}/eig_lH', scalarsH, trainer.current_epoch)
            writer.add_scalars(f'{tag}/eig_lK', scalarsK, trainer.current_epoch)

            # This part is specific to FCN3's 'fc2' layer.
            if hasattr(trainer.model, 'W0'):
                fc2_layer = trainer.model.W0 
                W = fc2_layer
                cov_W = torch.einsum('iaj,ibj->ab', W, W) / (ensemble_manager_instance.hidden_width1 * ensemble_manager.num_ensembles)
                ensemble_manager_instance._log_covariance_plot(writer, tag, cov_W.cpu().numpy(), trainer.current_epoch)

            else:
                print(f"Warning: Model has no layer W0. Skipping covariance plot.")

        writer.flush()

    
    model_type = initial_config_from_file.get("model_type", "fcn3")
    model_builders = {
#        "fcn2": fcn2_model_builder,
        "fcn3": fcn3_model_builder,
    }

    epoch_callbacks = {
        "fcn3": fcn3_epoch_callback
    }
    initial_config_from_file['model_builder'] = model_builders[model_type]
    initial_config_from_file['epoch_callback_func'] = epoch_callbacks[model_type]


    # Determine run_identifier
    if ensemble_dir:
        basename = unix_basename(ensemble_dir)
        run_identifier = basename.split('ensemble_', 1)[1] if basename.startswith('ensemble_') else basename
    else:
        dt = datetime.now().strftime("%a, %b %d %Y, %I:%M%p")
        # Use values from initial_config_from_file or hardcoded defaults for run_identifier
        input_dim_id = initial_config_from_file.get('input_dimension', 50)
        hidden_w2_id = initial_config_from_file.get('hidden_width2', 200)
        chi_val_id = initial_config_from_file.get('chi', hidden_w2_id)
        kappa_val_id = initial_config_from_file.get('kappa', 1.0 / chi_val_id if chi_val_id != 0 else 0)
        temp_val_id = initial_config_from_file.get('temperature', 2 * kappa_val_id)
        run_identifier = f"Ensemble_D{input_dim_id}_N{hidden_w2_id}_chi{chi_val_id}_T{temp_val_id}_{dt}"

    if 'teacher_builder' not in initial_config_from_file:
        initial_config_from_file['teacher_builder'] = teacher_builder

    ensemble_manager = EnsembleManager(
        run_identifier=run_identifier,
        deletedir=ensemble_dir is None,
        desc=initial_config_from_file.get('description', 'Training an ensemble of FCN3'),
        config=initial_config_from_file # Pass the loaded/augmented config here
    )

    if ensemble_dir or config_file:
        print(f"Proceeding with ensemble directory: {ensemble_dir}, config file: {config_file}")
    ensemble_manager.run_ensemble_training(desc=initial_config_from_file.get('description', 'Trainining ens of FCN3s'), ensemble_dir=ensemble_dir, config_file=config_file)


if __name__ == "__main__":
    main()
