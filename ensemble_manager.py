import standard_hyperparams as hp
import os
from json_handler import JsonHandler
import torch
import json
class EnsembleManager:
    """
    Manages the saving and loading of ensemble training results, including models, data, and metadata.
    """
    def __init__(self, run_identifier, 
                 json_handler=None, 
                 desc = ''): 
        """
        Initializes the EnsembleManager.

        Args:
            json_handler (JsonHandler, optional):  An instance of JsonHandler.
                                                If None, a default JsonHandler is created.
            run_identifier (str): A unique identifier for this training run.
        """
        self.run_identifier = run_identifier
        self.menagerie_dir = './Menagerie' #define the overall dir
        self.ensemble_dir = os.path.join(self.menagerie_dir, f'ensemble_{run_identifier}') #this is the ensemble specific directory
        self.tensorboard_dir = self.ensemble_dir
        self.json_handler = json_handler if json_handler else JsonHandler(directory=self.ensemble_dir) # pass ensemble dir to json handler
        # self.tensorwriter = SummaryWriter(log_dir=self.tensorboard_dir)
        self.training_config_path = os.path.relpath(os.path.join(self.ensemble_dir,'training_config'))

        if os.path.exists(self.ensemble_dir):
            self.training_config = self.json_handler.load_data(os.path.join(self.ensemble_dir,'training_config'))
            return 
        
        if not os.path.exists(self.ensemble_dir): #make the ensemble dir
            os.makedirs(self.ensemble_dir)
        
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        self.models = []
        self.trainers = []
        self.teacher = None  # Store the teacher network
        self.num_ensembles = 20 #set num ensembles to a constant
        self.num_datasets = 3 # We are generating three datasets. 
        self.training_config = { #store the training config here and update.  REMOVED network configs
            'run_identifier': self.run_identifier,
            'num_ensembles': self.num_ensembles,
            'num_datsets': 3,
            "description": desc, 
            "manifest": []
        }

        self.json_handler.save_data(self.training_config, self.training_config_path)

    def save_teacher(self):
        # Save the teacher network in the ensemble dir.

        if self.teacher is not None:
            torch.save(self.teacher, os.path.join(self.ensemble_dir, 'teacher_network.pth'))

    def add_model_to_manifest(self, 
                           model_identifier, 
                           model_architecture,
                           data_path = '',
                           targets_path = '',
                           current_epoch = None, 
                           converged = False):
        config_path = self.training_config_path
        model_config = {
                    "model_identifier": model_identifier,
                    "network_architecture": model_architecture,
                    "input_dimension": hp.INPUT_DIMENSION,
                    "num_epochs": hp.NUM_EPOCHS,
                    "num_data_points": hp.NUM_DATA_POINTS,
                    "target_noise": hp.TARGET_NOISE,
                    "weight_decay_config": hp.WEIGHT_DECAY_CONFIG,
                    "network_dir": os.path.join(self.ensemble_dir, f'model_{model_identifier}'),
                    "model_path": os.path.join(self.ensemble_dir, f'model_{model_identifier}', f"model_{model_identifier}.pth"),
                    "raw_X_path": data_path,
                    "raw_Y_path": targets_path,
                    "epochs_trained": 0 if current_epoch is None else current_epoch,
                    "converged": converged
                }
        self.training_config['manifest'].append(model_config)
        self.json_handler.save_data(self.training_config, config_path)
        return model_config
    def save_model(self, model, model_identifier, converged = True, epochs_trained = 0):

        # Save the model and data immediately after training
        network_dir = os.path.join(self.ensemble_dir, f'network_{model_identifier}')
        if not os.path.exists(network_dir):
            os.makedirs(network_dir)
        
        self.model_update(model_identifier, 'converged', converged)
        self.model_update(model_identifier, 'epochs_trained', epochs_trained)

        torch.save(model, os.path.join(network_dir, f'network_{model_identifier}'))

    def model_update(self, model_name, key, updated_value):
        index = next((i for i, m in enumerate(self.training_config['manifest']) if m.get('model_name') == model_name), -1)

        try:
            with open(f'{self.training_config_path}.json', 'w') as f:

                model_dict = self.training_config['manifest'][index] 
                model_dict[key] = updated_value
            # Now 'data' holds the content of the JSON file as a Python dictionary or list
                self.json_handler.save_data(self.training_config, self.training_config_path)
        except FileNotFoundError:
            print(f"Error: The file '{self.training_config_path}' was not found.")
            data = {} # Initialize with an empty dictionary if file not found, or handle as needed
        except json.JSONDecodeError:
            print(f"Error: The file '{self.training_config_path}' does not contain valid JSON.")
            data = {} # Initialize or handle as needed

    def save_data(self, X, x_identifer):
        xpath = os.path.join(self.ensemble_dir, f"raw_X_{x_identifer}.pth")
        torch.save(X, xpath)
        return xpath
    
    def save_targets(self, Y, y_identifer): 
        ypath = os.path.join(self.ensemble_dir, f"raw_Y_{y_identifer}.pth")
        torch.save(Y, ypath)
        return ypath

    def load_ensemble_data(self):
        """
        Loads the trained models, training data, and teacher network.
        """
        # Load the training config to get the structure of the paths
        training_config_path = os.path.join(self.ensemble_dir, 'training_config.json')
        if not os.path.exists(training_config_path):
            raise FileNotFoundError(f"Training config file not found: {training_config_path}")
        
        self.training_config = self.json_handler.load_data(os.path.join(self.ensemble_dir,'training_config'))
        training_config = self.training_config
        # Load the teacher network
        teacher_path = os.path.join(self.ensemble_dir, 'teacher_network.pth')
        if os.path.exists(teacher_path):
            self.teacher = torch.load(teacher_path, weights_only=False).to(hp.DEVICE)  # Load to the correct device
        else:
            raise FileNotFoundError(f"Teacher network file not found: {teacher_path}")
        print(training_config)
        # Load the raw data and models for each ensemble member
        models = []
        data_files = {}
        target_files = {}
        for i in range(training_config['num_datsets']): #use num_ensembles from config
            raw_x_path = os.path.join(self.ensemble_dir, f'raw_X_{i}.pth')
            raw_y_path = os.path.join(self.ensemble_dir, f'raw_Y_{i}.pth')
            if not os.path.exists(raw_x_path):
                raise FileNotFoundError(f"Raw X data file not found: {raw_x_path}")
            if not os.path.exists(raw_y_path):
                raise FileNotFoundError(f"Raw Y data file not found: {raw_y_path}")
            data_files[raw_x_path] = torch.load(raw_x_path).to(hp.DEVICE) # Load to the correct device
            target_files[raw_y_path] = torch.load(raw_y_path).to(hp.DEVICE) # Load to the correct device

        if len(training_config['manifest'])==0:
            for j in range(self.training_config['num_ensembles']):
                for i in range(self.training_config['num_datsets']):
                    mnum = i * self.training_config['num_ensembles'] + j
                    training_config['manifest'].append({
                        'model_path': os.path.join(self.ensemble_dir, f'network_{mnum}/network_{mnum}'),
                        'raw_X_path': os.path.join(self.ensemble_dir, f'raw_X_{i}.pth'),
                        'raw_Y_path': os.path.join(self.ensemble_dir, f'raw_Y_{i}.pth')
                    })

        for model_manifest in training_config['manifest']:
            model_path = model_manifest['model_path']

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            models.append({
                'model': torch.load(model_path, weights_only=False).to(hp.DEVICE),
                'data': data_files[model_manifest['raw_X_path']],
                'target': target_files[model_manifest['raw_Y_path']]
            })
        return models