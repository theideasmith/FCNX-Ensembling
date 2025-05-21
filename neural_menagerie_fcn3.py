"""
This file contains code for training an ensemble of neural networks using a single teacher network
to generate the training data. It includes a JsonHandler class for saving/loading data
and an EnsembleManager class for managing the training process and data persistence.
"""
import torch
import numpy as np
import json
import os
from FCN3 import *
import standard_hyperparams as hp
from datetime import datetime
import shutil

from json_handler import JsonHandler 
from ensemble_manager import EnsembleManager
from torch.utils.tensorboard import SummaryWriter

SELF_DESTRUCT = False

def destruct(deldir):
    if SELF_DESTRUCT:
    # print(f"SIMULATING DELETING: {ensemble_manager.ensemble_dir}")
        shutil.rmtree(deldir)
        raise KeyboardInterrupt

def do(fs):
    for f in fs:
        f()
    return 
def activation(x):
    return x

def main(desc = ''):
    """
    Main function to run the ensemble training and saving process.
    """
    # Generate a unique identifier for this training run
    now = datetime.now()
    run_identifier = now.strftime("%Y%m%d_%H%M%S")

    # Initialize JsonHandler and EnsembleManager
    json_handler = JsonHandler() #dont pass dir here
    ensemble_manager = EnsembleManager(run_identifier=run_identifier, json_handler=json_handler, desc = desc) #pass run identifier

    # 1. Generate a single teacher network
    teacher = SimpleNet(
        hp.INPUT_DIMENSION,
        activation,
        1.0 / hp.INPUT_DIMENSION,
    ).to(hp.DEVICE)
    torch.save(teacher, os.path.join(ensemble_manager.ensemble_dir, 'teacher_network.pth')) #save the teacher to the ensemble dir
    ensemble_manager.teacher = teacher #store the teacher
    writer = SummaryWriter(
                log_dir=ensemble_manager.tensorboard_dir
    )

    try: 
        # 2. Generate datasets and train ensembles
        for j in range(3): #loop 3 times for the datasets
            raw_X = HypersphereData.sample_hypersphere(hp.NUM_DATA_POINTS, hp.INPUT_DIMENSION, normalized=False).to(hp.DEVICE)
            # I spoke with Zohar – no need to add gaussian noise. It compllicates too much
            raw_Y = teacher(raw_X) #+ (torch.randn(hp.NUM_DATA_POINTS, 1) * hp.TARGET_NOISE).to(hp.DEVICE)
            xpath = ensemble_manager.save_data(raw_X, f"{j}")
            ypath = ensemble_manager.save_targets(raw_Y, f"{j}")
            os.path.join(ensemble_manager.tensorboard_dir)
            for i in range(ensemble_manager.num_ensembles):
                # Create and train the model
                model = FCN3Network(
                    input_dim=hp.INPUT_DIMENSION,
                    hidden_width_1=hp.HIDDEN_WIDTH_1,
                    hidden_width_2=hp.HIDDEN_WIDTH_2
                ).to(hp.DEVICE)
                model_identifier = f"{j * ensemble_manager.num_ensembles + i}"
                # Store the configuration of the network
                model_architecture_spec = {
                    'kind': 'FCN3',
                    'input_dim': hp.INPUT_DIMENSION,
                    'hidden_width_1': hp.HIDDEN_WIDTH_1,
                    'hidden_width_2': hp.HIDDEN_WIDTH_2,
                    'weight_sigma':hp.FCN3_WEIGHT_SIGMA,
                    'weight_decay':(hp.FCN3_LAMBDA_1,hp.FCN3_LAMBDA_2,hp.FCN3_LAMBDA_3),
                    'temperature':hp.TEMPERATURE
                }
                model_config = ensemble_manager.add_model_to_manifest(
                    model_identifier,
                    model_architecture_spec,
                    data_path = xpath,
                    targets_path = ypath
                )
                
                trainer = LangevinTrainer(
                    model=model,
                    manager= DataManager(raw_X.detach().to(hp.DEVICE), raw_Y.detach().to(hp.DEVICE), split=0.8),
                    weight_decay_config=hp.WEIGHT_DECAY_CONFIG,
                    num_epochs=hp.NUM_EPOCHS
                )
                logger = Logger(num_epochs=hp.NUM_EPOCHS, description=f"Training YURI {model_identifier}")

                with logger:
                    print(f"Training network {i} on dataset {j}")
                    trainer.train(
                        logger = logger,
                        interrupt_callback = lambda trainer: do([
                            lambda: ensemble_manager.save_model(trainer.model, model_identifier, converged = False, epochs_trained = trainer.current_epoch),
                            lambda: writer.close(),
                            lambda: destruct(ensemble_manager.ensemble_dir)
                        ])
                        , 
                        completion_callback = 
                            lambda trainer: ensemble_manager.save_model(model, 
                                    model_identifier, 
                                    converged = trainer.converged,
                                    epochs_trained = trainer.current_epoch),
                        epoch_callback =lambda trainer: do([
                            lambda: writer.add_scalar(f'Loss_FCN2_Run_{ensemble_manager.run_identifier}_Modelnum_{model_identifier}/Train_Step', trainer.current_train_loss, trainer.current_time_step),
                            lambda: writer.flush()
                            ]
                        )
                    )
    except KeyboardInterrupt: 
        print("The training loop ended with a keyboard interrupt")
        if SELF_DESTRUCT == True:
            destruct(ensemble_manager.ensemble_dir)
        writer.close()         
                    
                    


if __name__ == "__main__":
    try: 
        main(desc='Running with T~O(1)')
    except Exception as e:
        print(f'An error ocurred {e}')

    # # Example of loading the data
    # now = datetime.now()
    # run_identifier = now.strftime("%Y%m%d_%H%M%S")
    # ensemble_manager = EnsembleManager(run_identifier=run_identifier) #same number of ensembles as during training
    # ensemble_manager.load_ensemble_data()

    # # Now you can access the loaded data:
    # loaded_teacher = ensemble_manager.teacher
    # loaded_models = ensemble_manager.models
    # loaded_raw_X_data = ensemble_manager.raw_X_data
    # loaded_raw_Y_data = ensemble_manager.raw_Y_data

    # print("Data loaded successfully.  Here's a quick check:")
    # print(f"Loaded Teacher Network: {loaded_teacher}")
    # print(f"Number of loaded models: {len(loaded_models)}")

    # print(f"Shape of first loaded raw_X: {loaded_raw_X_data[0].shape}")