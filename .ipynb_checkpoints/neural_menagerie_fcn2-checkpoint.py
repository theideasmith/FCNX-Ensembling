"""
This file contains code for training an ensemble of neural networks using a single teacher network
to generate the training data. It includes a JsonHandler class for saving/loading data
and an EnsembleManager class for managing the training process and data persistence.
"""
from utils import unix_basename
import torch
import numpy as np
import json
import os
from FCN2Network import FCN2Network
from FCN3 import *
import standard_hyperparams_fcn2 as hp2
from datetime import datetime
import shutil
from torch.utils.tensorboard import SummaryWriter
import sys
from json_handler import JsonHandler
from ensemble_manager import EnsembleManager
import argparse
SELF_DESTRUCT=False
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



def main(desc = '',
         ensemble_dir=None):
    """
    Main function to run the ensemble training and saving process.
    """

    run_identifier = ''
    ensemble_manager = None
    json_handler = JsonHandler() #dont pass dir here
    print(f'Ensembling in directory {ensemble_dir}')
    if ensemble_dir is None:
        print(f'Ens dir thought to be None')
        print(f'Creating a new ensemble directory under the name')
        # Generate a unique identifier for this training run
        now = datetime.now()
        run_identifier = f"FCN2_TRAIN_{now.strftime('%Y%m%d_%H%M%S')}"
        print(f'./Menagerie/{run_identifier}/')

        # Initialize JsonHandler and EnsembleManager

        ensemble_manager = EnsembleManager(
                run_identifier=run_identifier,
                json_handler=json_handler, desc = desc) #pass run identifier
    else:
        basename = unix_basename(ensemble_dir)
        print(f'Basename are {basename}')
        run_identifier = basename.split('ensemble_', 1)[1] if basename.startswith('ensemble_') else basename
        print(run_identifier)
        ensemble_manager = EnsembleManager(
                run_identifier=run_identifier,
                json_handler=json_handler)
    """
    The logic I want to implement here is follows.
    If the ensemblemanager exists AND all models have _NOT_ been trained
    then we will RESUME TRAINING at the model given.

    Otherwise, we will continue with the existing training loop

    How this works:
    1. Search through the model manifest and find the nth model.
    2. Determine the dataset number and model number
    3. Start the training loop by initializing with
        (1) Dataset number
        (2) Model number
        (3) Configure the NetworkTrainer with the current epoch of the model
    4. Continue training loop
    """



    """
    If the teacher network exists, then load it
    If the teacher network does _not_ exist, then create it.

    Search through the models in the manifest and train the most
    recent one that has not completed training.

    We do that first by searching through the manifest, finding
    the first model whose converged == False

    """

    # Via command line arguments, check whether there has been
    # passed a training directory.
    teacher = None
    if ensemble_manager.teacher_exists():
        teacher = ensemble_manager.load_cached_teacher()
    else:

    # 1. Generate a single teacher network
        teacher = SimpleNet(
            hp2.INPUT_DIMENSION,
            activation,
            1.0 / hp.INPUT_DIMENSION, # No need to divide here.
        ).to(hp.DEVICE)
        torch.save(teacher, os.path.join(ensemble_manager.ensemble_dir, 'teacher_network.pth')) #save the teacher to the ensemble dir
        ensemble_manager.teacher = teacher #store the teacher

    writer = SummaryWriter(
                log_dir=ensemble_manager.tensorboard_dir
    )


    most_recent_model_manifest = ensemble_manager.most_recent_model()
    start_dataset_num = 0
    start_model_num = 0
    model_identifier = None
    if most_recent_model_manifest is None:
        mid = len(ensemble_manager.training_config['manifest'])
        model_identifier = mid
        nens = int(ensemble_manager.training_config['num_ensembles'])
        start_dataset_num = int((mid - mid%nens)/nens)
        start_model_num =  int(mid%nens)
        print(f'Running model: {mid} dnum: {start_dataset_num}, mnum:{start_model_num}')
    else:
        mid = int(most_recent_model_manifest.get('model_identifier'))
        model_identifier = mid
        nens = int(ensemble_manager.training_config['num_ensembles'])
        start_dataset_num = int((mid - mid%nens)/nens)
        start_model_num =int((mid%nens))
        print(f'Running model: {mid} dnum: {start_dataset_num}, mnum:{start_model_num}')
        # return
    try:
        # 2. Generate datasets and train ensembles

        is_model = most_recent_model_manifest is not None
        raw_X = None
        raw_Y = None
        xpath = None
        ypath = None
        model = None
        most_recent_epoch = 0
        most_recent_timestep = 0
        if  is_model:
            print("MODEL FOUND")
            model_path = most_recent_model_manifest['model_path']

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            most_recent_model_data = {
                'model': torch.load(model_path, weights_only=False).to(hp.DEVICE),
                'data': torch.load(most_recent_model_manifest['raw_X_path']).to(hp.DEVICE),
                'target': torch.load(most_recent_model_manifest['raw_Y_path']).to(hp.DEVICE)
            }

            model = most_recent_model_data['model']
            raw_X = most_recent_model_data['data']
            raw_Y = most_recent_model_data['target']
            xpath = most_recent_model_manifest['raw_X_path']
            ypath = most_recent_model_manifest['raw_Y_path']
            most_recent_epoch = int(most_recent_model_manifest['epochs_trained'])
            most_recent_timestep = int(most_recent_model_manifest['current_time_step'])

        for j in range(start_dataset_num,3): #loop 3 times for the datasets
            is_running_new_model =  not (is_model and j==start_dataset_num) or not is_model
            if is_running_new_model is True:
                raw_X = HypersphereData.sample_hypersphere(hp2.NUM_DATA_POINTS, hp2.INPUT_DIMENSION, normalized=False).to(hp.DEVICE)
                # I spoke with Zohar – no need to add gaussian noise. It compllicates too much
                raw_Y = teacher(raw_X) #+ (torch.randn(hp.NUM_DATA_POINTS, 1) * hp.TARGET_NOISE).to(hp.DEVICE)
                xpath = ensemble_manager.save_data(raw_X, f"{j}")
                ypath = ensemble_manager.save_targets(raw_Y, f"{j}")

            for i in range(start_model_num, ensemble_manager.training_config['num_ensembles']):

                manifest = most_recent_model_manifest
                if is_running_new_model is True:
                # Create and train the model
                model = FCN2Network(
                    hp2.INPUT_DIMENSION,
                    hp2.FCN2_HIDDEN_WIDTH,
                    activation,
                    hp2.FCN2_WEIGHT_SIGMA
                ).to(hp.DEVICE)
                
                model_identifier = f"{j * ensemble_manager.num_ensembles + i}"
                # Store the configuration of the network
                model_architecture_spec = {
                    'kind': 'FCN2',
                    'input_dim': hp2.INPUT_DIMENSION,
                   'hidden_width': hp2.FCN2_HIDDEN_WIDTH,
                    'weight_sigma':hp2.FCN2_WEIGHT_SIGMA,
                    'weight_decay':(hp2.FCN2_LAMBDA_1,hp2.FCN2_LAMBDA_2)
                }
                manifest = ensemble_manager.add_model_to_manifest(
                    model_identifier,
                    model_architecture_spec,
                    data_path = xpath,
                    targets_path = ypath,
                    langevin_noise = hp2.NOISE_STD_LANGEVIN,
                    chi = hp2.CHI,
                    temperature = hp2.TEMPERATURE,
                    kappa = hp2.KAPPA,
                    learning_rate = hp2.LEARNING_RATE,
                    num_epochs=hp2.NUM_EPOCHS
                )

                max_epochs  = manifest.get('num_epochs', hp2.NUM_EPOCHS)
                batch_size =  manifest.get("batch_size", hp2.BATCH_SIZE)
                weight_decay_config = manifest.get("weight_decay_config")
                learning_rate = manifest.get("learning_rate", hp2.LEARNING_RATE)
                trainer = LangevinTrainer(
                    model=model,
                    batch_size = batch_size,
                    learning_rate = learning_rate,
                    noise_std = manifest.get("langevin_noise", hp2.NOISE_STD_LANGEVIN),
                    manager= DataManager(raw_X.detach().to(hp.DEVICE), raw_Y.detach().to(hp.DEVICE), split=0.8),
                    weight_decay_config=weight_decay_config,
                    num_epochs=max_epochs
                )

                if not is_running_new_model:
                    print(f'Training cached_model:{model_identifier}')
                    print(f'Starting at epoch: {most_recent_epoch}')
                    trainer.current_epoch = most_recent_epoch
                    trainer.current_time_step = most_recent_timestep
                print(f'Initializing training for model: {model_identifier} @ {max_epochs} epochs')
                logger = Logger(num_epochs=max_epochs,
                                completed=trainer.current_epoch,
                                description=f"Training YURI {model_identifier}")
                with logger:
                    print(f"Training network {i} on dataset {j}")
                    trainer.train(
                        logger = logger,
                        continue_at_epoch = trainer.current_epoch,
                        current_time_step = trainer.current_time_step,
                        interrupt_callback = lambda trainer: do([
                            lambda: ensemble_manager.save_model(
                                trainer.model, model_identifier,
                                current_time_step = trainer.current_time_step,
                                converged = False,
                                epochs_trained = trainer.current_epoch),
                            lambda: writer.close(),
                            lambda: writer.flush,
                            lambda: destruct(ensemble_manager.ensemble_dir)
                        ])
                        ,
                        completion_callback =
                            lambda trainer: ensemble_manager.save_model(model,
                                    model_identifier,
                                    converged = trainer.converged,
                                    current_time_step = trainer.current_time_step,
                                    epochs_trained = trainer.current_epoch),
                        epoch_callback =lambda trainer: writer.add_scalar(f'Loss_FCN2_Run_{ensemble_manager.run_identifier}_Modelnum_{model_identifier}/Train_Step', trainer.current_train_loss, trainer.current_time_step)
                    )
    except KeyboardInterrupt:
        print("The training loop ended with a keyboard interrupt")
        destruct(ensemble_manager.ensemble_dir)
        writer.close()


if __name__ == "__main__":

    # Initialize default values
    CONTINUE_FROM_LAST = False
    ensemble_dir = None

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process a file or set an ensemble directory."
    )
    parser.add_argument(
        '-f', '--file',
        type=str,
        help="Specify an ensemble directory (filename) to process."
    )

    # Parse command-line arguments
    args = parser.parse_args()
    print(args)
    print(args.file)
    # Apply logic based on parsed arguments
    if args.file:
        ensemble_dir = args.file
        print(f"Running with -f option: ensemble_dir set to '{ensemble_dir}'")
        # You can call process_file here if the -f argument implies file processing
        # For now, based on your request, it just sets the variable.
        # process_file(ensemble_dir) # Uncomment if you want to process the file when -f is used
    else:
        print("Running vanilla: CONTINUE_FROM_LAST is False, ensemble_dir is None.")

    # Print current state of variables for demonstration
    print(f"Current state: CONTINUE_FROM_LAST = {CONTINUE_FROM_LAST}")
    print(f"Current state: ensemble_dir = {ensemble_dir}")
    # Example of how you might use these variables later in your script
    if ensemble_dir:
        print(f"Proceeding with ensemble directory: {ensemble_dir}")
        print("ENSING")
        main(ensemble_dir=ensemble_dir)
    elif not CONTINUE_FROM_LAST:
        main()
