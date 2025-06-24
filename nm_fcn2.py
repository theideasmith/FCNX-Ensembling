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
from ensemble_manager import EnsembleManager
from Covariance import * 
from FCN2Network import FCN2Network
from FCN3 import *
import standard_hyperparams as hp
from datetime import datetime 
import shutil
from torch.utils.tensorboard import SummaryWriter
import sys
from json_handler import JsonHandler
import matplotlib.pyplot as plt
import io
from PIL import Image
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

# How often the writer should cace to disk
# cache frequency
CASH_FREAK = 2000
def log_matrix_to_tensorboard(
    writer: SummaryWriter,
    tag: str,
    matrix: np.ndarray,
    global_step: int,
):
    """
    Logs a NumPy matrix to TensorBoard as an image and/or a histogram.

    Args:
        writer (SummaryWriter): The TensorBoard SummaryWriter instance.
        tag (str): The tag for the TensorBoard entry (e.g., 'layer_weights/conv1').
        matrix (np.ndarray): The matrix (2D NumPy array) to log.
        global_step (int): The global step for the TensorBoard event.
        as_image (bool): If True, logs the matrix as a grayscale image (heatmap).
                         Assumes the matrix is 2D. Values will be normalized to [0, 1].
        as_histogram (bool): If True, logs the distribution of matrix elements as a histogram.
    """
    if not isinstance(matrix, np.ndarray):
        print(f"Warning: Input for tag '{tag}' is not a NumPy array. Skipping logging.")
        return

    if matrix.ndim != 2:
        print(f"Warning: Matrix for tag '{tag}' has {matrix.ndim} dimensions, "
              f"but 'as_image' expects a 2D matrix. Skipping image logging.")
    else:
        # Normalize matrix to [0, 1] for image display
        min_val = matrix.min()
        max_val = matrix.max()
        
        # Avoid division by zero for constant matrices
        if max_val - min_val > 1e-8: 
            normalized_matrix = (matrix - min_val) / (max_val - min_val)
        else:
            # If matrix is constant, set it to zeros for visualization
            normalized_matrix = np.zeros_like(matrix, dtype=np.float32) 

        # Add a channel dimension for grayscale image (1, Height, Width)
        # TensorBoard's add_image expects (C, H, W) or (H, W, C) for 2D images.
        # (1, H, W) is a common way to represent grayscale.
        image_tensor = np.expand_dims(normalized_matrix, axis=0) 

        writer.add_image(tag + '/image', image_tensor, global_step=global_step, dataformats='CHW')

INIT_SEED = 222
input_dimension: int = 50
hidden_width: int = 100
chi = 1 #hidden_width
kappa = 1.0  / chi
temperature = 2 * kappa
torch.manual_seed(INIT_SEED)

N = hidden_width
lambda_1 = temperature * input_dimension #weight decay factor
lambda_2 = temperature * hidden_width
weight_sigma1: float = 1.0/input_dimension
weight_sigma2: float = 1.0/(hidden_width )
# Full batch in langevin dynamics
num_data_points: int = 300
batch_size : int =num_data_points
# Learning rate has to be normalized to the number of data points
learning_rate: float =  1e-3/num_data_points  
noise_std_ld: float = (2 * learning_rate * temperature )**0.5
num_epochs: int = 50000

weight_sigma = (weight_sigma1,
    weight_sigma2)

def epoch_callback_fcn2(trainer, writer, ensemble_manager, model_identifier):
    if trainer.current_epoch % CASH_FREAK !=0: return
    if False:
        writer.add_scalar(f'FCN2_Run_{ensemble_manager.run_identifier}_Modelnum_{model_identifier}/Train_Step', 
                                trainer.current_train_loss, trainer.current_time_step)
        H = compute_avg_channel_covariance(trainer.model, trainer.manager.data, layer_name='fc1')
        log_matrix_to_tensorboard(writer,f'FCN2_Run_{ensemble_manager.run_identifier}_Modelnum_{model_identifier}/H', H.cpu().numpy(),trainer.current_epoch)  
        X = trainer.manager.data
        d = X.shape[-1] # Get d from X's last dimension
        
        w = torch.eye(d).to(hp.DEVICE) # This is your d*d identity matrix

        y = X @ w # Matrix multiplication

        lH = project_onto_target_functions(H,y )
        lK = transform_eigenvalues(lH, 1.0, chi, num_data_points)
#
#     
        scalarsH= {}
        scalarsK={}
        for i, value_li in enumerate(lH):
        # The tag will group all series under 'List_Evolution' in TensorBoard's UI.
        # The specific series will be 'Item_0', 'Item_1', etc.
            scalarsH[str(i)] = value_li.cpu().numpy().item()
        for i, value_li in enumerate(lK):
            scalarsK[str(i)] = value_li.cpu().numpy().item()

        firstpairs = {k: scalarsH[k] for k in list(scalarsH)}

        writer.add_scalars(f'FCN2_Run_{ensemble_manager.run_identifier}_Modelnum_{model_identifier}/eig_lH',scalarsH,trainer.current_epoch)
        writer.add_scalars(f'FCN2_Run_{ensemble_manager.run_identifier}_Modelnum_{model_identifier}/eig_lK',scalarsK,trainer.current_epoch)

    if trainer.current_epoch % CASH_FREAK != 0: return
    with torch.no_grad():
        W = trainer.model.fc1.weight.detach().T
        cov_W = torch.einsum('aj,bj->ab', W, W) /      N
        cov_W = cov_W.cpu().numpy()
        diag = np.diag(cov_W)
        w0 = diag[0]
        wn = np.mean(diag[1:])
#       writer.add_scalars(f'FCN2_Run_{ensemble_manager.run_identifier}_Modelnum_{model_identifier}/weights',{'w0': w0, '<wp>': wn}, trainer.current_epoch)
        
        # Create a Matplotlib figure
        fig,( ax1,ax2)= plt.subplots(2,1, figsize=(10, 20)) # Adjust size as needed
        
        # Determine title and axis labels based on rowvar
        title = f'Covariance of Output Features (Step {trainer.current_epoch})'
        xlabel = 'Output Feature Index'
        ylabel = 'Output Feature Index'
        
        # Use imshow to visualize the matrix. 'coolwarm' is excellent for diverging data (positive/negative).
        # Setting vmin/vmax consistently is crucial for comparing evolution across steps.
        # You might need to adjust these based on the expected range of your covariance values.
        # Dynamically setting them based on current matrix's min/max:
        abs_max = np.max(np.abs(cov_W ))
        im = ax1.imshow(cov_W, cmap='viridis', vmin=-abs_max, vmax=abs_max)
        # Or, if you know your expected range:
        # im = ax.imshow(cov_matrix_np, cmap='coolwarm', vmin=-1.0, vmax=1.0) # Example fixed range for correlation-like values
        
        ax1.set_title(title)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04) # Add color bar
        
        
        diagonal_values = np.diag(cov_W)
        x_indices = np.arange(len(diagonal_values))
        
        ax2.plot(x_indices, diagonal_values, marker='o', linestyle='-', color='red')
        ax2.set_title('Main Diagonal of the fc1 Cov Matrix')
        ax2.set_xlabel('Diagonal Element Index')
        ax2.set_ylabel('Value')
        ax2.grid(True) # Add a grid for better readability
        ax2.set_xticks(x_indices) # Ensure ticks are at each index
        
        # Convert the matplotlib figure to an image (PNG) in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1) # save without extra padding
        plt.close(fig) # Close the figure to free memory and prevent display
        buf.seek(0)
        
        # Open with PIL and convert to NumPy array (HWC)
        pil_image = Image.open(buf)
        image_np_hwc = np.array(pil_image)
        
        # TensorBoard add_image for numpy expects (H, W, C) or (C, H, W)
        # If it's RGB from Matplotlib, it will be (H, W, 3).
        # PyTorch's add_image can handle HWC if dataformats='HWC' is specified.
        # Or convert to (C, H, W) for default behavior:
        image_tensor_chw = torch.from_numpy(image_np_hwc).permute(2, 0, 1).float() / 255.0 # Normalize to 0-1 if RGB
        
        # Log the image to TensorBoard
        writer.add_image(  f'FCN2_Run_{ensemble_manager.run_identifier}_Modelnum_{model_identifier}/fc2.Weights_Cov', image_tensor_chw, global_step=trainer.current_epoch)

    writer.flush()    

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
        dt = datetime.now()
        formatted_datetime = dt.strftime("%a, %b %d %Y, %I:%M%p")
        run_identifier = f"FCN2_D_{input_dimension}_N_{hidden_width}_chi_{chi}_T_{temperature}_{formatted_datetime}"
        print(f'./Menagerie/{run_identifier}/')

        # Initialize JsonHandler and EnsembleManager

        ensemble_manager = EnsembleManager(
                config={'num_ensembles': 1, 'num_datasets': 1},
                deletedir=True,
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
            input_dimension,
        ).to(hp.DEVICE)
        torch.save(teacher, os.path.join(ensemble_manager.ensemble_dir, 'teacher_network.pth')) #save the teacher to the ensemble dir
        ensemble_manager.teacher = teacher #store the teacher
    with SummaryWriter(log_dir=ensemble_manager.tensorboard_dir) as writer:
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
                    raw_X = HypersphereData.sample_hypersphere(num_data_points, input_dimension, normalized=False).to(hp.DEVICE)
                    # I spoke with Zohar – no need to add gaussian noise. It compllicates too much
                    raw_Y = teacher(raw_X) #+ (torch.randn(hp.NUM_DATA_POINTS, 1) * hp.TARGET_NOISE).to(hp.DEVICE)
                    xpath = ensemble_manager.save_data(raw_X, f"{j}")
                    ypath = ensemble_manager.save_targets(raw_Y, f"{j}")

                for i in range(start_model_num, ensemble_manager.training_config['num_ensembles']):

                    manifest = most_recent_model_manifest
                    if is_running_new_model is True:
                            # Create and train the model
                        weight_decay_config = {
                                    'fc1.weight': lambda_1,
                                    'fc2.weight': lambda_2,
                                    'fc1.bias': 0.0,
                                    'fc2.bias': 0.0
                        }
                        print(activation('From main script, this is the activation'))
                        hyperparameters = {
                            'input_dimension': input_dimension,
                            'hidden_width': hidden_width,
                            'activation': activation,
                            'weight_sigma1': weight_sigma1,
                            'weight_sigma2': weight_sigma2,
                        }
                        model: FCN2Network = FCN2Network.model_from_hyperparameters(
                                hyperparameters).to(hp.DEVICE)

                        model_identifier = f"{j * ensemble_manager.num_ensembles + i}"
                        # Store the configuration of the network
                        model_architecture_spec = {
                            'kind': 'FCN2',
                            'input_dim': input_dimension,
                            'hidde_width': hidden_width,
                            'weight_sigma':(weight_sigma1,weight_sigma2),
                            'weight_decay':(lambda_1,lambda_2)
                        }
                        manifest = ensemble_manager.add_model_to_manifest(
                            model_identifier,
                            model_architecture_spec,
                            data_path = xpath,
                            targets_path = ypath,
                            langevin_noise = noise_std_ld,
                            chi = chi,
                            temperature = temperature,
                            kappa = kappa,
                            learning_rate = learning_rate,
                            num_epochs=num_epochs
                        )

                    max_epochs  = manifest.get('num_epochs', num_epochs)
                    bsize =  batch_size

                    lrate = learning_rate

                    
                    trainer = LangevinTrainer(
                        model=model,
                        batch_size = bsize,
                        learning_rate = lrate,
                        noise_std = noise_std_ld,
                        manager= DataManager(raw_X.detach().to(hp.DEVICE), raw_Y.detach().to(hp.DEVICE), split=0.95),
                        weight_decay_config=weight_decay_config,
                        on_data=False,
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
                        model._reset_with_weight_sigma(weight_sigma)

                        # I know this is some of the most unreadable python code imaginable
                        # but I thought it was cool to introduce some functional 
                        # programming style. 
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
                                lambda trainer: do([
                                    lambda: ensemble_manager.save_model(model,
                                        model_identifier,
                                        converged = trainer.converged,
                                        current_time_step = trainer.current_time_step,
                                        epochs_trained = trainer.current_epoch),
                                    ]),
                            epoch_callback =lambda trainer: do([
                                lambda: epoch_callback_fcn2(trainer, writer, ensemble_manager, model_identifier),
                            ])
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
        main(ensemble_dir=ensemble_dir, desc='Jun 12; I am testing the GP limit for d=50, P=300, N=100, one dataset, one model')
    elif not CONTINUE_FROM_LAST:
        main()
