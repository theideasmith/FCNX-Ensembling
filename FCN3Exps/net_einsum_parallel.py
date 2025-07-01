"""
This file implements the FCN3 directly in weight space
using einsums, enabling parallelized training of ensembles. 

This is significantly, by an order of magnitude, faster
than training multiple networks in series
"""

import os
import sys


# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_script_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)
import torch
import numpy as np
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch import autograd
from datetime import datetime
from decimal import Decimal
import os
from PIL import Image
from opt_einsum import contract
import  standard_hyperparams_fcn2 as hp2


import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset

#from sklearn.metrics import mean_squared_error

import numpy as np
from numpy import linalg as LA
import random

# from mpltools import annotation

from scipy.io import savemat, whosmat
from scipy.optimize import fsolve
import scipy.signal

import time
import os
import sys
from tempfile import TemporaryFile

import copy
import pathlib

import matplotlib.pyplot as plt


import io
import signal
#import noisy_sgd
import glob
import argparse
from torch.utils.tensorboard import SummaryWriter
import csv

from tqdm import tqdm # Import tqdm

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description='Train a fcn3 neural network with specified hyperparameters.')
parser.add_argument('--chi', type=int, required=True, help='Set scaling regime (Chi)')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
parser.add_argument('--N', type=int, required=True, help='Hidden layer width (N)')
parser.add_argument('--D', type=int, required=True, help='Input dimension (d)')
parser.add_argument('--P', type=int, required=True, help='Number of training samples (n_train)')
parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train network for')
parser.add_argument('--to', type=str, required=True, help='Location where to store the results of the training') 
parser.add_argument('--ens', type=int, default=1, help='The number in the network of which network in an ensemble this is')
parser.add_argument('--off_data',action='store_true', help='Whether the network should be trained on data')
parser.add_argument('--ethereal', default=False,action='store_true', help='Dont save the model; debug mode')
parser.add_argument('--rate_decay', type=float, default=1.302e7, help='Decay constant for learning rate schedule')
parser.add_argument('--lr0', type=float, default=1e-3, help='Initial learning rate for schedule')
parser.add_argument('--lr_schedule', action='store_true', help='Whether to use a learning rate schedule')

args = parser.parse_args()


#---------------------------------------------------------------------------------------------------
# General Settings
dev_mode = args.ethereal
INIT_SEED = 222
DTYPE=torch.float32


train_seed, test_seed = 563,10
# Base path for saving
SAVE_PATH = args.to 
max_epochs = args.epochs
n_train = args.P # Use P from arguments
N = args.N     # Use N from arguments
chi = args.chi
d = args.D     # Use D from arguments
n_test = 200
s2, sa2, sh2, sw2 = 1.0, 1.0, 1.0, 1.0 # These are k, sa2, sw2
FL_scale = 1.0 # float(128)
activation = "lin"
eps = 0.
on_data = True if args.off_data is False else False
lr0 = args.lr0
lr_schedule = args.lr_schedule

ens = args.ens
# Update SAVE_PATH to include N, D, P
def trainid():
    add = ens
    return f"epochs_{max_epochs}_N:{N}_chi_{chi}_D:{d}_P:{n_train}_k:{s2}_ondata_{on_data}_{add}"

TRAINID = trainid()
SAVE_PATH = os.path.join(SAVE_PATH, TRAINID)

import shutil

if dev_mode is False:
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
# else:
#   shutil.rmtree(SAVE_PATH)
    # os.makedirs(SAVE_PATH)
#----------------------------------------------------------------------------------------------------
# Training and Network settings


from Covariance import *


class VectorizedSumSquaredDifferenceLoss(nn.Module):
    """
    A custom PyTorch loss class that calculates half the sum of the squared differences
    between ensemble output and labels.

    This class internally handles the broadcasting of labels to match the
    ensemble output dimensions before computing the squared difference and sum.

    Equivalent to: 0.5 * torch.sum((ensemble_output - labels)**2)
    """
    def __init__(self):
        super().__init__()
        # No specific parameters needed for this simple loss,
        # but you could add them here if you wanted configurable reductions, etc.

    def forward(self, ensemble_output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss.

        Args:
            ensemble_output (torch.Tensor): The output from the ensemble,
                                            e.g., shape [batch_size, 1, num_ensemble_members].
            labels (torch.Tensor): The ground truth labels,
                                   e.g., shape [batch_size, 1, 1].

        Returns:
            torch.Tensor: A scalar tensor representing half the total sum of squared differences.
        """
        # Step 1: Calculate the vectorized difference using broadcasting
        # labels (e.g., [30, 1, 1]) will be broadcast to match ensemble_output (e.g., [30, 1, 3])
        difference = ensemble_output - labels

        # Step 2: Square the differences
        squared_difference = difference ** 2

        # Step 3: Sum all the squared differences and multiply by 1/2
        # This will result in a single scalar value.
        loss = 0.5 * torch.sum(squared_difference) # Added multiplication by 0.5

        return loss

criterion = VectorizedSumSquaredDifferenceLoss()

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

# Assuming hp2.DEVICE is defined elsewhere or defaults to CPU/CUDA
# For this refactor, it's assumed to be accessible.
# If hp2.DEVICE is not defined, you might need to add:
DEVICE = "cuda:1" 



CASH_FREAK = 1000
def epoch_callback_fcn3(epoch, loss, writer, model, data, model_identifier):
    if epoch % CASH_FREAK !=0: return
    with torch.no_grad():
        if dev_mode is False:
            writer.add_scalar(f'FCN3_{TRAINID}/Train_Step',
                                loss, epoch)
        if dev_mode is False:
            if hasattr(model, 'W0'):                                                                    
                writer.add_histogram(f'FCN3_{TRAINID}/W0_hist', model.W0.detach().cpu().numpy(), epoch) 

        # Add histograms of the first and second layer weights
        if not dev_mode:
            if hasattr(model, 'W1'):
                writer.add_histogram(f'FCN3_{TRAINID}/W1_hist', model.W0.detach().cpu().numpy(), epoch)
            if hasattr(model, 'A'):
                writer.add_histogram(f'FCN3_{TRAINID}/A_hist', model.A.detach().cpu().numpy(), epoch)
#       H = compute_avg_channel_covariance(model, data, layer_name='lin1')
#       log_matrix_to_tensorboard(writer,f'FCN3_{model_identifier}/H', H.cpu().numpy(),epoch)
#       X = data
#       d = X.shape[-1] # Get d from X's last dimension
#
#       w = torch.eye(d).to(hp2.DEVICE) # This is your d*d identity matrix
#
#       y = X @ w # Matrix multiplication
#
#       lH = project_onto_target_functions(H,y )
#       lK = transform_eigenvalues(lH, 1.0, chi, n_train)
#
#       scalarsH= {}
#       scalarsK={}
#       for i, value_li in enumerate(lH):
#           scalarsH[str(i)] = value_li.cpu().numpy().item()
#       for i, value_li in enumerate(lK):
#           scalarsK[str(i)] = value_li.cpu().numpy().item()
#
#       writer.add_scalars(f'FCN3_{model_identifier}/eig_lH',scalarsH,epoch)
#       writer.add_scalars(f'FCN2_{model_identifier}/eig_lK',scalarsK,epoch)

    if epoch % CASH_FREAK != 0: return
    with torch.no_grad():
        # W: d, N, ensembles
        W0 = model.W0.permute(*torch.arange(model.W0.ndim - 1, -1, -1))
        W1 = model.W1.permute(*torch.arange(model.W1.ndim - 1, -1, -1))

        covW0W1 = contract('kje,ije,nme,kme->ine', W1,W0,W0,W1, backend='torch') / N

        # Average over the ensemble dimension
        cov_W_m = torch.mean(covW0W1, axis=2)
        lH = cov_W_m.diagonal().squeeze()
        scalarsH = {}

        for i in range(lH.shape[0]):
            scalarsH[str(i)] = lH[i]
        if dev_mode is False:
            writer.add_scalars(f'FCN3_{TRAINID}/lambda_H(W1)', scalarsH, epoch)

#       # Create a Matplotlib figure
#       fig,( ax1,ax2)= plt.subplots(2,1, figsize=(10, 20)) # Adjust size as needed
#
#       # Determine title and axis labels based on rowvar
#       title = f'Covariance of Output Features (Step {epoch})'
#       xlabel = 'Output Feature Index'
#       ylabel = 'Output Feature Index'
#
#       # Use imshow to visualize the matrix. 'coolwarm' is excellent for diverging data (positive/negative).
#       # Setting vmin/vmax consistently is crucial for comparing evolution across steps.
#       # You might need to adjust these based on the expected range of your covariance values.
#       # Dynamically setting them based on current matrix's min/max:
#       abs_max = np.max(np.abs(cov_W ))
#       im = ax1.imshow(cov_W, cmap='viridis', vmin=-abs_max, vmax=abs_max)
#
#       ax1.set_title(title)
#       ax1.set_xlabel(xlabel)
#       ax1.set_ylabel(ylabel)
#       fig.colorbar(im, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04) # Add color bar
#
#
#       diagonal_values = np.diag(cov_W)
#       x_indices = np.arange(len(diagonal_values))
#
#       ax2.plot(x_indices, diagonal_values, marker='o', linestyle='-', color='red')
#       ax2.set_title('Main Diagonal of the fc1 Cov Matrix')
#       ax2.set_xlabel('Diagonal Element Index')
#       ax2.set_ylabel('Value')
#       ax2.grid(True) # Add a grid for better readability
#       ax2.set_xticks(x_indices) # Ensure ticks are at each index
#
#       # Convert the matplotlib figure to an image (PNG) in memory
#       buf = io.BytesIO()
#       plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1) # save without extra padding
#       plt.close(fig) # Close the figure to free memory and prevent display
#       buf.seek(0)
#
#       # Open with PIL and convert to NumPy array (HWC)
#       pil_image = Image.open(buf)
#       image_np_hwc = np.array(pil_image)
#
#       # TensorBoard add_image for numpy expects (H, W, C) or (C, H, W)
#       # If it's RGB from Matplotlib, it will be (H, W, 3).
#       # PyTorch's add_image can handle HWC if dataformats='HWC' is specified.
#       # Or convert to (C, H, W) for default behavior:
#       image_tensor_chw = torch.from_numpy(image_np_hwc).permute(2, 0, 1).float() / 255.0 # Normalize to 0-1 if RGB
#
#       # Log the image to TensorBoard
#       writer.add_image(  f'FCN3_{model_identifier}/fc2.Weights_Cov', image_tensor_chw, global_step=epoch)

    writer.flush()

#-----------------------------------------------------------------------------------------------------
# Lengevin Optimizer

class LangevinSimple2(optim.Optimizer):


    def __init__(self, model: nn.Module, learning_rate, weight_decay_1, weight_decay_2, weight_decay_3, temperature):

        defaults = {
            'learning_rate': learning_rate,
            'weight_decay': 0.,
            'temperature': temperature
        }

        # a list of dictionaries which gives a simple way of breaking a model's parameters into separate components   for optimization.
        groups = [{'params' : model.W0, # input to hidden Conv
                   'learning_rate' : learning_rate,
                   'weight_decay' : weight_decay_1,
                   'temperature' : temperature},
                   {'params' : model.W1, # hidden to hidden
                    'learning_rate' : learning_rate,
                    'weight_decay'  : weight_decay_2,
                    'temperature' : temperature},
                   {'params' : model.A, # hidden to linear readout
                    'learning_rate' : learning_rate,
                    'weight_decay'  : weight_decay_3,
                    'temperature' : temperature},
                 ]
        super(LangevinSimple2, self).__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        for group in self.param_groups:

            learning_rate = group['learning_rate']
            weight_decay = group['weight_decay']
            temperature = group['temperature']
            for parameter in group['params']:
                eta = torch.randn_like(parameter)
                d_p = eta * (2*learning_rate*temperature)**0.5
                d_p = d_p.to(DEVICE)

                d_p.add_(parameter, alpha=-learning_rate*weight_decay)

                if on_data is True and parameter.grad is not None:
                    # The factor of 0.5 has to do with the normalization of the
                    # loss
                    d_p.add_(parameter.grad, alpha=-learning_rate)
                parameter.add_(d_p)


#--------------------------------------------------------------------------------------------
# Data functions

def get_x_conv(X):
    d = X.shape[2]
    w_star = torch.zeros(d)
    w_star[0]=1.
    W = w_star.repeat((1,1))
    return X[:,0,:]@W.T


def cubic_nonlin(X):
    H1 = get_x_conv(X)
    Y = (H1**3-3*H1)
    return Y

def target(X,eps):
    Y = get_x_conv(X)+cubic_nonlin(X)*eps
    Y = Y.unsqueeze(1)
    return Y



def get_data(d,n,seed,target_func):
    np.random.seed(seed)
    X = torch.tensor(np.random.normal(loc=0,scale=1.,size=(n,1,d))).to(dtype=DTYPE)
    return X, target_func(X)

def get_train_test_data(d,n,n_test,train_seed,test_seed,target_func):
    X_train, Y_train = get_data(d,n,train_seed,target_func)
    X_test, Y_test = get_data(d,n_test,test_seed,target_func)
    return X_train, Y_train, X_test, Y_test



def prep_train_test(d,n,n_test,train_seed,test_seed,eps):
    my_target = lambda X: target(X,eps)
    X_train, Y_train, X_test, Y_test =  get_train_test_data(d,n,n_test,train_seed,test_seed,my_target)
    train_data = torch.utils.data.TensorDataset(X_train.to(dtype=DTYPE), Y_train.to(dtype=DTYPE))
    test_data = torch.utils.data.TensorDataset(X_test.to(dtype=DTYPE), Y_test.to(dtype=DTYPE))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=n, shuffle=False, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=n_test, shuffle=False, num_workers=1)
    return X_train,X_test,Y_train,Y_test,train_loader,test_loader



def calc_run_time(start, end):
    seconds = int(end - start)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    run_time = '{:d}:{:02d}:{:02d}'.format(h, m, s)
    #print('run_time =', run_time)
    return run_time


#---------------------------------------------------------------------------------------------------------

class FCN3NetworkEnsembleLinear(nn.Module):

    def __init__(self, d, n1, n2,ensembles=1, weight_initialization_variance=(1.0, 1.0, 1.0)):
        super().__init__()

        self.arch = [d, n1, n2]
        self.d = d
        self.n1 = n1
        self.n2 = n2
        self.W0 = nn.Parameter(torch.normal(mean=0.0, 
                                            std=torch.full((ensembles, n1, d), weight_initialization_variance[0]**0.5)).to(DEVICE),
                                            requires_grad=True) # requires_grad moved here
        self.W1 = nn.Parameter(torch.normal(mean=0.0, 
                                            std=torch.full((ensembles, n2, n1), weight_initialization_variance[1]**0.5)).to(DEVICE),
                                            requires_grad=True) # requires_grad moved here
        self.A = nn.Parameter(torch.normal(mean=0.0, 
                                           std=torch.full((ensembles, n2), weight_initialization_variance[2]**0.5)).to(DEVICE),
                                           requires_grad=True) # requires_grad moved here


    def h1_activation(self, X):
        return contract(
            'ijk,ikl,unl->uij',
            self.W1, self.W0, X,
            backend='torch'
        )

    def h0_activation(self, X):
        return contract(
            'ikl,unl->uik',
            self.W0, X,
            backend='torch'
        )


    def forward(self, X):
        """

        Efficiently computes the outputs of a three layer network
        using opt_einsum

        f : P*d -> P*e*1
        C1_ui = W1_ijk*x_uk
        C2_uij = W2_ijk*C1_uik
        C3_ui = A_ij*C2_uij
        """
        A = self.A
        W1 = self.W1
        W0 = self.W0

        return contract(
            'ij,ijk,ikl,unl->ui',
            A, W1, W0, X,
          backend='torch'
        )                                            
#--------------------------------------------------------------------------------------------------------
# Class specific for the Lengevin network training



class MyNetwork():
    #sigmas should all be of order one, and then divided by the MF scaling
    def __init__(self, input_dim, num_channels,n_train,n_test,max_epochs,sigma_2,sigma_a2,sigma_h2,sigma_w2,FL_scale,eps,seeds,activation,save_path,lr0, chi_param):
        [self.train_seed, self.test_seed] = seeds

        self.d,self.N,self.n,self.n_test = int(input_dim),int(num_channels),int(n_train),int(n_test)

        self.activation = activation
        self.net = FCN3NetworkEnsembleLinear(
                self.d,
                self.N,
                self.N,
                weight_initialization_variance=(1.0/self.d,1.0/self.N,1.0/(self.N*chi_param)),
                ensembles=ens).to(DEVICE)
        self.chi = chi_param # Use the chi_param passed in

        self.lr = lr0/self.n
        self.max_epochs = max_epochs
        self.sa2,self.sh2,self.sw2,self.s2 = sigma_a2 ,sigma_h2,sigma_w2,sigma_2
        #print(self.sa2,self.sw2,self.s2)
        self.save_path = save_path
        self.eps,self.FL_scale = eps,FL_scale

        #To calculate the wd terms I used the term appearing after eqn. 2 in the paper: Predicting the outputs of finite deep neural networks...
        self.temperature = 2.0/(self.chi)
        self.wd_input = self.temperature*self.d
        self.wd_preactivation = self.temperature*self.N
        self.wd_output = self.temperature*self.N*self.chi

        self.X_train,self.X_test,self.Y_train,self.Y_test,self.train_loader,self.test_loader = prep_train_test(self.d,self.n,self.n_test,self.train_seed,self.test_seed,eps)


        for i, data in enumerate(self.train_loader, 0): # for Langevin training we use full-batch
            inputs, labels = data
            self.inputs, self.labels = inputs.to(DEVICE), labels.to(DEVICE)

        for i_test, data_test in enumerate(self.test_loader, 0):
            inputs_test, labels_test = data_test
            self.inputs_test, self.labels_test = inputs_test.to(DEVICE), labels_test.to(DEVICE)

        self.num_saved = 0




    def get_filename(self):
        filename =  "D_"+str(self.d)+"_N_"+str(self.N)+"_P_"+str(self.n)+"_FL_"+str(self.FL_scale)+"_s2_"+str(self.s2)+"_sa2_"+str(self.sa2)+"_sw2_"+str(self.sw2)
        return filename+".csv"


    def net_save_output_only(self):
        """
        This function saves the test and train outputs of the network to two different files.
        The rows in these files represent the number of nets, and the columns the number of data points.
        """
        filename = "outputs_"+self.get_filename()
        fs = self.net(self.X_train.to(DEVICE)).detach().cpu().numpy().flatten()
        filename_test = "test_outputs_"+self.get_filename()
        fs_test = self.net(self.X_test.to(DEVICE)).detach().cpu().numpy().flatten()
        fs, fs_test = np.append(fs,self.train_seed),np.append(fs_test,self.test_seed)

        with open(self.save_path+filename, 'a') as f:
            write = csv.writer(f)
            write.writerow(fs)
            self.num_saved += 1
        with open(self.save_path+filename_test, 'a') as f:
            write = csv.writer(f)
            write.writerow(fs_test)




    def one_epoch(self,epoch,optimizer):
        self.net.train()
        running_loss, train_correct, train_total = 0.0, 0, 0
        optimizer.zero_grad()
        # if ((np.mod(epoch, 1000)==0 and epoch < 200000) or (np.mod(epoch, 15000)==0 and epoch > 200000)) :
        #     fs = self.net(self.X_test.to(DEVICE)).detach().cpu().numpy().flatten()

            #calculate train loss

        outputs = self.net(self.inputs).unsqueeze(1)
        loss = criterion(outputs, self.labels)

            #calculate test loss
        if on_data is True:
            outputs_test_full = self.net(self.inputs_test).unsqueeze(1)
            loss_test = criterion(outputs_test_full, self.labels_test)
            loss.backward()
        optimizer.step()

        return loss.item()




    def train_net(self):
        with SummaryWriter(SAVE_PATH) as writer:
            start = time.time()
            optimizer = LangevinSimple2(self.net, self.lr, self.wd_input, self.wd_preactivation, self.wd_output, self.temperature)

            # Initialize tqdm progress bar
            with tqdm(total=self.max_epochs, desc=f"Training Net {ens}") as pbar:
                for epoch in range(self.max_epochs):
                    # Update learning rate according to schedule
                    if lr_schedule is True:
                        lr = lr0 * np.exp((-1.0 / args.rate_decay) * epoch)
                        self.lr = lr
                        print(f"Learning rate: {self.lr}, epoch: {epoch}")
                        for param_group in optimizer.param_groups:
                            param_group['learning_rate'] = lr
             
                    loss = self.one_epoch(epoch,optimizer)
                    optimizer =  LangevinSimple2(self.net, self.lr, self.wd_input,self.wd_preactivation, self.wd_output, self.temperature)
                    if epoch % 1000 == 0: # Ensure update happens every 1000 epochs or so
                        pbar.set_postfix(loss=f'{loss:.4f}, epoch:{epoch}')
                        pbar.update(1000 if epoch + 1000 <= self.max_epochs else self.max_epochs - epoch)
                    if epoch % CASH_FREAK == 0:
                        if dev_mode is False:
                            torch.save(self.net, os.path.join(SAVE_PATH, f'netnum_{ens}'))
                        epoch_callback_fcn3(epoch, loss, writer, self.net, self.inputs.squeeze(1), ens)

            if dev_mode is False:
                torch.save(self.net, os.path.join(SAVE_PATH, f'netnum_{ens}'))
                epoch_callback_fcn3(epoch, loss, writer, self.net, self.inputs.squeeze(1), ens)
#           self.net_save_output_only()
            end_epoch = time.time()
            run_time = calc_run_time(start, end_epoch)
            return


# This loop now runs only once per script execution, for k=0
# The outer script will handle the hyperparameter sweep
torch.seed() # make sure DNN initialization is different every time
np.random.seed()

# Check if the network already exists
net_path = os.path.join(SAVE_PATH, f'netnum_{ens}')
if os.path.exists(net_path):
    print(f"Loading existing network from {net_path}")
    net = MyNetwork(d, N, n_train, n_test, max_epochs, s2, sa2, sh2, sw2, FL_scale, eps, [train_seed, test_seed], activation, SAVE_PATH, lr0, chi)
    net.net = torch.load(net_path)
else:
    print(f"No existing network found at {net_path}. Creating and training a new network.")
    net = MyNetwork(d, N, n_train, n_test, max_epochs, s2, sa2, sh2, sw2, FL_scale, eps, [train_seed, test_seed], activation, SAVE_PATH, lr0, chi)
    net.train_net() # k=0 since we are running one net at a time
print(f"done with N:{N}, D:{d}, P:{n_train}")
