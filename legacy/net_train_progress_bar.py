import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

#---------------------------------------------------------------------------------------------------
# General Settings


INIT_SEED = 222
DTYPE=torch.float32
criterion = nn.MSELoss(reduction="sum")
train_seed, test_seed = 563,10
SAVE_PATH = "/home/akiva/nettrain/network_ensemble"
max_epochs = 300000
lr0 = 1e-3 
n_train,N,d = 300,100,50
n_test = 200
chi = 1
sw2, sa2, s2 = 1.0, 1.0, 1.0
FL_scale = 1.0 # float(128)
activation = "lin"
eps = 0.
on_data = False
# Create a unique timestamp string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Append the timestamp to the SAVE_PATH
SAVE_PATH = f"{SAVE_PATH}_{timestamp}_SUBTRACT_MEAN_H"
SAVE_PATH = f"{SAVE_PATH}_N:{N}_d:{d}_chi:{chi}_k:{s2}_ondata_{on_data}"
print(SAVE_PATH)
NUM_NETS = 1
import shutil

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
else:
    shutil.rmtree(SAVE_PATH)
    os.makedirs(SAVE_PATH)
#----------------------------------------------------------------------------------------------------
# Training and Network settings


from Covariance import *



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
hidden_width: int = 200
chi = hidden_width
kappa = 1.0  / chi
temperature = 2 * kappa
torch.manual_seed(INIT_SEED)


CASH_FREAK = 5000
def epoch_callback_fcn2(epoch, loss, writer, model, data, model_identifier):
    if epoch % CASH_FREAK !=0: return
    with torch.no_grad():
        writer.add_scalar(f'FCN2_{model_identifier}/Train_Step',
                                loss, epoch)
        # breakpoint() # Removed breakpoint
        H = compute_avg_channel_covariance(model, data, layer_name='lin1')
        log_matrix_to_tensorboard(writer,f'FCN2_{model_identifier}/H', H.cpu().numpy(),epoch)
        X = data
        d = X.shape[-1] # Get d from X's last dimension

        w = torch.eye(d).to(hp2.DEVICE) # This is your d*d identity matrix

        y = X @ w # Matrix multiplication

        lH = project_onto_target_functions(H,y )
        lK = transform_eigenvalues(lH, 1.0, chi, n_train)
        # print(lH.shape) # Removed print statement
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

        #firstpairs = {k: scalarsH[k] for k in list(scalarsH)} # Not used

        writer.add_scalars(f'FCN2_{model_identifier}/eig_lH',scalarsH,epoch)
        writer.add_scalars(f'FCN2_{model_identifier}/eig_lK',scalarsK,epoch)

    if epoch % CASH_FREAK != 0: return
    with torch.no_grad():
        W = model.lin1.weight.detach().T
        cov_W = torch.einsum('aj,bj->ab', W, W) /      N
        cov_W = cov_W.cpu().numpy()

        # Create a Matplotlib figure
        fig,( ax1,ax2)= plt.subplots(2,1, figsize=(10, 20)) # Adjust size as needed

        # Determine title and axis labels based on rowvar
        title = f'Covariance of Output Features (Step {epoch})'
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
        writer.add_image(  f'FCN2_{model_identifier}/fc2.Weights_Cov', image_tensor_chw, global_step=epoch)

    writer.flush()

#-----------------------------------------------------------------------------------------------------
# Lengevin Optimizer

class LangevinSimple2(optim.Optimizer):


    def __init__(self, model: nn.Module, learning_rate, weight_decay_1, weight_decay_2, temperature):

        defaults = {
            'learning_rate': learning_rate,
            'weight_decay': 0.,
            'temperature': temperature
        }

        # a list of dictionaries which gives a simple way of breaking a model’s parameters into separate components   for optimization.
        groups = [{'params' : list(model.modules())[1].parameters(), # input to hidden Conv
                   'learning_rate' : learning_rate,
                   'weight_decay' : weight_decay_1,
                   'temperature' : temperature},
                   {'params' : list(model.modules())[2].parameters(), # hidden to linear readout
                    'learning_rate' : learning_rate,
                    'weight_decay'  : weight_decay_2,
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

                if parameter.grad is None:
                    continue

                d_p = torch.randn_like(parameter) * (2*learning_rate*temperature)**0.5
                d_p.add_(parameter, alpha=-learning_rate*weight_decay)
                if on_data is True:
                    # The factor of 0.5 has to do with the normalization of the
                    # loss
                    d_p.add_(parameter.grad, alpha=-0.5*learning_rate)
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

class FCN_2_layers(nn.Module):
      def __init__(self, d,N,init_out,init_hidden,activation, init_seed=None):
        super().__init__()
        if init_seed is not None:
            torch.manual_seed(INIT_SEED)
        self.lin1 = nn.Linear(d,N, bias=False)
        self.lin2 = nn.Linear(N,1, bias=False)
        self.activation=activation
        self.N = N
        self.d = d
        #np.random.seed(5)
        nn.init.normal_(self.lin1.weight,0,(init_hidden)**0.5)
        nn.init.normal_(self.lin2.weight,0,(init_out)**0.5)

      def forward(self, x):
        x = self.lin1(x)

        res = self.lin2(torch.flatten(x, start_dim=1))
        return res


#--------------------------------------------------------------------------------------------------------
# Class specific for the Lengevin network training



class MyNetwork():
    #sigmas should all be of order one, and then divided by the MF scaling
    def __init__(self, input_dim, num_channels,n_train,n_test,max_epochs,sigma_2,sigma_a2,sigma_w2,FL_scale,eps,seeds,activation,save_path,lr0, chi):
        [self.train_seed, self.test_seed] = seeds

        self.d,self.N,self.n,self.n_test = int(input_dim),int(num_channels),int(n_train),int(n_test)

        self.activation = activation
        self.net = FCN_2_layers(self.d,self.N,1.0/(self.N*chi),1.0/self.d,self.activation).to(DEVICE)
        self.chi = chi

        self.lr = lr0/self.n
        self.max_epochs = max_epochs
        self.sa2,self.sw2,self.s2 = sigma_a2 ,sigma_w2,sigma_2
        #print(self.sa2,self.sw2,self.s2)
        self.save_path = save_path
        self.eps,self.FL_scale = eps,FL_scale

        #To calculate the wd terms I used the term appearing after eqn. 2 in the paper: Predicting the outputs of finite deep neural networks...
        self.temperature = 2/(self.chi)
        self.wd_input = self.temperature*self.d
        self.wd_output = self.temperature*(self.N*self.chi)

        self.X_train,self.X_test,self.Y_train,self.Y_test,self.train_loader,self.test_loader = prep_train_test(self.d,self.n,self.n_test,self.train_seed,self.test_seed,eps)


        for i, data in enumerate(self.train_loader, 0): # for Langevin training we use full-batch
            inputs, labels = data
            self.inputs, self.labels = inputs.to(DEVICE), labels.to(DEVICE)

        for i_test, data_test in enumerate(self.test_loader, 0):
            inputs_test, labels_test = data_test
            self.inputs_test, self.labels_test = inputs_test.to(DEVICE), labels_test.to(DEVICE)

        self.num_saved = 0




    def get_filename(self):
        filename =  "S_"+str(self.d)+"_N_"+str(self.N)+"_n_"+str(self.n)+"_FL_"+str(self.FL_scale)+"_s2_"+str(self.s2)+"_sa2_"+str(self.sa2)+"_sw2_"+str(self.sw2)
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
        if ((np.mod(epoch, 1000)==0 and epoch < 200000) or (np.mod(epoch, 15000)==0 and epoch > 200000)) :
            fs = self.net(self.X_test.to(DEVICE)).detach().cpu().numpy().flatten()
            #print(self.Y_test.flatten().T@fs/(self.Y_test.flatten().T@self.Y_test.flatten()))
            #input()

        #calculate train loss
        outputs = self.net(self.inputs).unsqueeze(1)
        loss = criterion(outputs, self.labels)
        #calculate test loss
        outputs_test_full = self.net(self.inputs_test).unsqueeze(1)
        loss_test = criterion(outputs_test_full, self.labels_test)
        loss.backward()
        optimizer.step()

        return loss.item()





    def train_net(self, k):
        with SummaryWriter(SAVE_PATH) as writer:
            start = time.time()
            optimizer = LangevinSimple2(self.net, self.lr, self.wd_input, self.wd_output, self.temperature)

            # Initialize tqdm progress bar
            with tqdm(total=self.max_epochs, desc=f"Training Net {k}") as pbar:
                for epoch in range(self.max_epochs):
                    loss = self.one_epoch(epoch,optimizer)
                    optimizer =  LangevinSimple2(self.net, self.lr, self.wd_input,self.wd_output, self.temperature)
                    if epoch % 1000:
                        pbar.set_postfix(loss=f'{loss:.4f}, epoch:{epoch}')
                        pbar.update(1000 if epoch + 1000 <= self.max_epochs else self.max_epochs - epoch) # Update by 100 or remaining

                    if epoch % CASH_FREAK == 0:
                        torch.save(self.net, os.path.join(SAVE_PATH, f'netnum_{k}'))
                        epoch_callback_fcn2(epoch, loss, writer, self.net, self.inputs.squeeze(1), str(k))

            self.net_save_output_only()
            end_epoch = time.time()
            run_time = calc_run_time(start, end_epoch)
            return



#------------------------------------------------------------------------------------------------------------


for k in range(NUM_NETS):
    torch.seed() # make sure DNN initialization is different every time
    np.random.seed()
    net = MyNetwork(d,N,n_train,n_test,max_epochs,s2,sa2,sw2,FL_scale,eps,[train_seed,test_seed],activation,SAVE_PATH,lr0, chi)
    net.train_net(k)
    print("done with ",k)
