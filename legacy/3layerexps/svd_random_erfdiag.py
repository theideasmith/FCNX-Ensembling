import sys
import scipy.stats as stats
sys.path.insert(0, '/home/akiva/FCNX-Ensembling')

from Experiment import Experiment 
experimentMF = Experiment(
    eps=0.03,
    file = '/home/akiva/exp/fcn3erf/erf_cubic_P_1600_D_40_N_256_epochs_40000000_lrA_6.25e-11_time_20251030_174053',
    N=256,
    d=40,
    chi=25,
    P=1600,
    ens=10,
)

import torch
# experimentMFSmallD.load()
experimentMF.load(compute_predictions=True)
model = experimentMF.model
experimentMF.model.device = experimentMF.device
experimentMF.model.to(experimentMF.device)
X, _ = experimentMF.large_dataset(p_large = 30_000, device = experimentMF.device)

ls = experimentMF.model.H_eig_random_svd(X)
experimentMF.predictions = experimentMF.eig_predictions()
torch.set_printoptions(precision=8)

print("Leading order eigenvalues:")
print(ls[(ls > 1e-2) & (ls < 1e-1)])
print("Second order eigenvalues")

print(ls[(ls > 1e-5) & (ls < 1e-4)])

import matplotlib.pyplot as plt
import matplotlib_inline
import matplotlib
plt.ion()

matplotlib.use('TkAgg') # or 'Qt5Agg', 'WebAgg' etc.

ls3 = ls[(ls > 1e-5) & (ls < 1e-4)]
plt.title("3rd Order Eigenspectrum")
plt.bar(torch.arange(0, ls3.shape[0]).detach().cpu().numpy(), ls3.detach().cpu().numpy())
plt.show()
