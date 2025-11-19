# Trying to understand why things aren't working in cubic erf
import torch
import sys
sys.path.insert(0, '/home/akiva/FCNX-Ensembling')

from Experiment import Experiment
experimentMF = Experiment(
    file='/home/akiva/exp/fcn3erf/erf_cubic_P_250_D_20_N_120_epochs_30000000_lrA_1.33e-07_time_20251116_210435',
    device=torch.device('cuda:1'),
    P=250,
    d=20,
    N=120,
    chi=60,
    eps=0.4,
    ens=10)

experimentMF = Experiment(
    file='/home/akiva/exp/fcn3erf/erf_cubic_P_1000_D_40_N_400_epochs_25000000_lrA_3.33e-07_time_20251117_011312',
    device=torch.device('cuda:1'),
    P=1000,
    d=40,
    N=400,
    chi=100,
    ens=2,
    eps=3.0)

experimentMF = Experiment(
    file='/home/akiva/exp/fcn3erf/erf_cubic_eps_0.3_P_2500_D_40_N_400_epochs_25000000_lrA_1.33e-07_time_20251117_132840/',
    P=1000,
    d=40,
    N=400,
    chi=100,
    eps=0.3,
    ens=3,
    device=torch.device('cuda:1')
)

# experimentMF.load()
experimentMF.load(compute_predictions=False)
# experimentMF.predictions = experimentMF.eig_predictions()
model = experimentMF.model
experimentMF.model.device = experimentMF.device
experimentMF.model.to(experimentMF.device)
# print("Predictions: ")
# print(experimentMF.predictions)
X, _ = experimentMF.large_dataset(p_large = 30_000, device = experimentMF.device)

ls = experimentMF.model.H_eig_random_svd(X, k =100, p =25 )
# experimentMF.predictions = experimentMF.eig_predictions()

# print("Predicted leading eigenvalues:")
# print(f"LH1P: {experimentMF.predictions.lH1P:0.8f}")
# print(f"LH3P: {experimentMF.predictions.lH3P:0.8f}")            
breakpoint()
torch.set_printoptions(precision=8)
print(ls[:1000].sorted(descending=True).values)
# print("Leading order eigenvalues:")
# print(ls[(ls > 1e-2) & (ls < 5e-1)])
# print("Second order eigenvalues")

# print(ls[(ls > 1e-3) & (ls < 1e-2)])
# print("Third order eigenvalues")
# print(ls[(ls > 1e-4) & (ls < 1e-3)])
# print("Fourth order eigenvalues")
# print(ls[(ls > 1e-5) & (ls < 1e-4)])

# print("All eigenvalues")
# # for i in range(ls.shape[0]):
# #     print(ls[i])
