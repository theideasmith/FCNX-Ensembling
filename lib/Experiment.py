DATA_SEED = 613
MODEL_SEED = 26
LANGEVIN_SEED = 480
import warnings

import os
# Set the JULIA_PYTHONCALL_EXE environment variable before importing juliacall

import numpy as np
from dataclasses import dataclass, field
import sys
import scipy.stats as stats
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/akiva/FCNX-Ensembling')
import os
os.environ["PYTHONCALL_JULIA_PYTHON"] = "yes"


import juliacall
from juliacall import Main as jl
import torch
torch.manual_seed(DATA_SEED)
torch.set_default_dtype(torch.float64)
DEVICE = torch.device('cuda:1')

from FCN3Network import FCN3NetworkEnsembleErf

def strip_orig_mod_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_key = k[len("_orig_mod."):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

from dataclasses import dataclass
from typing import List, Iterable, Optional

@dataclass
class Eigenvalues:
    lJ1T: float
    lJ3T: float
    lH1T: float
    lH3T: float
    lJ1P: float
    lJ3P: float
    lH1P: float
    lH3P: float
    # Readout / kernel eigenvalue predictions (optional — may be computed later)
    lK1T: Optional[float] = None
    lK3T: Optional[float] = None
    lK1P: Optional[float] = None
    lK3P: Optional[float] = None
    @classmethod
    def from_list(cls, values: Iterable[float]) -> 'Eigenvalues':
        return cls(
            lJ1T=values[0],
            lJ3T=values[1],
            lH1T=values[2],
            lH3T=values[3],
            lJ1P=values[4],
            lJ3P=values[5],
            lH1P=values[6],
            lH3P=values[7],
        )
    def to_list(self):
        return [
            self.lJ1T,
            self.lJ3T,
            self.lH1T,
            self.lH3T,
            self.lJ1P,
            self.lJ3P,
            self.lH1P,
            self.lH3P,
        ]

@dataclass
class Experiment:
    file: str
    N: int 
    d: int
    chi: float
    P: int
    ens: int
    model: FCN3NetworkEnsembleErf = field(init=False)
    He1: torch.Tensor = field(init=False)
    He3: torch.Tensor = field(init=False)
    device: torch.device = DEVICE
    eps: float = 0.03
    X: torch.Tensor  = field(init=False)


    def large_dataset(self, p_large = 1000, flat=False):
        torch.manual_seed(DATA_SEED)
        Xinf = torch.randn((p_large, self.d), dtype=torch.float64).to(self.device)
        z = Xinf[:,:]
        He1 = z
        He3 = 1/6 * (z**3 - 3.0 * z)
        Yinf = (He1, He3)
        if flat:
            return Xinf, He1, He3
        return Xinf,Yinf

    def __post_init__(self):
        # self.device = DEVICE
        self.lambdas_H = None
        torch.manual_seed(DATA_SEED)
        self.X = torch.randn((self.P, self.d), dtype=torch.float64)
        z = self.X[:,0]
        self.He1 = z
        self.He3 = z**3 - 3.0 * z
        self.Y = (self.He1 + self.eps * self.He3).unsqueeze(-1)
        self.model = self.networkWithDefaults()

        # Ensure the freshly created model lives on the experiment device
        try:
            self.model.to(self.device)
            # also set a `.device` attribute on the model if consumers expect it
            try:
                setattr(self.model, 'device', self.device)
            except Exception:
                pass
        except Exception:
            # best-effort; continue if device move fails
            pass

        jl.include('/home/akiva/FCNX-Ensembling/julia_lib/FCS.jl')
    def networkWithDefaults(self):
        model = FCN3NetworkEnsembleErf(self.d, self.N, self.N,
                        self.P,
                            ens=self.ens,
                            weight_initialization_variance=(1/self.d, 1.0/self.N, 1.0/(self.N*self.chi)))
        return model

    def load(self, compute_predictions = False):
 
        self.model = self.networkWithDefaults()
        self.model.to(self.device)
        load_model_filename2 = os.path.join(self.file, 'model.pth')
        # load_model_filename2 = os.path.join(file, 'model.pth')
        state_dict = torch.load(load_model_filename2, map_location=self.device)
        # Fix keys if needed
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = strip_orig_mod_prefix(state_dict)

        # If the saved state has an `A` parameter whose first dimension
        # indicates ensemble size, and it differs from the pre-initialized
        # `ens`, re-initialize the model to match the saved ensemble size.
        try:
            a_key = None
            for k in state_dict.keys():
                # match keys like 'A', 'module.A', or 'someprefix.A'
                if k == 'A' or k.endswith('.A') or k.split('.')[-1] == 'A':
                    a_key = k
                    break

            if a_key is not None:
                loaded_A = state_dict[a_key]
                if isinstance(loaded_A, torch.Tensor) and loaded_A.dim() >= 1:
                    loaded_ens = int(loaded_A.shape[0])
                    if loaded_ens != int(self.ens):
                        print(f"Reinitializing model: loaded ensemble size {loaded_ens} != preinitialized ens {self.ens}")
                        # update the Experiment ens and rebuild the model
                        self.ens = loaded_ens
                        self.model = self.networkWithDefaults()
                        self.model.to(self.device)
        except Exception as _e:
            # If anything goes wrong here, continue and let load_state_dict try
            print(f"Warning while aligning ensemble size from state_dict: {_e}")

        self.model.load_state_dict(state_dict)

        # After loading weights, ensure the model is on the desired device and
        # that consumers (e.g. H_eig_random_svd) can access `model.device`.
        try:
            self.model.to(self.device)
        except Exception:
            pass
        try:
            setattr(self.model, 'device', self.device)
        except Exception:
            pass
        print(f"Loaded model state_dict from {load_model_filename2}")
        self.jl = None
        if compute_predictions:
            jl.include('/home/akiva/FCNX-Ensembling/julia_lib/FCS.jl')

        self.predictions = self.eig_predictions() if compute_predictions else None

    
    def eig_predictions(self):
        try: 
            d = float(self.d)
            i0 = [1/d ** 0.5, 1/d ** (3/2), 1/d ** 0.5, 1/d ** (3/2)]
            i0 = juliacall.convert( jl.Vector[jl.Float64], i0)
            χ = self.chi
            d = self.d
            n = self.N
            ϵ = self.eps
            π = np.pi
            δ = 1.0
            P = self.P
            lr =1e-6
            Tf = 60_000
            kappa = 1.0
            lT = jl.FCS.nlsolve_solver(
                i0,
                chi=χ, d=d, kappa=kappa, delta=δ,
                epsilon=ϵ, n=n, b=4 / (3 * π),
                P=P, lr=lr, max_iter=Tf, verbose=False, anneal=True
            )

            i0 = [1/d, 1/d**3, 1/d, 1/d**3]
            i0 = juliacall.convert(jl.Vector[jl.Float64], i0)
            lP  = jl.FCS.nlsolve_solver(
                i0,
                chi=χ, d=d, kappa=1.0, delta=0.0,
                epsilon=ϵ, n=n, b=4 / (3 * π),
                P=P, lr=lr, max_iter=Tf, verbose=False, anneal=True
            )

            preds = Eigenvalues(*lT, *lP)

            # Try to compute readout/kernel eigenvalue predictions via Julia FCS
            try:
                lK_T = jl.FCS.compute_lK(lT, P, n,  χ, d, δ , kappa, ϵ, 4/(3*π))
                lK_T_py = [float(x) for x in lK_T]
                if len(lK_T_py) >= 1:
                    preds.lK1T = lK_T_py[0]
                if len(lK_T_py) >= 2:
                    preds.lK3T = lK_T_py[1]
            except Exception as _e:
                print(f"Warning: compute_lK (T) failed: {_e}")

            try:
                lK_P = jl.FCS.compute_lK(lP, P, n,  χ, d, 0.0 , kappa, ϵ, 4/(3*π))
                lK_P_py = [float(x) for x in lK_P]
                if len(lK_P_py) >= 1:
                    preds.lK1P = lK_P_py[0]
                if len(lK_P_py) >= 2:
                    preds.lK3P = lK_P_py[1]
            except Exception as _e:
                print(f"Warning: compute_lK (P) failed: {_e}")

            return preds
        except Exception as e:
            print(e)

    def diagonalize_H(self, X, k=None):
        # Ensure model and input are on the same device before calling SVD routines
        try:
            self.model.to(self.device)
        except Exception:
            pass
        try:
            setattr(self.model, 'device', self.device)
        except Exception:
            pass

        X_dev = X
        try:
            if isinstance(X, torch.Tensor):
                X_dev = X.to(self.device)
        except Exception:
            # If moving fails, fall back to original X
            X_dev = X

        ls = self.model.H_eig_random_svd(X_dev, k=k)
        self.lambdas_H = ls
        return ls

    def plot_spectrum(self):        # ------------------------------------------------------------------
        # 1. Choose colormap
        # ------------------------------------------------------------------
        cmap_name = "viridis"          # or "magma"
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, 4))
        if self.predictions is None:
            warnings.warn("Predictions not computed yet. Computing now...")
            self.predictions = self.eig_predictions()

        if self.lambdas_H is None:
            warnings.warn("Eigenvalues not computed yet. Computing now...")
            self.lambdas_H = self.diagonalize_H(self.X, k=1000)
    
        ls = self.lambdas_H
        MF_eigenvalues = self.predictions
        # ------------------------------------------------------------------
        # 2. Move ls3 to CPU and convert to NumPy early
        # ------------------------------------------------------------------
        ls3 = ls  # assuming ls is your eigenvalue tensor
        ls3_np = ls3.detach().cpu().numpy()

        # ------------------------------------------------------------------
        # 3. Create boolean masks → **convert to NumPy immediately**
        # ------------------------------------------------------------------
        idx_big_np = (ls3_np > 0.1)
        idx_mid_np = (ls3_np <= 0.1) & (ls3_np > 1e-3)
        idx_small_np = (ls3_np <= 1e-4)

        # Count sizes
        n_big = np.sum(idx_big_np)
        n_mid = np.sum(idx_mid_np)
        n_small = np.sum(idx_small_np)

        # ------------------------------------------------------------------
        # 4. Build bar positions (NumPy)
        # ------------------------------------------------------------------
        pos_big = np.arange(0, n_big)
        pos_mid = np.arange(n_big, n_big + n_mid)
        pos_small = np.arange(n_big + n_mid, n_big + n_mid + n_small)

        # ------------------------------------------------------------------
        # 5. Horizontal target lines
        # ------------------------------------------------------------------
        plt.axhline(y=MF_eigenvalues.lH1T,
                    color=colors[0], linestyle='--', label='$\mathbb{E}\;[\lambda^{H1}_T]$')
        plt.axhline(y=MF_eigenvalues.lH1P,
                    color=colors[1], linestyle='-',  label='$\mathbb{E}\;[\lambda^{H1}_P]$')
        plt.axhline(y=MF_eigenvalues.lH3T,
                    color=colors[2], linestyle='--', label='$\mathbb{E}\;[\lambda^{H3}_T]$')
        plt.axhline(y=MF_eigenvalues.lH3P,
                    color=colors[3], linestyle='-',  label='$\mathbb{E}\;[\lambda^{H3}_P]$')

        # ------------------------------------------------------------------
        # 6. Bar plots – **all NumPy**
        # ------------------------------------------------------------------
        if n_big > 0:
            plt.bar(pos_big, ls3_np[idx_big_np],
                    color=colors[0], label='$\lambda^{H1}_T$')

        if n_mid > 0:
            plt.bar(pos_mid, ls3_np[idx_mid_np],
                    color=colors[1], label='$\lambda^{H1}_P$')

        if n_small > 0:
            small_vals = ls3_np[idx_small_np]
            small_pos = pos_small

            plt.bar([small_pos[0]], [small_vals[0]],
                    color=colors[2], label='$\lambda^{H3}_T$')

            if n_small > 1:
                plt.bar(small_pos[1:], small_vals[1:],
                        color=colors[3], label='$\lambda^{H3}_P$')

        # ------------------------------------------------------------------
        # 7. Finalize
        # ------------------------------------------------------------------
        plt.title(f"FCN3-Erf on y = He1 + 0.0 He3 Eigenspectrum")
        plt.yscale('log')

        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Magnitude (log scale)")
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()
