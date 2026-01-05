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
import sys, os
sys.path.append(os.path.dirname(__file__))


import juliacall
from juliacall import Main as jl
import torch
torch.manual_seed(DATA_SEED)
torch.set_default_dtype(torch.float32)
DEVICE = torch.device('cuda:1')

from FCN3Network import FCN3NetworkEnsembleErf, FCN3NetworkEnsembleLinear

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
import json

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
    N: Optional[int] = None
    d: Optional[int] = None
    chi: Optional[float] = None
    P: Optional[int] = None
    ens: Optional[int] = None
    model: FCN3NetworkEnsembleErf = field(init=False)
    He1: torch.Tensor = field(init=False)
    He3: torch.Tensor = field(init=False)
    device: torch.device = DEVICE
    eps: float = 0.03
    kappa: float = 1.0

    X: torch.Tensor  = field(init=False)


    def large_dataset(self, p_large = 1000, flat=False):
        torch.manual_seed(DATA_SEED)
        Xinf = torch.randn((p_large, self.d), dtype=torch.float32).to(self.device)
        z = Xinf[:,:]
        He1 = z
        He3 = 1/6 * (z**3 - 3.0 * z)
        Yinf = (He1, He3)
        if flat:
            return Xinf, He1, He3
        return Xinf,Yinf

    def __post_init__(self):
        # Read parameters from config.json if not provided
        if self.N is None or self.d is None or self.chi is None or self.P is None or self.ens is None:
            config_path = os.path.join(self.file, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                if self.N is None:
                    self.N = config.get('N')
                if self.d is None:
                    self.d = config.get('d')
                if self.chi is None:
                    self.chi = config.get('chi')
                if self.P is None:
                    self.P = config.get('P')
            else:
                raise FileNotFoundError(f"config.json not found at {config_path} and required parameters (N, d, chi, or P) were not provided")
        
        # self.device = DEVICE
        self.lambdas_H = None
        torch.manual_seed(DATA_SEED)
        self.X = torch.randn((self.P, self.d), dtype=torch.float32)
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
            i0 = [1/d ** 0.5, 1/d ** (3/2), 1/d ** 0.5, 1/d ** (3/2), 1/d]
            i0 = juliacall.convert( jl.Vector[jl.Float64], i0)
            χ = self.chi 
            d = self.d
            n = self.N
            n1 = self.N
            n2 = self.N 
            ϵ = self.eps
            π = np.pi
            δ = 1.0
            P = self.P 
            lr =1e-6
            Tf = 60_000
            kappa = self.kappa # My experiment multiplied the noise by a factor of 2, because I defined the loss with 1/2 and also 
            # multiplied k by 2 when I defined temperature in my langevin dynamics. 
            lT = jl.FCS.nlsolve_solver(
                i0,
                chi=χ, d=d, kappa=kappa, delta=δ,
                epsilon=ϵ, n1=n1, n2=n2, b=4 / (3 * π),
                P=P, lr=lr, max_iter=Tf, verbose=False, anneal=True
            )

            i0 = [1/d, 1/d**3, 1/d, 1/d**3, 1/d]
            i0 = juliacall.convert(jl.Vector[jl.Float64], i0)
            lP  = jl.FCS.nlsolve_solver(
                i0,
                chi=χ, d=d, kappa=1.0, delta=0.0,
                epsilon=ϵ, n1=n1, n2=n2, b=4 / (3 * π),
                P=P, lr=lr, max_iter=Tf, verbose=False, anneal=True
            )

            preds = Eigenvalues(*lT, *lP)

            # Try to compute readout/kernel eigenvalue predictions via Julia FCS
            try:
                lK_T = jl.FCS.compute_lK(lT, P, n1, n2,  χ, d, δ , kappa, ϵ, 4/(3*π))
                lK_T_py = [float(x) for x in lK_T]
                if len(lK_T_py) >= 1:
                    preds.lK1T = lK_T_py[0]
                if len(lK_T_py) >= 2:
                    preds.lK3T = lK_T_py[1]
            except Exception as _e:
                print(f"Warning: compute_lK (T) failed: {_e}")

            try:
                lK_P = jl.FCS.compute_lK(lP, P, n1, n2,  χ, d, 0.0 , kappa, ϵ, 4/(3*π))
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


# ============================================================================
# ExperimentLinear — Subclass for Linear Activation Networks
# ============================================================================

@dataclass
class EigenvaluesLinear:
    """Eigenvalues for linear networks (only He1, no He3)"""
    lJT: float
    lHT: float
    lJP: float
    lHP: float
    # Readout / kernel eigenvalue predictions (optional)
    lKT: Optional[float] = None
    lKP: Optional[float] = None
    
    @classmethod
    def from_list(cls, values: Iterable[float]) -> 'EigenvaluesLinear':
        return cls(
            lJT=values[0],
            lHT=values[1],
            lJP=values[2],
            lHP=values[3],
        )
    
    def to_list(self):
        return [self.lJT, self.lHT, self.lJP, self.lHP]


@dataclass
class ExperimentLinear(Experiment):
    """Experiment subclass for linear activation networks.
    
    Inherits most functionality from Experiment but uses:
    - FCN3NetworkEnsembleLinear model
    - FCSLinear.jl for predictions
    - Only He1 targets (no He3)
    """
    
    model: FCN3NetworkEnsembleLinear = field(init=False)
    
    def __post_init__(self):
        # Read parameters from config.json if not provided
        if self.N is None or self.d is None or self.chi is None or self.P is None or self.ens is None:
            config_path = os.path.join(self.file, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    
                    config = json.load(f)
                if self.N is None:
                    self.N = config.get('N')
                if self.d is None:
                    self.d = config.get('d')
                if self.chi is None:
                    self.chi = config.get('chi')
                if self.P is None:
                    self.P = config.get('P')
                if self.ens is None:
                    self.ens = config.get('ens', 1)
            else:
                raise FileNotFoundError(f"config.json not found at {config_path} and required parameters (N, d, chi, or P) were not provided")
            
        self.lambdas_H = None
        torch.manual_seed(DATA_SEED)
        self.X = torch.randn((self.P, self.d), dtype=torch.float32)
        z = self.X[:, 0]
        self.He1 = z
        # Linear networks: only He1 target, no He3
        self.He3 = None  # Not used in linear networks
        self.Y = self.He1.unsqueeze(-1)  # Only He1, no eps mixing
        self.model = self.networkWithDefaults()

        # Ensure the freshly created model lives on the experiment device
        try:
            self.model.to(self.device)
            try:
                setattr(self.model, 'device', self.device)
            except Exception:
                pass
        except Exception:
            pass

        jl.include('/home/akiva/FCNX-Ensembling/julia_lib/FCSLinear.jl')
    def load(self):
        self.model.load_state_dict(torch.load(os.path.join(self.file, 'model.pth'), map_location=self.device))
    def networkWithDefaults(self):
        """Create linear network with proper initialization"""
        model = FCN3NetworkEnsembleLinear(
            self.d, self.N, self.N, self.P,
            ensembles=self.ens,
            weight_initialization_variance=(1/self.d, 1.0/self.N, 1.0/(self.N*self.chi))
        )
        return model
    
    def large_dataset(self, p_large=1000, flat=False):
        """Generate large dataset for linear network (only He1)"""
        torch.manual_seed(DATA_SEED)
        Xinf = torch.randn((p_large, self.d), dtype=torch.float32).to(self.device)
        z = Xinf[:, 0]
        He1 = z
        # No He3 for linear networks
        Yinf = He1
        if flat:
            return Xinf, He1, None
        return Xinf, Yinf
    
    def eig_predictions(self):
        """Compute eigenvalue predictions using FCSLinear.jl"""
        try: 
            d = float(self.d)
            # Initial guess: [lJ, lH] for training regime
            i0 = [1/d ** 0.5, 1/d ** 0.5]
            i0 = juliacall.convert(jl.Vector[jl.Float64], i0)
            
            χ = self.chi 
            d = self.d
            n1 = self.N
            n2 = self.N 
            δ = 1.0
            P = self.P 
            lr = 1e-6
            Tf = 60_000
            kappa = self.kappa
            
            # Solve for training regime (delta=1.0)
            lT = jl.FCSLinear.nlsolve_solver(
                i0,
                chi=χ, d=d, kappa=kappa, delta=δ,
                n1=n1, n2=n2,
                P=P, lr=lr, max_iter=Tf, verbose=False, anneal=True, anneal_steps=30_000
            )

            # Initial guess for population regime
            i0 = [1/d, 1/d]
            i0 = juliacall.convert(jl.Vector[jl.Float64], i0)
            
            # Solve for population regime (delta=0.0)
            lP = jl.FCSLinear.nlsolve_solver(
                i0,
                chi=χ, d=d, kappa=1.0, delta=0.0,
                n1=n1, n2=n2,
                P=P, lr=lr, max_iter=Tf, verbose=False, anneal=True, anneal_steps=30_000
            )

            preds = EigenvaluesLinear(*lT, *lP)

            # Try to compute readout/kernel eigenvalue predictions
            try:
                lK_T = jl.FCSLinear.compute_lK(lT, P, n1, n2, χ, d, δ, kappa)
                lK_T_py = [float(x) for x in lK_T]
                if len(lK_T_py) >= 1:
                    preds.lKT = lK_T_py[0]
            except Exception as _e:
                print(f"Warning: compute_lK (T) failed: {_e}")

            try:
                lK_P = jl.FCSLinear.compute_lK(lP, P, n1, n2, χ, d, 0.0, kappa)
                lK_P_py = [float(x) for x in lK_P]
                if len(lK_P_py) >= 1:
                    preds.lKP = lK_P_py[0]
            except Exception as _e:
                print(f"Warning: compute_lK (P) failed: {_e}")

            return preds
        except Exception as e:
            print(f"Error computing linear network predictions: {e}")
            return None
    
    def plot_spectrum(self):
        """Plot eigenvalue spectrum for linear network (only He1)"""
        cmap_name = "viridis"
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, 2))  # Only 2 colors needed (He1 T and P)
        
        if self.predictions is None:
            warnings.warn("Predictions not computed yet. Computing now...")
            self.predictions = self.eig_predictions()

        if self.lambdas_H is None:
            warnings.warn("Eigenvalues not computed yet. Computing now...")
            self.lambdas_H = self.diagonalize_H(self.X, k=1000)
    
        ls = self.lambdas_H
        MF_eigenvalues = self.predictions
        ls_np = ls.detach().cpu().numpy()

        # For linear networks, we typically have fewer eigenvalues
        # Just plot them all in order
        n_eigs = len(ls_np)
        pos = np.arange(n_eigs)

        # Plot horizontal target lines
        plt.axhline(y=MF_eigenvalues.lHT,
                    color=colors[0], linestyle='--', label=r'$\mathbb{E}[\lambda^{H}_T]$')
        plt.axhline(y=MF_eigenvalues.lHP,
                    color=colors[1], linestyle='-', label=r'$\mathbb{E}[\lambda^{H}_P]$')

        # Bar plot
        plt.bar(pos, ls_np, color=colors[0], alpha=0.7, label='Empirical $\lambda^H$')

        plt.title(f"FCN3-Linear on y = He1 Eigenspectrum")
        plt.yscale('log')
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Magnitude (log scale)")
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()
