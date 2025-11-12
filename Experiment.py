DATA_SEED = 613
MODEL_SEED = 26
LANGEVIN_SEED = 480
from juliacall import Main as jl
import juliacall
import os
import torch
import numpy as np
from dataclasses import dataclass, field
import sys
import scipy.stats as stats

sys.path.insert(0, '/home/akiva/FCNX-Ensembling')
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
from typing import List, Iterable

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


    def large_dataset(self, p_large = 1000, device=DEVICE):
        torch.manual_seed(DATA_SEED)
        Xinf = torch.randn((p_large, self.d), dtype=torch.float64).to(device)
        z = Xinf[:,:]
        He1 = z
        He3 = z**3 - 3.0 * z
        Yinf = (He1, He3)
        return Xinf,Yinf

    def __post_init__(self):
        self.device = DEVICE
        self.X = torch.randn((self.P, self.d), dtype=torch.float64)
        z = self.X[:,0]
        self.He1 = z
        self.He3 = z**3 - 3.0 * z
        self.Y = (self.He1 + self.eps * self.He3).unsqueeze(-1)
        self.model = self.networkWithDefaults()

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
        self.model.load_state_dict(state_dict)
        print(f"Loaded model state_dict from {load_model_filename2}")
        self.jl = None
        if compute_predictions:
            jl.include('/home/akiva/FCNX-Ensembling/3layerexps/erfeigensolvers/FCS.jl')

        self.predictions = self.eig_predictions() if compute_predictions else None

    
    def eig_predictions(self):
        try: 
            d = float(self.d)
            i0 = [1/d, 1/d**3, 1/d, 1/d**3]
            i0 = juliacall.convert( jl.Vector[jl.Float64], i0)
            χ = self.chi
            d = self.d
            n = self.N
            ϵ = self.eps
            π = np.pi
            δ = 1.0
            P = self.P
            lr =1e-6
            Tf = 6000000

            lT = jl.FCS.nlsolve_solver(
                i0,
                chi=χ, d=d, kappa=1.0, delta=δ,
                epsilon=ϵ, n=n, b=4 / (3 * π),
                P=P, lr=lr, max_iter=Tf, verbose=True, anneal=True
            )

            lP  = jl.FCS.nlsolve_solver(
                i0,
                chi=χ, d=d, kappa=1.0, delta=0.0,
                epsilon=ϵ, n=n, b=4 / (3 * π),
                P=P, lr=lr, max_iter=Tf, verbose=True, anneal=True
            )

            return Eigenvalues(*lT, *lP)
        except Exception as e:
            print(e)
