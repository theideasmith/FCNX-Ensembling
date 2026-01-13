#!/usr/bin/env python3
"""
Quick test for FCN2NetworkDataAveragedEnsemble and its H_eig_data_averaged function.
"""
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from DataAveragedNetworks import FCN2NetworkDataAveragedEnsemble

def test_data_averaged_network():
    d = 3
    P = 8
    N = 5
    D = 2  # num_datasets
    Q = 2  # num_ensembles
    X = torch.randn(P, d)
    model = FCN2NetworkDataAveragedEnsemble(
        d=d, n1=N, P=P, num_datasets=D, num_ensembles=Q,
        activation="erf", weight_initialization_variance=(1/d, 1/N), device="cpu"
    )
    # Forward shape
    out = model(X)
    assert out.shape == (P, D, Q), f"Output shape {out.shape} != (P, {D}, {Q})"
    # H_eig_data_averaged shape: use Y = identity for all P eigenfunctions
    Y = torch.eye(P)
    eigs = model.H_eig_data_averaged(X, Y)
    assert eigs.shape[0] == P, f"H_eig_data_averaged returned shape {eigs.shape}"
    print("Test passed: output shape and H_eig_data_averaged shape correct.")
    print("Sample eigenvalues:", eigs[:5].cpu().numpy())

if __name__ == "__main__":
    test_data_averaged_network()
