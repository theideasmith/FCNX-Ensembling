import torch
import torch.nn as nn
import torch.optim as optim
import pytest


import numpy as np

from lib.DataAveragedNetworks import FCN3NetworkDataAveragedEnsemble
DEVICE = "cpu"
def test_initialization_shapes():
    d, n1, n2, P = 8, 16, 32, 10
    D, Q = 3, 4
    net = FCN3NetworkDataAveragedEnsemble(d, n1, n2, P, num_datasets=D, num_ensembles=Q, device=torch.device(DEVICE))
    assert net.W0.shape == (D, Q, n1, d)
    assert net.W1.shape == (D, Q, n2, n1)
    assert net.A.shape == (D, Q, n2)

def test_forward_output_shape():
    d, n1, n2, P = 5, 7, 9, 12
    D, Q = 2, 3
    net = FCN3NetworkDataAveragedEnsemble(d, n1, n2, P, num_datasets=D, num_ensembles=Q, device=torch.device(DEVICE))
    X = torch.randn(D, P, d)
    out = net(X)
    assert out.shape == (D, P, Q)

def test_loss_independence():
    d, n1, n2, P = 4, 6, 8, 20
    D, Q = 2, 2
    net = FCN3NetworkDataAveragedEnsemble(d, n1, n2, P, num_datasets=D, num_ensembles=Q, device=torch.device(DEVICE))
    X = torch.randn(D,P, d)
    Y = torch.randn(D, P, Q)
    out = net(X)
    mse = nn.MSELoss(reduction='none')
    loss = mse(out, Y).mean(0)  # shape (D, Q)
    # Change Y for one (d,q) and check only that loss changes
    Y2 = Y.clone()
    Y2[:, 0, 1] += 10.0
    out2 = net(X)
    loss2 = mse(out2, Y2).mean(0)
    assert torch.allclose(loss[:, 0], loss2[:, 0])
    assert not torch.allclose(loss[:, 1], loss2[:, 1])

def test_langevin_update_changes_params():
    d, n1, n2, P = 3, 5, 7, 15
    D, Q = 2, 2
    net = FCN3NetworkDataAveragedEnsemble(d, n1, n2, P, num_datasets=D, num_ensembles=Q, device=torch.device(DEVICE))
    X = torch.randn(D, P, d)
    Y = X[:, :, 0].unsqueeze(-1).expand(-1, -1, Q)  # (D, P, Q)
    assert Y.shape == (D, P, Q)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    out = net(X)
    assert out.shape == (D, P, Q)
   
    loss = nn.MSELoss()(out, Y)
    loss.backward()
    # Langevin noise
    for param in net.parameters():
        param.data.add_(0.01 * torch.randn_like(param))
    optimizer.step()
    # Check params changed
    out2 = net(X)
    assert not torch.allclose(out, out2)

def test_multiple_configurations():
    configs = [
        (10, 20, 30, 5, 1, 1),
        (6, 8, 10, 7, 2, 3),
        (4, 4, 8, 12, 3, 2),
    ]
    for d, n1, n2, P, D, Q in configs:
        net = FCN3NetworkDataAveragedEnsemble(d, n1, n2, P, num_datasets=D, num_ensembles=Q, device=torch.device(DEVICE))
        X = torch.randn(D, P, d)
        out = net(X)
        assert out.shape == (D, P, Q)

def arcsin_kernel(X: torch.Tensor) -> torch.Tensor:
    """
    Compute arcsin kernel matrix for batched inputs X (D, N, d).
    Returns (D, N, N): one kernel matrix per dataset.
    """
    D, N, d = X.shape
    XXT = torch.einsum('dni,dmi->dnm', X, X) / d  # (D, N, N)
    diag = torch.sqrt((1 + 2 * XXT.diagonal(dim1=1, dim2=2)).unsqueeze(-1))  # (D, N, 1)
    denom = diag @ diag.transpose(1, 2)  # (D, N, N)
    arg = 2 * XXT / denom
    arg = torch.clamp(arg, -1.0, 1.0)
    return (2 / torch.pi) * torch.arcsin(arg)

def cosine_squared_distance(a, b):
    # a, b: (N,) or (N,1)
    a = a.flatten()
    b = b.flatten()
    cos = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)
    return cos**2

def test_arcsin_kernel_multi_dataset():
    D, N, d = 3, 5, 4
    torch.manual_seed(42)
    X = torch.randn(D, N, d)
    K = arcsin_kernel(X)
    # Check shape
    assert K.shape == (D, N, N)
    # Check symmetry for each dataset
    for i in range(D):
        assert torch.allclose(K[i], K[i].T, atol=1e-6)
    # Check diagonal is positive
    for i in range(D):
        assert torch.all(K[i].diagonal() > 0)
    print("arcsin_kernel multi-dataset test passed, shape:", K.shape)

def test_gpr_prediction_multi_dataset_with_ensemble():
    D, P, Q = 4, 10, 3
    d = 5
    kappa = 1.0
    torch.manual_seed(123)
    # X: (D, P, d), Y: (D, P, Q)
    X = torch.randn(D, P, d)
    Y = torch.randn(D, P, Q)
    # Average over ensemble dimension
    Y_gpr = Y.mean(dim=2)  # (D, P)
    # Kernel: (D, P, P)
    K = arcsin_kernel(X) + kappa * torch.eye(P).unsqueeze(0)
    pred_gpr = torch.zeros_like(Y_gpr.T)  # (P, D)
    for dset in range(D):
        alpha = torch.linalg.solve(K[dset], Y_gpr[dset])
        pred_gpr[:, dset] = K[dset] @ alpha
    # Check shapes
    assert pred_gpr.shape == (P, D)
    # Check prediction is finite
    assert torch.isfinite(pred_gpr).all()
    print("GPR multi-dataset prediction with ensemble test passed, shape:", pred_gpr.shape)

def test_fcn3_vs_gpr_cosine_distance_long_train():
    
    torch.manual_seed(0)
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    P, d = 300, 150
    n1, n2 = 600, 600
    D, Q = 1, 100
    kappa = 1.0
    langevin_T = 2 * kappa
    print("Initializing test with DEVICE =", DEVICE)
    # Generate random data and targets
    X = torch.randn(D, P, d, device=DEVICE)
    Y = X[:, :, 0].unsqueeze(-1).expand(-1, -1, Q)  # (D, P, Q)

    assert Y.shape == (D, P, Q)

    net = FCN3NetworkDataAveragedEnsemble(
        d, n1, n2, P, num_datasets=D, num_ensembles=Q, activation='erf', device=torch.device(DEVICE)
    ).to(DEVICE)

    # Weight decay for each layer: T * input-dim * CHI (CHI=1)
    wd0 = langevin_T * d
    wd1 = langevin_T * n1
    wd2 = langevin_T * n2
    lr = 3e-6
    # Use full batch optimizer (no shuffling, all data at once)
    optimizer = optim.SGD([
        {'params': net.W0, 'weight_decay': wd0},
        {'params': net.W1, 'weight_decay': wd1},
        {'params': net.A,  'weight_decay': wd2},
    ], lr=lr)
    steps = 300000
    for step in range(steps):
        optimizer.zero_grad()
        out = net(X)  # (N, D, Q)
        loss = (out - Y).pow(2).sum()  # sum reduction
        loss.backward()
        # Langevin noise
        for param in net.parameters():
            param.data.add_(np.sqrt(2 * lr * langevin_T) * torch.randn_like(param))
        optimizer.step()
        if step % 5000 == 0:
            print(f"Step {step} / {steps}, loss: {loss.item()}")

    # Average prediction over ensemble index for each dataset
    pred = net(X).mean(dim=2)  # (D, P)

    with torch.no_grad():
        X_cpu = X.cpu()
        Y_cpu = Y.cpu()
        K = arcsin_kernel(X_cpu) + kappa * torch.eye(P).unsqueeze(0)  # (D, P, P)
        assert K.shape == (D, P, P)
        Y_gpr = Y_cpu.mean(dim=2)  # (D, P)
        assert Y_gpr.shape == (D, P)
        pred_gpr = torch.zeros(P, D)  # <-- FIXED SHAPE
        assert pred_gpr.shape == (P, D)
        for dset in range(D):
            alpha = torch.linalg.solve(K[dset], Y_gpr[dset])  # (P,)
            pred_gpr[:, dset] = K[dset] @ alpha  # (P,)

    # Cosine squared distance for each dataset, then average
    cos2s = []
    pred_cpu = pred.cpu()
    assert pred_cpu.shape == (D, P)
    for dset in range(D):
        cos2 = cosine_squared_distance(pred_cpu[dset,:], pred_gpr[:, dset])
        cos2s.append(cos2.item())
    mean_cos2 = np.mean(cos2s)
    print("Mean cosine squared distance (FCN3 vs GPR, averaged over datasets):", mean_cos2)
    assert mean_cos2 > 0.90  # Should be close if correspondence holds
