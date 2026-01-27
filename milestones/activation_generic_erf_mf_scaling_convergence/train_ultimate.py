#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import time
import math

class OptimizedEnsembleFCN(nn.Module):
    def __init__(self, d, n1, n2, ens, device, weight_vars):
        super().__init__()
        self.ens, self.n1, self.n2, self.d = ens, n1, n2, d
        v0, v1, v2 = weight_vars
        self.W0 = nn.Parameter(torch.randn(ens, n1, d, device=device) * (v0**0.5))
        self.W1 = nn.Parameter(torch.randn(ens, n2, n1, device=device) * (v1**0.5))
        self.A  = nn.Parameter(torch.randn(ens, n2, device=device) * (v2**0.5))
        
    def forward(self, X_b):
        h0 = torch.erf(torch.bmm(self.W0, X_b))
        h1 = torch.erf(torch.bmm(self.W1, h0))
        return torch.bmm(self.A.unsqueeze(1), h1).squeeze(1).t()

def benchmark_ultimate(d, P, N, chi, kappa, lr, epochs, device_str, ens=50, log_interval=1000):
    device = torch.device(device_str)
    temperature = 2 * kappa / chi
    lr_normalized = lr / P
    noise_scale = math.sqrt(2.0 * lr_normalized * temperature)
    
    # 1. Setup Static Buffers
    X_raw = torch.randn(P, d, device=device)
    static_X = X_raw.t().unsqueeze(0).expand(ens, -1, -1).contiguous()
    X0 = X_raw[:, 0].unsqueeze(-1)
    static_Y = (X0 + 0.03 * (X0**3 - 3 * X0)).contiguous()

    # 2. Model & Foreach Setup
    weight_vars = (1/d, 1/N, 1/(N * chi))
    model = OptimizedEnsembleFCN(d, N, N, ens, device, weight_vars).to(device)
    params = [model.W0, model.W1, model.A]
    wd_scalars = [1.0 - lr_normalized * w for w in [d*temperature, N*temperature, N*temperature*chi]]

    # 3. Capture Graph
    print(f"--- Benchmarking on {device} (N={N}, P={P}, Ens={ens}) ---")
    print("Capturing Graph...")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(20):
            model.zero_grad(set_to_none=False)
            loss = torch.sum((model(static_X) - static_Y)**2)
            loss.backward()
    torch.cuda.current_stream().wait_stream(s)
    
    grads = [p.grad for p in params]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(static_X)
        static_loss = torch.sum((out - static_Y)**2)
        static_loss.backward()

    # 4. Warmup (Run 500 epochs to stabilize GPU boost clocks)
    print("Warming up GPU...")
    for _ in range(500):
        g.replay()
        with torch.no_grad():
            torch._foreach_add_(params, grads, alpha=-lr_normalized)
            torch._foreach_mul_(params, wd_scalars)
            for p in params: p.add_(torch.randn_like(p), alpha=noise_scale)
        model.zero_grad(set_to_none=False)
    torch.cuda.synchronize()

    # 5. Benchmark Loop
    print(f"{'Epoch':>10} | {'Time/Epoch':>12} | {'Loss':>12}")
    print("-" * 45)
    
    total_start = time.perf_counter()
    interval_start = total_start
    
    for epoch in range(1, epochs + 1):
        g.replay()
        with torch.no_grad():
            torch._foreach_add_(params, grads, alpha=-lr_normalized)
            torch._foreach_mul_(params, wd_scalars)
            for p in params: p.add_(torch.randn_like(p), alpha=noise_scale)
        model.zero_grad(set_to_none=False)

        if epoch % log_interval == 0:
            torch.cuda.synchronize() # Only sync when logging
            now = time.perf_counter()
            mspf = ((now - interval_start) / log_interval) * 1000
            loss_val = static_loss.item() / (ens * P)
            print(f"{epoch:10d} | {mspf:8.4f} ms | {loss_val:.4e}")
            interval_start = now

    torch.cuda.synchronize()
    final_avg = ((time.perf_counter() - total_start) / epochs) * 1000
    print("-" * 45)
    print(f"FINAL AVERAGE: {final_avg:.4f} ms/epoch")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10000, help="Total benchmark epochs")
    parser.add_argument('--interval', type=int, default=1000, help="Logging frequency")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()
    
    benchmark_ultimate(d=100, P=1200, N=800, chi=80, kappa=0.0125, lr=3e-5, 
                       epochs=args.epochs, device_str=args.device, log_interval=args.interval)