import torch

# ...existing code...

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            # Train set
            X_train = torch.randn(P, d, dtype=torch.double, device=device)
            y_train = torch.randn(P, dtype=torch.double, device=device)
            # Test set
            X_test = torch.randn(n_test, d, dtype=torch.double, device=device)
            y_test = torch.randn(n_test, dtype=torch.double, device=device)
            # Ensemble predictions
            with torch.no_grad():
                f_train = model(X_train)  # shape: (ens, P)
                f_test = model(X_test)    # shape: (ens, n_test)
            # Average over ensemble
            fbar_train = f_train.mean(dim=0)  # shape: (P,)
            fbar_test = f_test.mean(dim=0)    # shape: (n_test,)
            # MSE
            train_mse = torch.mean((fbar_train - y_train) ** 2).item()
            test_mse = torch.mean((fbar_test - y_test) ** 2).item()
            train_mse_seeds.append(train_mse)
            test_mse_seeds.append(test_mse)
            # Bias contribution: <(<fbar>_ensemble - y_target)**2>_dataset_seeds
            bias_train_seeds.append(((fbar_train - y_train).mean().item()) ** 2)
            bias_test_seeds.append(((fbar_test - y_test).mean().item()) ** 2)