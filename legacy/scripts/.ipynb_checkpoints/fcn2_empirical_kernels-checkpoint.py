import sys
import torch
import os
import matplotlib.pyplot as plt
def activation(x):
    return x

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from ensemble_manager import NetworksLoader, EnsembleManager
from json_handler import JsonHandler
import standard_hyperparams_fcn2 as hp2
ABS_MENAGERIE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Menagerie'))

def update_dict_paths(config_dict):
    """
    Recursively updates paths in keys of a single dictionary.
    """
    updated_dict = {}
    for key, value in config_dict.items():

        new_value = value
        # If the key is a string and starts with './Menagerie', replace it.
        if isinstance(value, str) and value.startswith('./Menagerie'):
            new_value = value.replace('./Menagerie', ABS_MENAGERIE_ROOT)

        # If the value is a dictionary, recursively call update_dict_paths on it.
        if isinstance(value, dict):
            updated_dict[key] = update_dict_paths(value)
        else:
            updated_dict[key] = new_value
    return updated_dict


if __name__ == '__main__':
    Menagerie_dir = os.path.join(parent_dir, 'Menagerie')
    print(Menagerie_dir)

    # Define the specific ensemble directory you want to load
    ensemble_dir_name = 'ensemble_FCN2_TRAIN_20250523_114716'
#   ensemble_dir_name = 'ensemble_FCN3_TRAIN_20250525_173242'
    ensemble_full_path = os.path.join(Menagerie_dir, ensemble_dir_name)

    # Extract the run_identifier from the ensemble directory name
    run_identifier = ensemble_dir_name.replace('ensemble_', '')

    print(f"Attempting to load ensemble with run_identifier: {run_identifier}")
    print(f"From Menagerie directory: {Menagerie_dir}")
    print(f"Full ensemble path: {ensemble_full_path}")

    # Initialize EnsembleManager.
    # It will load the training_config and manifest if the directory exists.
    # Pass the run_identifier and the base menagerie directory.
    ensemble_manager = EnsembleManager(
        run_identifier=run_identifier,
        menagerie_dir=Menagerie_dir,
        json_handler=JsonHandler() # Pass an instance of JsonHandler
    )
    updated_manifest = []

    for item_dict in ensemble_manager.training_config['manifest']:
        updated_manifest.append(update_dict_paths(item_dict))
    ensemble_manager.training_config['manifest'] = updated_manifest


    # Check if the manifest is populated after initialization
    if not ensemble_manager.training_config.get('manifest'):
        print(f"Error: No manifest found in {ensemble_full_path}. Please ensure the ensemble directory and its 'training_config.json' are correctly set up, or run the dummy data creation section.")
        sys.exit(1)


    config = ensemble_manager.training_config
    # Dimensions for the new tensor
    num_batches_dim1 = config['num_datsets']
    num_batches_dim2 = config['num_ensembles']
    num_samples_per_network = 50
    output_feature_dim = 1 # As raw_Y is P*1, model output should be 1

    # Initialize the tensor to store results: networks x samples x output_dim
    # It's (3, 20, 5, 1) as requested.
    output_tensor = torch.empty(num_batches_dim1, num_batches_dim2, num_samples_per_network, output_feature_dim).to(hp2.DEVICE)

    # Initialize NetworksLoader with the ensemble_manager instance
    networks_loader = NetworksLoader(ensemble_manager)

    print(f"\nStarting to iterate through {len(networks_loader.manifest)} networks...")

    # Iterate through the networks using the NetworksLoader
    loaded_count = 0
    for i, network_info in enumerate(networks_loader):
        loaded_count += 1
        raw_X = network_info.get('data').to(hp2.DEVICE)[:160]
        raw_Y = network_info.get('target').to(hp2.DEVICE)[:160]
        P =raw_X.shape[0]

        n_dat = int((i-i%20)/20)

        nmod = i%20
        random_indices = torch.randperm(P)[:num_samples_per_network]

        sampled_X = raw_X[random_indices] # This will be a (5, d) tensor
        sampled_Y = raw_Y[random_indices]
        model = network_info.get('model').to(hp2.DEVICE)


        # Pass the 5*d random sample of indices from raw_X through the model
        # Assuming model takes (num_samples, d) and returns (num_samples, 1)
        with torch.no_grad():
            model_output = model(sampled_X).detach() # This will be (5, 1)

            # Set the slice of the new tensor
            output_tensor[n_dat, nmod, :, :] = model_output

            plt.figure(figsize=(7, 7)) # Adjust figure size as needed

            plt.scatter(
                sampled_Y.cpu().numpy(),    # True values (x-axis)
                model_output.cpu().numpy(), # Model's predictions (y-axis)
                color='blue',
                alpha=0.7,
                label='Model Output vs. True Y'
            )

            min_val = min(sampled_Y.min().item(), model_output.min().item())
            max_val = max(sampled_Y.max().item(), model_output.max().item())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction (y=x)')

            plt.xlabel('True Y (sampled_Y)')
            plt.ylabel('Model Output')
            plt.title('Comparison of Model Output and True Y for Sampled Data')
            plt.grid(True)
            plt.legend()

            plot_filename =f"../plots/fcn2_plots/fcn2_model_output_comparison_{i}.png"
            plt.savefig(plot_filename)
        

    ouput_tensor = output_tensor[0,:,:,:]
    means_over_networks = output_tensor.mean(dim=1, keepdim=True).squeeze(-1)
    column_vectors = means_over_networks.unsqueeze(-1) # Adds a dimension at the end: (3, 1, 10, 1)
    row_vectors = means_over_networks.unsqueeze(-2)    # Adds a dimension before the last: (3, 1, 1, 10)
    mean_outer_product = (column_vectors @ row_vectors).squeeze()

    input_matrix = output_tensor
    vectors_squeezed = input_matrix.squeeze(-1) # Shape: (3, 20, 10)
    column_vectors = vectors_squeezed.unsqueeze(-1) # Shape: (3, 20, 10, 1)
    row_vectors = vectors_squeezed.unsqueeze(-2)    # Shape: (3, 20, 1, 10)
    outer_product_matrices = column_vectors @ row_vectors
    print(outer_product_matrices.shape)
    averaged_outer_products = outer_product_matrices.mean(dim=1)
    

    print(averaged_outer_products.shape)
    print(mean_outer_product.shape)

    cov = averaged_outer_products - mean_outer_product
    print(cov.shape)

    e_cov = cov.mean(dim=0)
    print(e_cov.shape)
    eig_vals = torch.linalg.eigvalsh(e_cov)
    sorted_eigvals,_ = torch.sort(eig_vals, descending=True)
    print(sorted_eigvals.shape)
    print(sorted_eigvals)



             



        

    if loaded_count == 0:
        print("\nNo networks were successfully loaded. Check manifest paths and file existence.")
    else:
        print(f"\nSuccessfully loaded {loaded_count} networks.")
    
    print("\nScript finished.")
