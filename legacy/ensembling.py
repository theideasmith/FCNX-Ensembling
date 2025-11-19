import FCN3 as nnkit
from torch.utils.data import TensorDataset
import torch
def ensemble_train(dataset: nnkit.DataManager, n_ensemble: int = 10, n_epochs: int = 1000):
    """
    Train an ensemble of FCN3s and compute the posterior kernels for each internal layer.
    
    Args:
        dataset (nnkit.DataManager): The dataset to train on.
        n_ensemble (int): The number of models in the ensemble.
        n_epochs (int): The number of epochs to train each model.
        
    Returns:
        list: A list of trained models.
    """
    models = []
    for _ in range(n_ensemble):
        model = nnkit.FCN3()
        model.train(dataset, n_epochs)
        models.append(model)
    return models

def ensemble_forward_internals(models : list, data : TensorDataset):

    inner_outputs = {}
    inner_outputs['readout'] = []
    for m in models:
        tracker = nnkit.LayerOutputTracker(m)
        output = tracker.forward(data)
        inner_output = tracker.get_layer_outputs()
        for k, v in inner_output.items():
            if k not in inner_outputs:
                inner_outputs[k] = []
            inner_outputs[k].append(v)
        inner_outputs['readout'].append(output)
    return inner_outputs