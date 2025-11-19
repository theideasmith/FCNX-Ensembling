
import torch.nn as nn
import torch
from typing import Callable, Dict

class LayerOutputTracker:
    """
    A class to track the outputs of each layer in a PyTorch nn.Module during a forward pass.

    Attributes:
        model (nn.Module): The PyTorch model to track.
        layer_outputs (Dict[str, torch.Tensor]): A dictionary to store the outputs of each layer,
            where the keys are the layer names and the values are the output tensors.
        hooks (List[torch.Tensor]): A list to store the registered hooks, allowing for easy removal.
    """

    def __init__(self, model: nn.Module):
        """
        Initializes the LayerOutputTracker with a PyTorch model.

        Args:
            model (nn.Module): The PyTorch model to track.
        """
        self.model = model
        self.layer_outputs = {}
        self.hooks = []

    def _create_hook(self, layer_name: str) -> Callable:
        """
        Creates a forward hook function for a specific layer.  This function captures the output
        of the layer and stores it in the `layer_outputs` dictionary.

        Args:
            layer_name (str): The name of the layer for which to create the hook.

        Returns:
            Callable: A forward hook function.
        """
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            """
            Forward hook function that stores the output of the layer.  Uses detach() to avoid
            interfering with the gradient computation.

            Args:
                module (nn.Module): The layer itself.
                input (torch.Tensor): The input to the layer.
                output (torch.Tensor): The output of the layer.
            """
            self.layer_outputs[layer_name] = output.detach()  # Detach to prevent gradient issues
        return hook

    def attach_hooks(self) -> None:
        """
        Attaches forward hooks to all layers in the model.  It uses named_modules() to get
        all sub-modules, and attaches a hook to each one *that doesn't already have children*.
        This prevents duplicate hooks on parent modules.  It stores the hook handles in
        self.hooks for later removal.
        """
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0: # Only attach to leaf nodes
                hook = self._create_hook(name)
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model with the given input, after attaching the hooks.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The final output of the model.
        """
        self.attach_hooks()
        self.layer_outputs.clear()  # Clear any previous outputs
        output = self.model(x)
        return output

    def remove_hooks(self) -> None:
        """
        Removes all registered hooks.  This is crucial to prevent memory leaks and unexpected
        behavior in subsequent forward passes.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []  # Clear the list of hooks

    def get_layer_outputs(self) -> Dict[str, torch.Tensor]:
        """
        Returns the dictionary containing the outputs of each layer.

        Returns:
            Dict[str, torch.Tensor]: A dictionary where keys are layer names and values are
                the corresponding output tensors.
        """
        return self.layer_outputs

    def __del__(self):
        """
        Destructor to ensure hooks are removed when the object is garbage collected.
        """
        self.remove_hooks()