
import torch
import math
class FusedEnsembleLinearErf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W):
        """
        x: (ens, n_in, P)
        W: (ens, n_out, n_in)
        """
        # Compute pre-activation
        pre = torch.bmm(W, x)
        output = torch.erf(pre)
        
        # Save tensors for backward - we only save what's absolutely necessary
        ctx.save_for_backward(x, W, pre)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (ens, n_out, P)
        """
        x, W, pre = ctx.saved_tensors
        
        # 1. Compute derivative of Erf: (2/sqrt(pi)) * exp(-pre^2)
        # We do this in-place where possible to save memory
        grad_erf = torch.exp(-pre.pow(2)).mul_(2.0 / math.sqrt(math.pi))
        
        # 2. Chain rule: grad_output * grad_erf
        # This is the gradient with respect to the pre-activations
        dL_dpre = grad_output * grad_erf
        
        # 3. Gradient w.r.t W: (dL_dpre) @ x.T
        # (ens, n_out, P) @ (ens, P, n_in) -> (ens, n_out, n_in)
        grad_W = torch.bmm(dL_dpre, x.transpose(1, 2))
        
        # 4. Gradient w.r.t x: W.T @ (dL_dpre)
        # (ens, n_in, n_out) @ (ens, n_out, P) -> (ens, n_in, P)
        grad_x = torch.bmm(W.transpose(1, 2), dL_dpre)

        return grad_x, grad_W

# Helper wrapper
def fused_ensemble_erf(x, W):
    return FusedEnsembleLinearErf.apply(x, W)