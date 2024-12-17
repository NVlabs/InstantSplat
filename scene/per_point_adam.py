import torch
from torch.optim import Optimizer

class PerPointAdam(Optimizer):
    """Implements Adam optimizer with per-point learning rates.
    
    Allows unique learning rates for each point in specified parameter tensors,
    useful for point cloud optimization.

    Args:
        params: Iterable of parameters to optimize or parameter groups
        lr (float, optional): Default learning rate (default: 1e-3)
        betas (tuple, optional): Coefficients for moving averages (default: (0.9, 0.999))
        eps (float, optional): Term for numerical stability (default: 1e-8)
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not all(0.0 <= x for x in [lr, eps, weight_decay]):
            raise ValueError(f"Invalid learning parameters: lr={lr}, eps={eps}, weight_decay={weight_decay}")
        if not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"Invalid beta parameters: {betas}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, per_point_lr=None)
        super().__init__(params, defaults)

    def _adjust_per_point_lr(self, per_point_lr, grad, mask):
        """Adjusts per-point learning rates based on gradient magnitudes."""
        grad_magnitude = grad.norm(dim=-1)
        scaling_factor = torch.ones_like(grad_magnitude)
        grad_sigmoid = torch.sigmoid(grad_magnitude[mask])
        scaling_factor[mask] = 0.99 + (grad_sigmoid * 0.02)
        return per_point_lr * scaling_factor.unsqueeze(1)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            per_point_lr = group.get('per_point_lr')
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('PerPointAdam does not support sparse gradients')

                # Initialize state if needed
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Get state values
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Apply weight decay if specified
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Compute mask for non-zero gradients
                grad_norm = grad.norm()
                mask = grad_norm > 0

                # Update momentum terms
                exp_avg.masked_scatter_(mask, 
                    exp_avg[mask].mul_(beta1).add_(grad[mask], alpha=1 - beta1))
                exp_avg_sq.masked_scatter_(mask,
                    exp_avg_sq[mask].mul_(beta2).addcmul_(grad[mask], grad[mask], value=1 - beta2))

                # Compute bias corrections
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute step size
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr'] * (bias_correction2 ** 0.5 / bias_correction1)

                # Apply updates
                if per_point_lr is not None:
                    if not isinstance(per_point_lr, torch.Tensor):
                        raise TypeError("per_point_lr must be a torch.Tensor")
                    if per_point_lr.device != p.data.device:
                        raise ValueError("per_point_lr must be on the same device as parameter")
                    
                    expected_shape = p.data.shape[:1] + (1,) * (p.data.dim() - 1)
                    if per_point_lr.shape != expected_shape:
                        raise ValueError(f"Invalid per_point_lr shape. Expected {expected_shape}, got {per_point_lr.shape}")

                    scaled_step_size = step_size * per_point_lr
                    p.data.add_(-scaled_step_size * (exp_avg / denom))
                    per_point_lr = self._adjust_per_point_lr(per_point_lr, grad, mask)
                else:
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss