import torch
import numpy as np


def compute_relative_world_to_camera(R1, t1, R2, t2):
    zero_row = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device="cuda") #, requires_grad=True
    E1_inv = torch.cat([torch.transpose(R1, 0, 1), -torch.transpose(R1, 0, 1) @ t1.reshape(-1, 1)], dim=1)
    E1_inv = torch.cat([E1_inv, zero_row], dim=0)
    E2 = torch.cat([R2, -R2 @ t2.reshape(-1, 1)], dim=1)
    E2 = torch.cat([E2, zero_row], dim=0)

    # Compute relative transformation
    E_rel = E2 @ E1_inv

    # # Extract rotation and translation
    # R_rel = E_rel[:3, :3]
    # t_rel = E_rel[:3, 3]
    # E_rel = torch.cat([E_rel, zero_row], dim=0)

    return E_rel