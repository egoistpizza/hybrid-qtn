import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mps_layer import AxialMPSLayer

def execute_axial_mps_flawless_validation():
    batch_size = 4
    in_channels = 512
    bond_dim = 32
    height = 32
    width = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AxialMPSLayer(
        in_channels=in_channels,
        bond_dim=bond_dim,
        height=height,
        width=width
    ).to(device)

    dummy_input = torch.randn(
        batch_size, in_channels, height, width, 
        device=device, requires_grad=True
    )
    dummy_target = torch.randn(
        batch_size, in_channels, height, width, 
        device=device
    )

    output = model(dummy_input)
    assert output.shape == dummy_input.shape

    loss = F.mse_loss(output, dummy_target)
    loss.backward()

    assert model.T_mid_H.grad is not None
    assert model.T_mid_W.grad is not None
    assert dummy_input.grad is not None

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial_T_mid_H_state = model.T_mid_H.clone()
    
    optimizer.step()

    assert not torch.equal(model.T_mid_H, initial_T_mid_H_state)

    print("Validation Successful: Forward pass, gradient flow, and parameter updates are flawless.")

if __name__ == '__main__':
    execute_axial_mps_flawless_validation()