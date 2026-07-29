import torch
import torch.nn as nn

class SpatialFlattening(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        b, c, h, w = x.shape
        
        # Flatten spatial dims and swap to sequence format: B x (H*W) x C
        x_seq = x.view(b, c, h * w).permute(0, 2, 1).contiguous()
        
        return x_seq, (h, w)

    def reverse(self, x_seq: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
        h, w = spatial_shape
        b, l, c = x_seq.shape
        
        # Revert sequence format back to spatial map: B x C x H x W
        x_out = x_seq.permute(0, 2, 1).contiguous().view(b, c, h, w)
        
        return x_out

import torch
import torch.nn as nn

class CustomMPSLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, seq_len: int, bond_dim: int = 16):
        super().__init__()
        
        self.core_left = nn.Parameter(torch.empty(in_channels, bond_dim, bond_dim))
        self.core_mid = nn.Parameter(torch.empty(bond_dim, seq_len, seq_len, bond_dim))
        self.core_right = nn.Parameter(torch.empty(bond_dim, bond_dim, out_channels))
        
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.core_left)
        nn.init.xavier_normal_(self.core_mid)
        nn.init.xavier_normal_(self.core_right)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Channel to Bond Projection
        left_contract = torch.einsum('blc, cij -> blij', x, self.core_left)
        
        # 2. Global Spatial Context Mixing
        mid_contract = torch.einsum('blij, ilmk -> bmjk', left_contract, self.core_mid)
        
        # 3. Bond to Channel Projection
        out = torch.einsum('bmjk, jkd -> bmd', mid_contract, self.core_right)
        
        return out


if __name__ == "__main__":
    print("[debug] Starting MPS Layer Dummy Run...")
    
    BATCH_SIZE = 4
    IN_CHANNELS = 1024
    OUT_CHANNELS = 1024
    SEQ_LEN = 256  # 16x16 spatial resolution
    BOND_DIM = 32
    
    mps_bottleneck = CustomMPSLayer(
        in_channels=IN_CHANNELS, 
        out_channels=OUT_CHANNELS, 
        seq_len=SEQ_LEN, 
        bond_dim=BOND_DIM
    ).cuda()
    
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, IN_CHANNELS).cuda()
    print(f"[debug] Input Shape: {dummy_input.shape}")
    
    try:
        dummy_output = mps_bottleneck(dummy_input)
        print(f"[debug] Output Shape: {dummy_output.shape}")
        
        assert dummy_output.shape == (BATCH_SIZE, SEQ_LEN, OUT_CHANNELS)
        print("[success] CustomMPSLayer forward pass completed successfully.")
        
    except RuntimeError as e:
        print(f"[error] Tensor contraction failed:\n{e}")