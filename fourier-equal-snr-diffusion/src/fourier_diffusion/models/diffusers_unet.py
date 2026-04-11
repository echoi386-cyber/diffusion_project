import torch
import torch.nn as nn
from diffusers import UNet2DModel


class DiffusersUNet(nn.Module):
    def __init__(self, in_ch: int, sample_size: int = 32, base_channels: int = 128):
        super().__init__()

        self.net = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_ch,
            out_channels=in_ch,
            layers_per_block=2,
            block_out_channels=(
                base_channels,
                base_channels,
                base_channels * 2,
                base_channels * 2,
            ),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            norm_num_groups=32,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # diffusers expects floating timesteps
        t = t.float()
        return self.net(x, t).sample