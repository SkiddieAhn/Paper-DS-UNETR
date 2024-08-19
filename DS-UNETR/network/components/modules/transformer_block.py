import torch.nn as nn
from network.components.modules.tf_module import SpatialTransformer, ChannelTransformer


class SpatialTransformerBlock(nn.Module):
    def __init__(
        self,
        input_size,
        dim,
        num_heads,
        window_size=[4,4,4],
        drop=0.,
        attn_drop=0.1
    ) -> None:
        super().__init__()

        self.stf = SpatialTransformer(
            input_size = input_size,
            dim = dim,
            num_heads = num_heads,
            window_size = window_size,
            drop = drop,
            attn_drop = attn_drop
        )

    def forward(self,x):
        return self.stf(x)


class ChannelTransformerBlock(nn.Module):
    def __init__(
        self,
        input_size,
        dim,
        num_heads,
    ) -> None:
        super().__init__()

        self.ctf = ChannelTransformer(
            input_size = input_size,
            dim = dim,
            num_heads = num_heads,
        )

    def forward(self,x):
        return self.ctf(x)


