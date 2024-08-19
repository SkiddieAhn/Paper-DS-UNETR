from einops import rearrange
import torch
from torch import nn
from typing import Tuple, Union
from network.components.neural_network import SegmentationNetwork
from network.components.model_components import SpatialEncoder, ChannelEncoder, UnetrUpBlock
from network.components.modules.dynunet_block import UnetOutBlock, UnetResBlock, get_conv_layer
from network.components.modules.bf_module import BIFPN_Fusion_Conv


class ConcatConv(nn.Module):
    def __init__(self,hidden_size,convOp=True):
        super().__init__()

        self.convOp = convOp
        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))
        self.conv1 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv2 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")

    def forward(self,x1,x2):
        x1 = rearrange(x1, "b c d h w -> b d h w c")        
        x2 = rearrange(x2, "b c d h w -> b d h w c")        

        x1 = self.out_proj(x1)
        x2 = self.out_proj(x2)
        x = torch.cat((x1,x2),dim=-1)
        x = rearrange(x, "b d h w c -> b c d h w")        
        
        if self.convOp:
            x = self.conv1(x)
            x = self.conv2(x)

        return x

class DS_UNETR(SegmentationNetwork):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [64, 128, 128],
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        Examples::
                    network = DS_UNETR(in_channels=input_channels,
                                    out_channels=num_classes,
                                    img_size=crop_size,
                                    feature_size=16,
                                    num_heads=4,
                                    depths=[3, 3, 3, 3,3],
                                    dims=[16,32, 64, 128, 256],
                                    do_ds=True,)
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (2, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        self.spatial_encoder = SpatialEncoder(dims=dims, depths=depths, num_heads=num_heads)
        self.channel_encoder = ChannelEncoder(dims=dims, depths=depths, num_heads=num_heads)

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            num_heads = num_heads[2],
            norm_name=norm_name,
            out_size=8 * 8 * 8,
            depth = depths[2]
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            num_heads = num_heads[1],
            norm_name=norm_name,
            out_size=16 * 16 * 16,
            depth = depths[1]
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            num_heads = num_heads[0],
            norm_name=norm_name,
            out_size=32 * 32 * 32,
            depth = depths[0]
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
            norm_name=norm_name,
            out_size=64 * 128 * 128,
            conv_decoder=True, # upsampling -> TrasposedConv
        )

        self.fusion = nn.ModuleList([BIFPN_Fusion_Conv(feature_size) for _ in range(3)])

        self.concat_conv1=ConcatConv(hidden_size=32)
        self.concat_conv2=ConcatConv(hidden_size=64)
        self.concat_conv3=ConcatConv(hidden_size=128)
        self.concat_conv4=ConcatConv(hidden_size=256)
        
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        _, hidden_states1 = self.spatial_encoder(x_in)
        _, hidden_states2 = self.channel_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders (1)
        enc1_1 = hidden_states1[0]
        enc2_1 = hidden_states1[1]
        enc3_1 = hidden_states1[2]
        enc4_1 = hidden_states1[3]
        enc4_1 = self.proj_feat(enc4_1, self.hidden_size, self.feat_size)

        # Four encoders (2)
        enc1_2 = hidden_states2[0]
        enc2_2 = hidden_states2[1]
        enc3_2 = hidden_states2[2]
        enc4_2 = hidden_states2[3]
        enc4_2 = self.proj_feat(enc4_2, self.hidden_size, self.feat_size)

        # concat & conv
        enc1 = self.concat_conv1(enc1_1, enc1_2)
        enc2 = self.concat_conv2(enc2_1, enc2_2)
        enc3 = self.concat_conv3(enc3_1, enc3_2)
        enc4 = self.concat_conv4(enc4_1, enc4_2)

        # bifpn fusion
        for i, _ in enumerate(self.fusion):
            if i==0:
                fs1, fs2, fs3, fs4 = self.fusion[i](enc1,enc2,enc3,enc4)
            else:
                fs1, fs2, fs3, fs4 = self.fusion[i](fs1, fs2, fs3, fs4)

        # Four decoders
        dec4 = self.decoder4(fs4, fs3)
        dec3 = self.decoder3(dec4, fs2)
        dec2 = self.decoder2(dec3, fs1)
        dec1 = self.decoder1(dec2, convBlock)

        if self.do_ds:
            logits = [self.out1(dec1), self.out2(dec2), self.out3(dec3)]
        else:
            logits = self.out1(dec1)

        return logits