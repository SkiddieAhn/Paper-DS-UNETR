import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from torch import nn
from einops import rearrange


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
================================== Transformers for Encoder & Decoder ===============================================
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class SpatialTransformer(nn.Module):
    def __init__(self, input_size, dim, num_heads, window_size=[4,4,4], drop=0., attn_drop=0.1):
        super().__init__()
        '''
        input_size = input_width * input_height * input_depth
        dim = input_channel
        drop = dropout rate for feed forward
        attn_drop = dropout rate for attention
        '''
        self.window_size = window_size
        self.shift_size = [window_size[0] // 2, window_size[1] // 2, window_size[2] // 2]
        self.swindual = nn.ModuleList([
                        SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                            shift_size=[0,0,0] if (i % 2 == 0) else self.shift_size,
                                            drop=drop, attn_drop=attn_drop)
                        for i in range(2)])
        
    def forward(self,x):
        '''
        X : [B, C, D, H, W]
        '''
        x = rearrange(x, "b c d h w -> b d h w c")
        b, d, h, w, c = x.size()

        # make attention_mask
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)

        for blk in self.swindual:
            x = blk(x, attn_mask)

        x = rearrange(x, "b d h w c -> b c d h w")        
        
        return x

class ChannelTransformer(nn.Module):
    def __init__(self, input_size, dim, num_heads):
        super().__init__()
        '''
        input_size = input_width * input_height * input_depth
        dim = input_channel
        drop = dropout rate for feed forward
        attn_drop = dropout rate for attention
        '''
        self.cnlblock = ChannelAttnBlock(input_size=input_size, dim=dim, num_heads=num_heads)
        
    def forward(self,x):
        '''
        X : [B, C, D, H, W]
        '''
        x = rearrange(x, "b c d h w -> b d h w c")
        
        # channel attention
        x = self.cnlblock(x)
        x = rearrange(x, "b d h w c -> b c d h w")        
        
        return x


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
==================================== Attention Block for Transformer ================================================
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
SwinB :Swin Transformer Block
'''
class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size,
        shift_size,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer = nn.GELU,
        norm_layer = nn.LayerNorm,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x, mask_matrix):
        '''
        X : [B, D, H, W, C]
        '''
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        dims = [b, d, h, w]
        
        shortcut = x
        x = self.norm1(x)

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))

        # reverse cyclic shift
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

'''
CAB : Channel Attention Block
'''
class ChannelAttnBlock(nn.Module):
    def __init__(self, input_size, dim, num_heads=4, qkv_bias=False, drop=0., attn_drop=0.1, mlp_ratio=4, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop_path=0.):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size, dim))

        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttn(dim, num_heads, qkv_bias, attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        '''
        X : [B, D, H, W, C]
        '''
        B, D, H, W, C = x.size()
        x = x.contiguous().view(B, H*W*D, C)

        # positional encoding
        x = x + self.pos_embed

        shortcut = x
        x = self.norm1(x)

        # Channel Attn
        x = self.attn(x) 
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.contiguous().view(B,D,H,W,C)
        return x


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
==================================== Attention Module for Attention Block ===========================================
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
WA : Window Attention 
'''
class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size, # tuple[int]
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                num_heads,
            )
        )
    
        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        '''
        x: [B, N, C] // N=HWD
        '''
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

'''
CA : Channel Attention 
'''
class ChannelAttn(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0.1, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkv are 3 linear layers (query, key, value)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop) 

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        '''
        Channel Attention
        : [ Q_T x K ]

        x: [B, N, C] // N=HWD
        '''
        B, N, C = x.shape # N=HWD
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads) # B x N x 3 x h x C/h
        qkv = qkv.permute(2, 0, 3, 1, 4) # 3 x B x h x N x C/h
        q, k, v = qkv[0], qkv[1], qkv[2] # B x h x N x C/h

        q_t = q.transpose(-2, -1) # B x h x C/h x N
        attn_CA = (q_t @ k) * self.temperature # [Q_T x K] B x h x C/h x C/h 

        attn_CA = attn_CA.softmax(dim=-2)
        attn_CA = self.attn_drop(attn_CA) # [Channel Attn Map] B x h x C/h x C/h

        # [V x Channel Attn Map] B x h x N x C/h -> B x C/h x h x N -> B x N x C
        x_CA = (v @ attn_CA).permute(0, 3, 1, 2).reshape(B, N, C)
        
        # linear
        x = self.proj(x_CA)
        x = self.proj_drop(x)

        return x



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
==================================== Any Modules for SwinTransformer ================================================
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
MLP : Multi Layer Perceptron
'''
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

'''
Window Partition
'''
def window_partition(x, window_size):
    b, d, h, w, c = x.size()
    x = x.view(
        b,
        d // window_size[0],
        window_size[0],
        h // window_size[1],
        window_size[1],
        w // window_size[2],
        window_size[2],
        c,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
    )
 
    return windows


'''
Window Reverse
'''
def window_reverse(windows, window_size, dims):
    b, d, h, w = dims
    x = windows.view(
        b,
        d // window_size[0],
        h // window_size[1],
        w // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    return x

'''
get window size
'''
def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


'''
Compute Mask
'''
def compute_mask(dims, window_size, shift_size, device):
    cnt = 0

    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
==================================== Patch Merging, Expanding =======================================================
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
Patch Merging
'''
class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        '''
        we remove layer norm. because we use GroupNorm outside.
        we assume that h,w,d are even numbers.
        '''
        super().__init__()
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x):
        '''
        x: B,C,D,H,W
        '''
        x = x.permute(0,2,3,4,1) # [B, D, H, W, C]
        
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)

        x = self.reduction(x)
        x = x.permute(0, 4, 1, 2, 3) # [B, C, D, H, W]
        
        return x

'''
Patch Expanding
'''
class PatchExpanding3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 4 * dim, bias=False)

    def forward(self, y):
        """
        y: B,C,D,H,W
        """
        y=y.permute(0,3,4,2,1) # [B, D, H, W, C]
        B, D, H, W, C = y.size()

        y=self.expand(y) # B, H, W, D, 4*C
    
        y=rearrange(y,'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c', p1=2, p2=2, p3=2, c=C//2) # B, 2*D, 2*H, 2*W, C//2

        y=y.permute(0,4,3,1,2) # B, C//2, 2*D, 2*H, 2*W
        
        return y