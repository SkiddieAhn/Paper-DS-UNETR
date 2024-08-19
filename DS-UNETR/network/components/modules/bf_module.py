import torch
import torch.nn as nn


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
==================================== BIFPN Structure for UNET =======================================================
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        
        self.bn = nn.BatchNorm3d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class MFC_two(nn.Module):
    def __init__(self, dim_e, dim_r, epsilon=0.0001, version='up'): 
        super().__init__()
        # dim_e = local feature map channel (16,32,64,128)
        # dim_r = global feature map channel (32,64,128,256)
    
        self.version = version
        self.epsilon = epsilon
        self.w1 = nn.Parameter(torch.ones(2, 1))
        self.w1_relu = nn.ReLU()
        self.upsample = nn.ConvTranspose3d(in_channels=dim_r, out_channels=dim_e, kernel_size=2, stride=2)

        if version == 'up':
            self.upsample = nn.ConvTranspose3d(in_channels=dim_r, out_channels=dim_e, kernel_size=2, stride=2)
            self.conv = DepthwiseConvBlock(in_channels=dim_e, out_channels=dim_e)
        else:
            self.downsample = nn.Conv3d(in_channels=dim_e, out_channels=dim_r, kernel_size=2, stride=2)
            self.conv = DepthwiseConvBlock(in_channels=dim_r, out_channels=dim_r)

    def forward(self,e,r):
        '''
        e: local feature (H x W x D x C)
        r: global feature (H/2 x W/2 x D/2 x 2C)
        '''
        w1 = self.w1_relu(self.w1)
        w1 = w1/(torch.sum(w1, dim=0) + self.epsilon)    

        if self.version == 'up':
            x = self.conv((w1[0, 0] * e) + (w1[1, 0] * self.upsample(r)))
        else:
            x = self.conv((w1[0, 0] * self.downsample(e)) + (w1[1, 0] * r))

        return x


class MFC_three(nn.Module):
    def __init__(self, dim_e, dim_r, epsilon=0.0001): 
        super().__init__()
        # dim = channel

        self.conv = DepthwiseConvBlock(in_channels=dim_r, out_channels=dim_r)
        self.epsilon = epsilon
        self.w2 = nn.Parameter(torch.ones(3, 1))
        self.w2_relu = nn.ReLU()
        self.downsample = nn.Conv3d(in_channels=dim_e, out_channels=dim_r, kernel_size=2, stride=2)

    def forward(self,e, r1, r2):
        '''
        e: local feature (H x W x D x C)
        r1, r2: global feature (H/2 x W/2 x D/2 x 2C)
        '''
        w2 = self.w2_relu(self.w2)
        w2 = w2/(torch.sum(w2, dim=0) + self.epsilon)    

        x = self.conv((w2[0, 0] * self.downsample(e)) + (w2[1, 0] * r1) + (w2[2, 0] * r2))

        return x


class BIFPN_Fusion_Conv(nn.Module):
    def __init__(self, feature_size=16):
        super().__init__()

        self.fusion1 = MFC_two(dim_e=feature_size*8, dim_r=feature_size*16, version='up')
        self.fusion2 = MFC_two(dim_e=feature_size*4, dim_r=feature_size*8, version='up')
        self.fusion3 = MFC_two(dim_e=feature_size*2, dim_r=feature_size*4, version='up')
        self.fusion4 = MFC_three(dim_e=feature_size*2, dim_r=feature_size*4)
        self.fusion5 = MFC_three(dim_e=feature_size*4, dim_r=feature_size*8)
        self.fusion6 = MFC_two(dim_e=feature_size*8, dim_r=feature_size*16, version='down')

    def forward(self,x1,x2,x3,x4):
        '''
        x1: local feature (H x W x D x C)
        x2: local feature (H/2 x W/2 x D/2 x 2C)
        x3: global feature (H/4 x W/4 x D/4 x 4C)
        x4: global feature (H/8 x W/8 x D/8 x 8C)
        '''          

        first = self.fusion1(x3, x4)
        second = self.fusion2(x2, first)
        third = self.fusion3(x1, second)
        fourth = self.fusion4(third, x2, second)
        fifth = self.fusion5(fourth, x3, first)
        sixth = self.fusion6(fifth, x4)
        
        return third, fourth, fifth, sixth