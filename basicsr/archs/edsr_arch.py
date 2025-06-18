import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
from numpy.core.fromnumeric import size
from .arch_util import Upsample, make_layer
from torch.nn import functional as F
from torch.nn.modules import module
import cv2

class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0*g:1*g, 1, 2] = 1.0
        mask[:, 1*g:2*g, 1, 0] = 1.0
        mask[:, 2*g:3*g, 2, 1] = 1.0
        mask[:, 3*g:4*g, 0, 1] = 1.0
        mask[:, 4*g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1)
        return y
class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0*g:1*g, 0, 1, 2] = 1.0 ## left
        self.weight[1*g:2*g, 0, 1, 0] = 1.0 ## right
        self.weight[2*g:3*g, 0, 2, 1] = 1.0 ## up
        self.weight[3*g:4*g, 0, 0, 1] = 1.0 ## down
        self.weight[4*g:, 0, 1, 1] = 1.0 ## identity

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y)
        return y
class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory':
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y
class LFE(nn.Module):
    def __init__(self, inp_channels=32, out_channels=32, exp_ratio=4, act_type='relu'):
        super(LFE, self).__init__()
        self.exp_ratio = exp_ratio
        self.act_type  = act_type

        self.conv0 = ShiftConv2d(inp_channels, out_channels*exp_ratio)
        self.conv1 = ShiftConv2d(out_channels*exp_ratio, out_channels)

        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)
        return y

class CALayer(nn.Module):
    def __init__(self, channel=32, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            ShiftConv2d(channel,channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class DPCAB(nn.Module):
    def __init__(self, nf1, nf2, ksize1=3, ksize2=3, reduction=4):
        super().__init__()

        self.body1 = nn.Sequential(
            LFE(nf1,nf1),CALayer(nf1, reduction),
        )
        self.body11 = nn.Sequential(
            LFE(nf1,nf1),CALayer(nf1, reduction),
        )
        self.body2 = nn.Sequential(
            LFE(nf2,nf2),CALayer(nf2, reduction),
        )
        self.body22 = nn.Sequential(
            LFE(nf2,nf2),
        )
        self.CA_body1 = nn.Sequential(
            LFE(nf1+nf2,nf1),CALayer(nf1, reduction),
        )
        self.CA_body2 = CALayer(nf2, reduction)

    def forward(self, x):

        a = x[0]
        f1 = self.body1(x[0])
        f11 = a + f1
        f111 = f11 +self.body11(f11)

        b = x[1]
        f2 = self.body2(x[1])
        f22 = f2 + b
        f222 = self.body22(f22)

        ca_f1 = self.CA_body1(torch.cat([f111, f222], dim=1))
        ca_f2 = f2

        x[0] = x[0] + ca_f1
        x[1] = x[1] + ca_f2
        return x
class DPCAG(nn.Module):

    def __init__(self, nf1, nf2, ksize1, ksize2, nb):
        super().__init__()

        self.body = nn.Sequential(*[DPCAB(nf1, nf2, ksize1, ksize2) for _ in range(nb)])

    def forward(self, x):
        y = self.body(x)
        y[0] = x[0] + y[0]
        y[1] = x[1] + y[1]
        return y


@ARCH_REGISTRY.register()
class EDSR(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=32,
                 num_block=16,
                 upscale=2,
                 res_scale=1,
                 nb=5,ng=1,scale=2, #new
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR, self).__init__()

        self.conv_after_body = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        nf = num_feat
        nf2 = num_feat // 2

        self.head1 = nn.Conv2d(nf, nf2, 3, 1, 1)

        body1 = [DPCAG(nf, nf2, 3, 3, nb) for _ in range(ng)]
        self.body1 = nn.Sequential(*body1)

        self.fusion = nn.Conv2d(nf+nf2, nf, 3, 1, 1)
        self.LFE = LFE()
        self.LFE1 = LFE()
        self.LFE2 = LFE()
        self.LFE3 = LFE()

        self.nearup = nn.Upsample(scale_factor=2,mode='nearest')
        self.conv1x1 = nn.Conv2d(3,3,1)
        self.sigmoid = nn.Sigmoid()
        self.conv1x1last = nn.Conv2d(3,3,1)
        self.CA1 = CALayer()
        self.CA2 = CALayer()
        self.CA3 = CALayer()
        self.CA4 = CALayer()
        self.sconv = ShiftConv2d(3, 3)


    def forward(self, x):
        x_high = []
        for i in range(x.shape[0]):
            # 对每个图像应用高斯模糊
            blurred_image = cv2.GaussianBlur(x[i], (5, 5), 0)
            x_high.append(blurred_image)
        x_high = np.array(x_high)
        LR = x
        feature1 = self.conv_after_body(x) #3_>32
        f1 = self.head1(feature1) #channel reduce gy  32/2=16

        f2 = feature1   #new++++
        # f21 = feature1
        # f2 = self.LFE(feature1)           #R
        # f2 = self.CA1(f2)
        # f3 = f2 + f21

        # f4 = self.LFE1(f3)
        # f4 = self.CA2(f2)
        # f4 = f4 + f3

        # f5 = self.LFE2(f4)
        # f5 = self.CA3(f5)
        # f5 = f5 + f4

        # f6 = self.LFE3(f5)
        # f6 = self.CA4(f6)
        # f2 = f5 + f6
        # f2 = f2 + f21

        inputs = [f2, f1] # 32 16
        f2, f1 = self.body1(inputs)  # DPG

        # f = self.fusion(torch.cat([f1, f2], dim=1)) # 32+16=48->32
        # ff = f+ feature1

        # # LRu = self.nearup(LR)
        # x = self.conv_last(self.nearup(ff))
        # # x = torch.mul(self.sigmoid(self.conv1x1()),x)+f
        # x = self.conv1x1last(x)

        f = self.fusion(torch.cat([f1, f2], dim=1)) # 32+16=48->32
        ff = f+ feature1
        f = self.sigmoid(self.conv_last(self.nearup(ff)))
        LRu = self.nearup(LR)
        x = torch.mul(LRu,f)+self.conv_last(self.nearup(ff))
        # x = torch.mul(self.sigmoid(self.conv1x1()),x)+f
        x = self.sconv(x)
        return x

if __name__ == "__main__":
    model = EDSR(3,3)
    img = torch.randn()