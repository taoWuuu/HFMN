import torch
import math
from torch import nn as nn
import numbers
from thop import profile
from thop import clever_format
from einops import rearrange
from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.archs.arch_util import trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY
from numpy.core.fromnumeric import size
from torch.nn import functional as F
from torch.nn.modules import module
from timm.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat

    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dim = dim
        self.softmax = nn.Softmax(dim=-1)

        self.q_c = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv_c = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_c = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q_t = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv_t = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_t = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv_t = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out_c = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out_t = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.concat = nn.Conv2d(dim * 2, dim, 1)

    def _forward(self, q, kv):
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax(attn)
        out = (attn @ v)
        return out

    def forward(self, low, high):
        self.h, self.w = low.shape[2:]

        q_c = self.q_dwconv_c(self.q_c(high))
        kv_c = self.kv_dwconv_c(self.kv_c(high))
        q_t = self.q_dwconv_t(self.q_t(low))
        kv_t = self.kv_dwconv_t(self.kv_t(low))
        out_c = self._forward(q_c, kv_t)
        out_t = self._forward(q_t, kv_c)
        out_c = rearrange(out_c, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=kv_t.shape[-2], w=kv_t.shape[-1])
        out_c = self.project_out_c(out_c)
        out_t = rearrange(out_t, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=kv_c.shape[-2],w=kv_c.shape[-1])
        out_t = self.project_out_t(out_t)
        out = torch.cat([out_c, out_t], dim=1)
        out = self.concat(out)
        return out

    
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.hid_fea = hidden_features
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        self.h, self.w = x.shape[2:]
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class DIAB(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(DIAB, self).__init__()

        self.norm = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.dim = dim

    def forward(self, low, high):
        h, w = low.shape[2:]
        x = low + self.attn(self.norm(low), high)
        x = x + self.ffn(self.norm(x))
        
        
        return x

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, dim):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

class PAB(nn.Module):

    def __init__(self, dim, k_size=3):

        super(PAB, self).__init__()
        self.k2 = nn.Conv2d(dim, dim, 1) # 1x1 convolution dim->dim
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(dim, dim, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(dim, dim, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x):

        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out

class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            # ChannelAttention(num_feat, squeeze_factor)
        )
    def forward(self, x):
        return self.cab(x)

class LFHB(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.mid_dim = dim // 2
        self.dim = dim
        self.conv1_a = nn.Conv2d(dim, self.mid_dim, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(dim, self.mid_dim, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=3, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.last_fc = nn.Conv2d(self.dim, self.dim, 1)
        self.PAB = PAB(self.mid_dim)
        # High-frequency enhancement branch
        self.max_pool = nn.MaxPool2d(3, 1, 1)

    def forward(self, x):
        self.h, self.w = x.shape[2:]
        short = x

        # Local feature extraction branch
        lfe = self.lrelu(self.conv1_a(x))
        lfe = self.lrelu(self.PAB(lfe))

        # High-frequency enhancement branch
        hfe = self.lrelu(self.conv1_b(x))
        hfe = self.lrelu(self.conv3(self.max_pool(hfe)))

        x = torch.cat([lfe, hfe], dim=1)
        x = short + self.last_fc(x)
        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class MAB(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 expand=2,
                 attn_drop_rate=0,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.VSSM = SS2D(d_model=dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.channelBlock = CAB(dim)
        self.skip_scale = nn.Parameter(torch.ones(dim))
        self.skip_scale2 = nn.Parameter(torch.ones(dim))
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0,2,3,1)#B,H,W,C
        shortcut1 = x
        x = self.norm1(x)
        x = self.VSSM(x) + shortcut1 * self.skip_scale
        shortcut2 = x
        x = self.channelBlock(self.norm2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous() + shortcut2 * self.skip_scale2
        # x = self.norm2(x) + shortcut2 * self.skip_scale2
        x = x.permute(0, 3, 1, 2)
        return x

class Low_Branch(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 expand=2,
                 attn_drop_rate=0,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.VMlist = nn.Sequential(*[MAB(dim=dim,
                                          d_state=d_state,
                                          attn_drop_rate=attn_drop_rate,
                                          expand=2,
                                          norm_layer=nn.LayerNorm) for _ in range(2)])
        self.conv = nn.Conv2d(dim,dim,3,1,1,groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.VMlist(x)
        low = self.conv(x)
        return low


class CFMB(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 d_state,
                 expand=2,
                 attn_drop_rate=0,
                 norm_layer=nn.LayerNorm,
                 mlp_ratio=1.):
        super().__init__()

        # self.lpa =LPA(dim,dim)
        # self.atten = Attention(dim)
        self.high_branch = LHFB(dim)
        self.low_branch = Low_Branch(dim=dim,
                                     d_state=d_state,
                                     expand=mlp_ratio,
                                     attn_drop_rate=attn_drop_rate,
                                     norm_layer=nn.LayerNorm)
        self.hfb = DIAB(dim, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        # high = self.lpa(x)
        b, c, h, w = x.shape
        high = self.high_branch(x)
        low = self.low_branch(x)  
        main_out = self.hfb(low, high)
        out = self.conv(main_out) + x
        return out


class CMFG(nn.Module): 

    def __init__(self, dim, depth, num_heads, num_feat, mlp_ratio, d_state, norm_layer, bias=False):
        super().__init__()

        self.body = nn.Sequential(*[CFMB(dim=dim,
                                          num_heads=num_heads,
                                          mlp_ratio=mlp_ratio,
                                          d_state=d_state,
                                          expand=mlp_ratio,
                                          norm_layer=nn.LayerNorm) for _ in range(depth)])
        self.conv3x3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        y = self.body(x)  # 得到最后的全局信息
        y = self.conv3x3(y)
        return y


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        self.scale = scale
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


# @ARCH_REGISTRY.register()
class HFMN(nn.Module): 
    def __init__(self,
                 in_chans=3,
                 embed_dim=32,
                 depths=(2, 2, 2, 2),
                 num_heads=8,
                 d_state=10,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 upscale=4,
                 img_range=1.,
                 resi_connection='1conv',
                 **kwargs):
        super(HFMN, self).__init__()
        num_out_ch = in_chans
        num_feat = embed_dim
        self.img_range = img_range
        self.conv_first = nn.Conv2d(in_chans, num_feat, 3, 1, 1)
        # self.conv_first = MSDCB(num_feat)
        self.upsample = UpsampleOneStep(upscale, num_feat, num_out_ch)
        self.mlp_ratio = mlp_ratio
        self.d_state = d_state

        body1 = [CMFG(
            dim=num_feat,
            depth=depths[i_layer],
            num_heads=num_heads,
            num_feat=num_feat,
            mlp_ratio=self.mlp_ratio,
            d_state=self.d_state,
            norm_layer=nn.LayerNorm) for i_layer in range(len(depths))]
        self.body1 = nn.Sequential(*body1)
        self.conv_after = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.norm = LayerNorm(num_feat)
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        feature = self.conv_first(x)
        f = self.body1(feature)
        feature = self.conv_after(f) + feature
        x = self.upsample(feature)
        x = x / self.img_range + self.mean
        return x


if __name__ == "__main__":
    from torchsummary import summary

    model = HFMN()
    model = model.cuda()
    num_params = 0
    for param in model.parameters(): 
        num_params += param.nelement()
    print("# of params:", num_params)
    x = torch.randn(1, 3, 320, 180).cuda()
    flops, params = profile(model, inputs=(x, ))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs,params)
    y = model(x)
    print(y.shape)