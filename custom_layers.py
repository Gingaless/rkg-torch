
import math
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
import stylegan1.custom_layers as stg1cl
from stylegan1.c_utils import calc_pool2d_pad
from kornia.filters import filter2D

stg1cl.default_conv_weight_norm = None
stg1cl.default_fc_weight_norm = None
default_up_sample = partial(F.interpolate, scale_factor=2, mode='binear')
image_channels = 3
leaky_relu_alpha = 0.2

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        self.blur_kernel = torch.Tensor([[
            [0.1,0.2,0.1],
            [0.2,0.4,0.2],
            [0.1,0.2,0.1]
        ]])
        #self.register_buffer('blur_kernel', self.blur_kernel)
    def forward(self, x):
        return filter2D(x, self.blur_kernel, normalized=True)


class EqualConv2D(nn.Module):

    def __init__(self, input_size, in_channels, out_channels, kernel_size, *args, **kwargs):

        super().__init__()
        self._conv = stg1cl.PadConv2D(input_size, in_channels, out_channels, kernel_size if input_size > kernel_size else input_size, *args, **kwargs)
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)

        self.stride = self._conv.stride

    def forward(self, x):
        return self._conv(x)*self.scale

class toRGB(EqualConv2D):
    def __init__(self, input_size, in_channels, kernel_size):
        super().__init__(input_size, in_channels, image_channels, 3)


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, lr_mul = 1, activation=None, *args, **kwargs):

        super().__init__()
        self._linear = nn.Linear(in_dim, out_dim, *args, **kwargs)
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.activation = activation

    def forward(self, x):
        out = self._linear(x)
        return out if self.activation==None else self.activation(out)

class StyleConv2D(nn.Module):

    def __init__(self, input_size, in_channels, out_channels, kernel_size, style_dim, demodulate=True, upsample=False, downsample=False):

        super().__init__()
        self.eps = 10e-8
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.input_size = input_size
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)
        self.weight = nn.Parameter(torch.randn(1,out_channels,in_channels,kernel_size,kernel_size))
        self.blur = Blur()
        self.modulation = EqualLinear(style_dim, in_channels)
        self.upsample = stg1cl.UpSamplingBlock() if upsample else None
        self.downsample = stg1cl.PadAvgPool2d(input_size) if downsample else None
        self.padding = calc_pool2d_pad(self.input_size,kernel_size,1)
        self.upsamp_padding = calc_pool2d_pad(2*self.input_size,kernel_size,1)

    def forward(self, x, style):

        batch, in_chan, h, w = x.size()
        style = self.modulation(style).view(batch,1,in_chan,1,1)
        weight = self.weight.repeat(batch,1,1,1,1)
        weight = self.scale * self.weight * style
        demod = torch.rsqrt(weight.pow(2).sum([2,3,4],keepdim=True)+self.eps)
        weight = weight / demod
        weight = weight.view(batch*self.out_channels,self.in_channels,self.kernel_size,self.kernel_size)
        res = x

        if self.upsample is not None:
            res = self.upsample(res)
            res = self.blur(res)
            res = res.view(1,batch*self.in_channels,2*h,2*w)
            res = F.pad(res,self.upsamp_padding)
            res = F.conv2d(res,weight,stride=1,groups=batch)
            res = res.view(batch,self.out_channels,2*h,2*w)
            return res
        elif self.downsample is not None:
            res = F.pad(res, self.padding)
            res = F.conv2d(res,weight,stride=1,groups=batch)
            res = res.view(batch,self.out_channels,h,w)
            res = self.downsample(res)
            return res
        else:
            res = res.view(1,batch*self.in_channels,h,w)
            res = F.pad(res,self.padding)
            res = F.conv2d(res,weight,stride=1,groups=batch)
            res = res.view(batch,self.out_channels,h,w)
            return res

class StyleUpSkipBlock(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, style_dim, upsample=True,self_attn=False):
        super().__init__()
        self.b1, self.b2, self.b3 = nn.Parameter(torch.zeros(3,1))
        self.input_size = input_size
        self.output_size = input_size*2 if upsample else input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_dim = style_dim
        self.upsample_prev = nn.Sequential(stg1cl.UpSamplingBlock(), Blur()) if upsample else None
        self.blur = Blur() if upsample else None
        self.activation1, self.activation2, self.activation3 = [nn.LeakyReLU(leaky_relu_alpha) for _ in range(3)]
        self.noise1, self.noise2, self.noise3 = [stg1cl.Scale_B(out_channels) for _ in range(3)]
        self.conv1 = StyleConv2D(input_size, in_channels, out_channels, 3, style_dim)
        self.conv2 = StyleConv2D(self.output_size, out_channels, out_channels, 3, style_dim, upsample=upsample)
        self.conv3 = StyleConv2D(self.output_size, out_channels, out_channels, 3, style_dim)
        self.to_rgb = toRGB(self.output_size, out_channels, 3)
        self.self_attn = stg1cl.SelfAttention(out_channels) if self_attn else None

    def unit_operation(self, conv_layer, bias, noise_layer, activation, input_feature_map, style, prev_rgb, noise):
        out_ft_map = conv_layer(input_feature_map, style)
        if conv_layer.upsample is not None:
            out_ft_map = self.blur(out_ft_map)
        out_ft_map = out_ft_map + bias
        if noise is not None:
            out_ft_map = out_ft_map + noise_layer(noise)
        out_ft_map = activation(out_ft_map + bias)
        return out_ft_map

    def forward(self, input_feature_map, style, prev_rgb=None, noise=None):
        out_ft_map = self.unit_operation(self.conv1, self.b1, self.noise1, self.activation1, input_feature_map, style, prev_rgb, noise) # output feature map
        out_ft_map = self.unit_operation(self.conv2, self.b2, self.noise2, self.activation2, out_ft_map, style, prev_rgb, noise)
        out_ft_map = self.unit_operation(self.conv3, self.b3, self.noise3, self.activation3, out_ft_map, style, prev_rgb, noise)
        if self.self_attn is not None:
            out_ft_map = self.self_attn(out_ft_map)
        out_rgb = self.to_rgb(out_ft_map)
        if prev_rgb is not None:
            if self.upsample_prev is not None:
                prev_rgb = self.upsample_prev(prev_rgb)
            out_rgb = out_rgb + prev_rgb
        return out_ft_map, out_rgb if out_ft_map.size(1) > image_channels else out_rgb

class ResDownBlock(stg1cl.ResDownBlock):
    def __init__(self, input_size, in_channel, out_channel, pooling='avg', alpha=0.2, self_attn=False):
        super().__init__(input_size, in_channel, out_channel, pooling, alpha)
        modules = [EqualConv2D(input_size, in_channel, out_channel, 3),
        nn.LeakyReLU(alpha), EqualConv2D(input_size, out_channel, out_channel, 3)]
        if self_attn:
            modules = [stg1cl.SelfAttention(in_channel)] + modules
        self.self_attn = self_attn
        self.func = nn.Sequential(*modules)
        self.mod_id = EqualConv2D(input_size, in_channel, out_channel, 3)

class IntermediateG(stg1cl.IntermediateG):
    def __init__(self,n_fc, dim_latent):
        super().__init__(n_fc,dim_latent,leaky_relu_alpha)
        layers=[stg1cl.PixelNorm()]
        for _ in range(n_fc):
            layers.append(EqualLinear(dim_latent,dim_latent))
            layers.append(nn.LeakyReLU(leaky_relu_alpha))
        self.mapping = nn.Sequential(*layers)
