import torch
import torch.nn as nn
from .c_utils import wrap_weight_norm, calc_pool2d_pad, apply_wn
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.functional as F


# FC_A, PixelNorm, AdaIn, Scale_B, etc. are from SiskonEmilia's github.

#must wrap normalization function when setting default weight norm function.
default_conv_weight_norm = wrap_weight_norm(spectral_norm)
default_fc_weight_norm = wrap_weight_norm(weight_norm)

init_weight_normal_mean = 0
init_weight_normal_stddev = 0.02


def wn_Conv2D(size, *args, **kwargs) : return apply_wn(PadConv2D(size, *args, **kwargs), default_conv_weight_norm)

def wn_Linear(*args, **kwargs) : return apply_wn(nn.Linear(*args, **kwargs), default_fc_weight_norm)

def conv_module_bn(size, dim_in, dim_out, kernel_size,alpha=0.2):
    return nn.Sequential(wn_Conv2D(size, dim_in, dim_out, kernel_size),
    nn.LeakyReLU(0.2), wn_Conv2D(size, dim_out, dim_out, kernel_size), nn.LeakyReLU(0.2))


class PadConv2D(nn.Conv2d):

    def __init__(self, input_size, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.h = input_size
        self.w = input_size
        if (isinstance(input_size, list) or isinstance(input_size, tuple)):
            self.h = input_size[0]
            self.w = input_size[1]
        hpad = ((self.h - 1) % self.stride[0]) + self.dilation[0]*(self.kernel_size[0] - 1) # height pad
        wpad = ((self.w - 1) % self.stride[1]) + self.dilation[1]*(self.kernel_size[1] - 1) # width pad
        lpad = wpad // 2 # left pad
        rpad = lpad if ((wpad%2)==0) else lpad + 1 # right pad
        tpad = hpad // 2 # top pad
        bpad = tpad if ((hpad%2)==0) else tpad + 1
        self._pad = (lpad, rpad, tpad, bpad)

        #parameter initialization
        self.weight.data.normal_(init_weight_normal_mean, init_weight_normal_stddev)
        self.bias.data.zero_()


    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        return F.pad(result, self._pad)



class PadAvgPool2d(nn.AvgPool2d):

    #size, kernel_size, stride, ...
    def __init__(self, size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pad = calc_pool2d_pad(size, self.kernel_size, self.stride)

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        return F.pad(result, self._pad)



class PadMaxPool2d(nn.MaxPool2d):

    #size, kernel_size, stride, ...
    def __init__(self, size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pad = calc_pool2d_pad(size, self.kernel_size, self.stride)

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        return F.pad(result, self._pad)



class ResNetBaseBlock(nn.Module):

    def __init__(self, func, mod_id=None):

        super().__init__()

        if isinstance(func, list):
            self.func = nn.Sequential(*func)
        else:
            self.func = func

        if isinstance(mod_id, list):
            self.mod_id = nn.Sequential(*mod_id)
        else:
            self.mod_id = mod_id

    def forward(self, x, *args, **kwargs):
        if self.mod_id == None:
            identity = x
        else:
            identity = self.mod_id(x)
        fx = self.func(x, *args, **kwargs)
        return fx + identity


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon=epsilon

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


# "learned affine transform" A
class FC_A(nn.Module):
    '''
    Learned affine transform A, this module is used to transform
    midiate vector w into a style vector
    '''
    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = wn_Linear(dim_latent, n_channel * 2)
        
        # initialize weights.
        self.transform.weight.normal_(0, init_weight_normal_stddev)

        # "the biases associated with ys that we initialize to one"
        # executing .chunk(2,1), entries such that bias = 1 become to factors,
        # and entrie such that bias = 0 become to bias.
        # eo ke haet noo si bal leo ma.
        # why yi ge dweom?

        self.transform.bias.data[:n_channel] = 1
        self.transform.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style


# AdaIn (AdaptiveInstanceNorm)
class AdaIn(nn.Module):
    '''
    adaptive instance normalization
    '''
    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)
        
    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias  
        return result


#Scale Background Noise.
#inputs' size is (batch_size, n_channel).
class Scale_B(nn.Module):
    '''
    Learned per-channel scale factor, used to scale the noise
    the size of noise : (batch_size, 1)
    '''
    def __init__(self, n_channel):
        super().__init__()
        self.n_channel = n_channel
        self._scale = wn_Linear(1, n_channel, bias=False)
    
    def forward(self, noise):
        return self._scale(noise).view(-1, self.n_channel, 1, 1)



class EarlyStyleConvBlock(nn.Module):

    def __init__(self, input_size, dim_latent, n_channel, start_img_size, alpha=0.2):

        super().__init__()

        self.constant = nn.Parameter(torch.randn(1, n_channel, start_img_size, start_img_size))*init_weight_normal_stddev
        
        self.style1 = FC_A(dim_latent, n_channel)
        self.style2 = FC_A(dim_latent, n_channel)

        self.noise1 = Scale_B(n_channel)
        self.noise2 = Scale_B(n_channel)

        self.adain = AdaIn(n_channel)

        self.lrelu = nn.LeakyReLU(alpha)
        self.conv = wn_Conv2D(input_size, n_channel, n_channel, 3)


    def forward(self, latent_w, noise):

        result = self.constant.repeat(noise.size(0), 1, 1, 1).to(next(self.parameters()).device)
        result = result + self.noise1(noise)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)
        result = self.conv(result)
        result = result + self.noise2(noise)
        result = self.adain(result, self.style2(latent_w))
        result = self.lrelu(result)

        return result


class UpSamplingBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2)


class StyleConvBlock(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''
    def __init__(self, input_size, in_channel, out_channel, dim_latent, alpha=0.2):
        super().__init__()
        
        self.style1 = FC_A(dim_latent, out_channel)
        self.style2 = FC_A(dim_latent, out_channel)

        self.noise1 = Scale_B(out_channel)
        self.noise2 = Scale_B(out_channel)

        self.adain = AdaIn(out_channel)

        self.lrelu = nn.LeakyReLU(alpha)
        self.conv1 = wn_Conv2D(input_size, in_channel, out_channel, 3)
        self.conv2 = wn_Conv2D(input_size, out_channel, out_channel, 3)

    def forward(self, previous_result, latent_w, noise):

        result = self.conv1(previous_result)
        result = result + self.noise1(noise)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)
        result = self.conv2(result)
        result = result + self.noise2(noise)
        result = self.adain(result, self.style2(latent_w))
        result = self.lrelu(result)
        
        return result


class StyleUpResBlock(ResNetBaseBlock):

    def __init__(self, input_size, in_channel, out_channel, dim_latent, alpha=0.2, upsampling=True):

        super().__init__(StyleConvBlock(input_size, in_channel, out_channel, dim_latent, alpha),
        wn_Conv2D(input_size*2 if upsampling else input_size, in_channel, out_channel, 1))
        self.upsamp = None
        if upsampling:
            self.upsamp = UpSamplingBlock()
        

    def forward(self, previous_result, latent_w, noise):
        identity = previous_result
        if self.upsamp != None:
            identity = self.upsamp(identity)
        return super().forward(identity, latent_w, noise)


class ResDownBlock(ResNetBaseBlock):

    def __init__(self, input_size, in_channel, out_channel, pooling='avg', alpha=0.2):

        super().__init__(conv_module_bn(input_size, in_channel, out_channel, 3, alpha),
        wn_Conv2D(input_size, in_channel, out_channel, 1))

        self.pooling = None
        if pooling=='avg':
            self.pooling = PadAvgPool2d(input_size, 2, 2)
        if pooling=='max':
            self.pooling=PadMaxPool2d(input_size, 2, 2)

    def forward(self, x):
        fx_plus_x = super().forward(x)
        return self.pooling(fx_plus_x) if self.pooling != None else fx_plus_x


class IntermediateG(nn.Module):
    '''
    A mapping consists of multiple fully connected layers.
    Used to map the input to an intermediate latent space W.
    '''
    def __init__(self, n_fc, dim_latent, alpha=0.2):
        super().__init__()
        layers = [PixelNorm()]
        for _ in range(n_fc):
            layers.append(wn_Linear(dim_latent, dim_latent))
            layers.append(nn.LeakyReLU(alpha))
            
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, latent_z):
        latent_w = self.mapping(latent_z)
        return latent_w



class MiniBatchStdLayer(nn.Module):

    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    
    def forward(self, x):

        size = x.size()
        subGroupSize = min(size[0], self.group_size)
        if size[0] % subGroupSize != 0:
            subGroupSize = size[0]
        G = size[0] // subGroupSize
        if subGroupSize > 1:
            y = x.view(-1, subGroupSize, *size[1:]) # G x subGroupSize x C x H x W
            stddev = torch.sqrt(torch.var(y, dim=1)+10e-8) # G x C x H x W
            std_mean = torch.mean(stddev, [1,2,3], keepdim=True) # G x 1 x 1 x 1
            std_mean = std_mean.repeat(subGroupSize, 1, *size[2:]) # B x 1 x H x W
        else:
            std_mean = torch.zeros(size[0], 1, *size[2:], device=x.device)

        return torch.cat([x,std_mean], dim=1) # B x (C+1) x H x W



class _SelfAttention(nn.Module):

    def __init__(self, in_channel, attn_proj_factor=8):

        #attn_proj_factor = attention projection factor.
        super().__init__()
        self.channels = in_channel
        
        self.query_conv = nn.Conv2d(in_channels = in_channel, out_channels = in_channel // attn_proj_factor, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_channel, out_channels = in_channel // attn_proj_factor , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_channel, out_channels = in_channel , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.query_conv.weight.data.normal_(init_weight_normal_mean, init_weight_normal_stddev)
        self.key_conv.weight.data.normal_(init_weight_normal_mean, init_weight_normal_stddev)
        self.value_conv.weight.data.normal_(init_weight_normal_mean, init_weight_normal_stddev)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
                c = in_channel // attn_proj_factor
        """
        m_batchsize,C, w, h = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,w*h).permute(0,2,1) # (B X N X c) by .permute(=transpose)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,w*h) # B X c x N
        energy =  torch.bmm(proj_query,proj_key) # inner product.
        attention = self.softmax(energy) # B X (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,w*h) # B X C X N

        out = torch.bmm(proj_value,attention) # (B X C X N). In the original source code, attention is attention.permute(0,2,1)(= transpose). why?
        out = out.view(m_batchsize,C,w,h)
        
        out = self.gamma*out + x
        return out, attention

# below is self-attention layer that only return final outputs.
class SelfAttention(_SelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return super().forward(x)[0]


