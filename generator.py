from custom_layers import IntermediateG, StyleUpSkipBlock, image_channels
from stylegan1.custom_layers import init_weight_normal_stddev
import torch
import torch.nn as nn


class SG2_Generator(nn.Module):
    
    def __init__(self, image_size, img_channels, dim_latent, n_fc, insert_sa_layers=[]):

        if len(img_channels) < 2:
            raise Exception('the length of the image channel list should be bigger than 1')
        super().__init__()
        self.image_size = image_size
        self.img_channels = img_channels
        self.dim_latent = dim_latent
        self.n_fc = n_fc
        self.insert_sa_layers = insert_sa_layers
        self.style_conv_blocks = nn.ModuleList()
        img_size_buf = image_size
        for i in reversed(range(1,len(img_channels))):
            img_size_buf = img_size_buf // 2
            self.style_conv_blocks.insert(0,StyleUpSkipBlock(img_size_buf, img_channels[i-1],img_channels[i],dim_latent,upsample=True, self_attn=(i in insert_sa_layers)))
        self.constant = nn.Parameter(torch.randn(1,img_channels[0],img_size_buf,img_size_buf))*init_weight_normal_stddev
        self.intermediate = IntermediateG(n_fc, dim_latent)

    
    def state_dict(self):
        sdict = {}
        sdict['model'] = super().state_dict()
        sdict['arguments'] = {'image_size' : self.image_size, 'img_channels' : self.img_channels,
        'dim_latent' : self.dim_latent, 'n_fc' : self.n_fc, 'insert_sa_layers' : self.insert_sa_layers}
        return sdict

    def load_from_state_dict(sdict):
        model = SG2_Generator(**sdict['arguments'])
        model.load_state_dict(sdict['model'])
        return model


    def forward(self,latent_z,noise=None,style_mix_steps=[],return_dlatents=False):
        latent_w = []
        if not isinstance(latent_z, list):
            latent_z = [latent_z]
        for z in latent_z:
            latent_w.append(self.intermediate(z))
        out = self.constant.repeat(latent_z[0].size(0),1,1,1).to(next(self.parameters()).device)
        prev_rgb = None
        for i, module in enumerate(self.style_conv_blocks):
            if module.out_channels > image_channels:
                out, prev_rgb = module(out,latent_w[1] if i in style_mix_steps else latent_w[0],prev_rgb,noise)
            else:
                return module(out,latent_w[1] if i in style_mix_steps else latent_w[0],prev_rgb,noise)
        if return_dlatents:
            return prev_rgb, latent_w
        else:
            return prev_rgb