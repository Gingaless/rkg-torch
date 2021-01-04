from prog_base_model import ProgressiveBaseModel
import torch
from torch import nn
from custom_layers import wn_Conv2D, EarlyStyleConvBlock, SelfAttention, StyleUpResBlock, IntermediateG, UpSamplingBlock


#Progressively Growing Style-Based Generator
class PGSB_Generator(ProgressiveBaseModel):

    #start_img_size must be a integer.
    def __init__(self, n_fc, dim_latent, start_img_size, transition_channels, insert_sa_layers=None, img_channel=3):

        super().__init__(start_img_size, transition_channels)
        self.dim_latent = dim_latent
        self.insert_sa_layers = insert_sa_layers
        self.img_channel =img_channel
        self.to_rgb_new = wn_Conv2D(start_img_size, transition_channels[0],self.img_channel,1) # to_rgb is the last layer.
        self.to_rgb_old = self.to_rgb_new

        modules = [EarlyStyleConvBlock(start_img_size, dim_latent, transition_channels[0], start_img_size)]
        if (isinstance(insert_sa_layers, list) and insert_sa_layers[0]):
            modules.append(SelfAttention(transition_channels[0]))
        self.intermediate = IntermediateG(n_fc, dim_latent)
        self.early_blocks = nn.ModuleList(modules)
        self.core_blocks = nn.ModuleList([])
        self.new_blocks = None

    def extend(self):
        in_channel = self.transition_channels[self.transition_step]
        super().extend()
        out_channel = self.transition_channels[self.transition_step]
        self.to_rgb_old = nn.Sequential(self.to_rgb_new, UpSamplingBlock())
        self.to_rgb_new = wn_Conv2D(self.current_img_size, out_channel, self.img_channel,1)
        if self.new_blocks != None:
            self.core_blocks.append(self.new_blocks)
        new_modules = [StyleUpResBlock(self.current_img_size, in_channel, out_channel, self.dim_latent)]
        if self.insert_sa_layers[self.transition_step]:
            new_modules.append(SelfAttention(out_channel))
        self.new_blocks = nn.ModuleList(new_modules)


    def upres_attn(self, modules, previous_result, latent_z, noise):
        result = previous_result
        for f in modules:
            if (isinstance(f, EarlyStyleConvBlock)):
                result = f(latent_z, noise)
            elif (isinstance(f, StyleUpResBlock)):
                result = f(result, latent_z, noise)
            else:
                result = f(result)
        return result

    def forward(self, latent_z, noise, style_mix_steps=[]):

        if (isinstance(latent_z, list)):
            latent_w = [self.intermediate(z) for z in latent_z]
        else:
            latent_w = [self.intermediate(latent_z)]
        
        y = self.upres_attn(self.early_blocks, None, latent_w[0], noise)

        for i in range(len(self.core_blocks)):
            y = self.upres_attn(self.core_blocks[i], y, latent_w[1] if (i+1) in style_mix_steps else latent_w[0], noise)

        if self.new_blocks == None:
            return self.to_rgb_old(y)
        else:
            y_old = self.to_rgb_old(y)
            y_new = self.upres_attn(self.new_blocks, y, latent_w[1] if (self.transition_step-1) in style_mix_steps else latent_w[0], noise)
            y_new = self.to_rgb_new(y_new)
            return self.transition_value*y_new + (1 - self.transition_value)*y_old

    def new_parameters(self):
        new_paramters = list(self.new_blocks.parameters()) + list(self.to_rgb_new.parameters())
        return new_paramters


    def device(self):
        return next(self.parameters()).device

    
    def state_dict(self):
        
        n_fc = (len(self.intermediate.mapping) - 1)//2
        start_image_size = self.current_img_size // (2**self.transition_step)
        dim_latent = self.intermediate.mapping[1].in_features
        dct = super().state_dict()
        dct["arguments"] = {"n_fc" : n_fc, 
        "dim_latent" : dim_latent, 
        "start_img_size" : start_image_size, 
        "transition_channels" : self.transition_channels, 
        "insert_sa_layers" : self.insert_sa_layers, 
        "img_channel" : self.img_channel}
        return dct


if __name__=='__main__':

    g = PGSB_Generator(4,512,4,[512,256,128,64,32,16,8,4],[False,False,False,True,False,False,False,False])
    g.eval()
    print(g(torch.ones(1,512)*0.001, torch.zeros(1)))
    #print(g.state_dict())
    [g.extend() for i in range(3)]
    print(g(torch.ones(1,512), torch.zeros(1)))