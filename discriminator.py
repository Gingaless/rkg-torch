
from prog_base_model import ProgressiveBaseModel
import torch
from torch import nn
from custom_layers import ResDownBlock, conv_module_bn, wn_Linear, default_conv_weight_norm, MiniBatchStdLayer, SelfAttention, PadAvgPool2d, PadMaxPool2d


class PGSB_Discriminator(ProgressiveBaseModel):

    def __init__(self, start_img_size, transition_channels, insert_sa_layers=None, pooling='avg', last_fc_double=False, img_channel=3, alpha=0.2):

        super().__init__(start_img_size, transition_channels)

        self.from_rgb_old = conv_module_bn(start_img_size, img_channel, transition_channels[0], 1)
        self.from_rgb_new = self.from_rgb_old
        self.new_blocks = None
        self.pooling = pooling
        self.alpha = alpha
        self.insert_sa_layers = insert_sa_layers
        self.img_channel = img_channel
        self.alpha = alpha

        self.core_blocks = nn.Sequential(MiniBatchStdLayer(), conv_module_bn(start_img_size, transition_channels[0]+1,
        transition_channels[0], 3), default_conv_weight_norm(nn.Conv2d(transition_channels[0],transition_channels[0], start_img_size)),
        nn.Flatten(), nn.LeakyReLU(alpha), nn.Sequential(wn_Linear(transition_channels[0], transition_channels[0]), wn_Linear(transition_channels[0], 1)) 
        if last_fc_double else wn_Linear(transition_channels[0], 1))
        
        if (isinstance(insert_sa_layers, list) and insert_sa_layers[0]):
            self.core_blocks = nn.Sequential(self.new_blocks, *self.core_blocks)

    
    def get_pool_layer(self):
        if self.pooling=='max':
            return PadMaxPool2d(self.current_img_size, 2, 2)
        else:
            return PadAvgPool2d(self.current_img_size, 2, 2)


    def extend(self):

        old_new_block = self.new_blocks
        previous_channels = self.transition_channels[self.transition_step]
        super().extend()
        current_channels = self.transition_channels[self.transition_step]
        self.from_rgb_old = nn.Sequential(self.get_pool_layer(), self.from_rgb_new)
        self.from_rgb_new = conv_module_bn(self.current_img_size, 3, current_channels, 3, self.alpha)
        self.new_blocks = ResDownBlock(self.current_img_size, current_channels, previous_channels, self.pooling, self.alpha)
        
        if self.insert_sa_layers[self.transition_step]:
            self.new_blocks = nn.Sequential(SelfAttention(current_channels),self.new_blocks)
        if old_new_block != None:
            self.core_blocks = nn.Sequential(old_new_block, *self.core_blocks.children())
            
    def forward(self, image):
        y = image
        if self.transition_step == 0:
            y = self.from_rgb_new(y)
        else:
            y_old = self.from_rgb_old(y)
            y_new = self.from_rgb_new(y)
            y_new = self.new_blocks(y_new)
            y = self.transition_value*y_new + (1 - self.transition_value)*y_old
        
        y = self.core_blocks(y)
        return y

    def state_dict(self):
        
        start_image_size = self.current_img_size // (2**self.transition_step)
        last_fc_double = isinstance(self.core_blocks, nn.Sequential)
        dct = super().state_dict()
        dct["arguments"] = {
        "start_img_size" : start_image_size, 
        "transition_channels" : self.transition_channels, 
        "insert_sa_layers" : self.insert_sa_layers, 
        "img_channel" : self.img_channel,
        "pooling" : self.pooling, 
        "last_fc_double" : last_fc_double, 
        "alpha" : self.alpha }
        return dct


if __name__ == '__main__':

    x = torch.ones(1,3,4,4)
    d = PGSB_Discriminator(4,[512,256,128,64,32,16,8,4], 
    [False,False,False,True,True,True,True,True], last_fc_double=True)
    print(d)
    print(d(x))
    d.extend()
    print(d)
    x = torch.ones(1,3,8,8)
    d.transition_value = 1.0
    print(d(x).size())
    #print(d.state_dict())
    from generator import PGSB_Generator
    import numpy as np
    g = PGSB_Generator(4,512,4,[512,256,128,64,32,16,8,4],[False,False,False,True,False,False,False,False])
    g.extend()
    x = np.random.normal(0, 0.02, (1,512))
    noise = np.random.normal(0, 0.5, (1,1))
    x = torch.Tensor(x)
    noise = torch.Tensor(noise)
    print(d(g(x,noise)))