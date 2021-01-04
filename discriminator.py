
from custom_layers import EqualConv2D, ResDownBlock, image_channels, leaky_relu_alpha, EqualLinear
from stylegan1.custom_layers import MiniBatchStdLayer
import torch.nn as nn


class SG2_Discriminator(nn.Module):

    def __init__(self, image_size, img_channels, pooling='avg', last_fc_double=False, insert_sa_layers=[]):
        super().__init__()
        core = []
        self.image_size = image_size
        self.img_channels = img_channels
        self.from_rgb = ResDownBlock(image_size, image_channels, img_channels[0],None,leaky_relu_alpha,False)
        self.last_fc_double = last_fc_double
        self.insert_sa_layers = insert_sa_layers
        self.pooling = pooling
        input_size_buf = image_size
        for i in range(len(img_channels)-1):
            core.append(ResDownBlock(input_size_buf,img_channels[i],img_channels[i+1],pooling,leaky_relu_alpha,(i in insert_sa_layers)))
            input_size_buf = input_size_buf // 2
        core = core + [MiniBatchStdLayer(),
        ResDownBlock(input_size_buf,img_channels[-1]+1,img_channels[-1],None,leaky_relu_alpha,(len(img_channels)-1 in insert_sa_layers)),
        EqualConv2D(input_size_buf,img_channels[-1],img_channels[-1],input_size_buf),nn.Flatten(),nn.LeakyReLU(leaky_relu_alpha)]
        core.append(EqualLinear((input_size_buf**2)*img_channels[-1],img_channels[-1],activation=nn.LeakyReLU(leaky_relu_alpha)))
        if last_fc_double:
            core.append(EqualLinear(img_channels[-1],img_channels[-1],activation=nn.LeakyReLU(leaky_relu_alpha)))
        core.append(EqualLinear(img_channels[-1],1,activation=nn.LeakyReLU(leaky_relu_alpha)))
        self.core = nn.ModuleList(core)

    def state_dict(self):
        sdict = {}
        sdict['model'] = super().state_dict()
        sdict['arguments'] = {'image_size' : self.image_size, 'img_channels' : self.img_channels,
        'pooling' : self.pooling, 'last_fc_double' : self.last_fc_double, 'insert_sa_layers' : self.insert_sa_layers}
        return sdict

    def load_from_state_dict(sdict):
        model = SG2_Discriminator(**sdict['arguments'])
        model.load_state_dict(sdict['model'])
        return model

    def forward(self,x):
        out = self.from_rgb(x)
        for layer in self.core:
            out = layer(out)
        return out

if __name__=='__main__':
    import stylegan1.c_dset as dset
    dis = SG2_Discriminator(256,[4,8,16,32,32,64,128,256],insert_sa_layers=[4])
    paimon = dset.create_image_loader_from_path('paimon/',256,1)
    x = next(iter(paimon))[0]
    print(dis(x))