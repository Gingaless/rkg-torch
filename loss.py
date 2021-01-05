
import sys
import torch
from torch.nn import functional as F




class __loss():

    def __init__(self):
        pass

    def D(self, train, real_sample):
        if train.optimizer['optimizer type']=='lsgan':
            return self.lsgan_D(train, real_sample)
        if train.optimizer['optimizer type']=='logistic_loss':
            return self.logistic_D(train, real_sample)

    def G(self, train):
        if train.optimizer['optimizer type']=='lsgan':
            return self.lsgan_G(train)
        if train.optimizer['optimizer type']=='logistic_loss':
            return self.logistic_G(train)

    def lsgan_D(self, train, real_sample):

        batch_size = real_sample.size(0)
        real_label = 1.0
        fake_label = 0.0

        labels = torch.full((batch_size,), real_label, dtype=torch.float, device=train.device)
        output = train.out_D(real_sample).view(-1)
        loss_D_real = train.loss['loss criterion of D'](output, labels)

        labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=train.device)
        output = train.out_D(train.out_G(batch_size).detach()).view(-1)
        loss_D_fake = train.loss['loss criterion of D'](output, labels)

        return (loss_D_real + loss_D_fake)*0.5

    def lsgan_G(self, train):

        real_label = 1.0

        labels = torch.full((train.batch_size,), real_label, dtype=torch.float, device=train.device)
        fakes = train.out_G(train.batch_size)
        output = train.out_D(fakes).view(-1)
        loss_G = train.loss['loss criterion of G'](output, labels)

        return loss_G

    def logistic_D(self, train, real_sample):

        batch_size = real_sample.size(0)
        fakes = train.out_G(batch_size).detach()
        out_real = train.out_D(real_sample).view(-1)
        out_fake = train.out_D(fakes).view(-1)
        loss_real = F.softplus(out_real)
        loss_fake = F.softplus(-out_fake)
        
        return loss_real.mean() + loss_fake.mean()

    def logistic_G(self, train):

        fakes = train.out_G(train.batch_size)
        output = train.out_D(fakes).view(-1)
        loss_G = F.softplus(-output)

        return loss_G.mean()



sys.modules[__name__] = __loss()