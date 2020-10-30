
import torch
from torch import nn, optim
from generator import PGSB_Generator
from discriminator import PGSB_Discriminator
import c_utils
from c_utils import load_model
from c_dset import create_image_loader_from_path, show_images_from_tensors, save_images_from_tensors


device_type = "gpu" if torch.cuda.is_available() else "cpu"
device = c_utils.set_device(device_type)

dataroot = "/home/shy/kiana_resized"

save_checkpoint_path = "rkg-checkpoint.tar"

batch_size = 32

n_images_eval = 64

start_image_size = 4

# number of image channels
nc = 3

# size of latent vectors.
nz = 512

transition_channels = [512,256,128,64,32,16,8,8]

insert_sa_layers = [False,False,False,True,False,False,False,False]

num_epochs = 10

lr_g = 0.0002 # learning rate of generator

lr_d = 0.0002 # learning rate of discriminator

n_fc = 6 # number of full connected layers in intermediate generator.

criterion = nn.MSELoss()

beta1 = 0.5

beta2 = 0.999

start_step = 1 # start_step.
current_step = start_step

n_critic = 2

step_mix_style = 3 # the minimum step where to apply stlye mixing.

n_show_loss = 1 # loss will be recorded every n_show_loss iterations.

pooling = 'avg'

last_fc_double = True

total_epochs = [0]*len(transition_channels)
total_iters = [0]*len(transition_channels)
total_G_losses = [[]]*len(transition_channels)
total_D_losses = [[]]*len(transition_channels)
netD = None
netG = None
optim_G = None
optim_D = None

eval_clip = (-1.5, 1.5)

transition_rate = 1.0 / 200

min_fake_size = 32
max_fake_size = 128

save_checkpoint_period = 1



def save_checkpoint():

    s_dict = { 'discriminator' : netD.state_dict(), 
    'generator' : netG.state_dict(), 'total_epochs' : total_epochs, 
    'total_iters' : total_iters, 'total_G_losses' : total_G_losses, 
    'total_D_losses' : total_D_losses, 'n_critic' : n_critic, 
    'optimizerD' : optim_D.state_dict(), 'optimizerG' : optim_G.state_dict(), 
    'step_mix_style' : step_mix_style, 'beta1' : beta1, 'beta2' : beta2, 
    'criterion' : criterion, 'lr_d' : lr_d, 'lr_g' : lr_g, 'step' : current_step }
    torch.save(s_dict, save_checkpoint_path)

def init_optims():
    global optim_D, optim_G
    if optim_D==None:
        optim_D = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
    if optim_G==None:
        optim_G = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))

def init_models():
    global netD, netG
    if netD==None:
        netD = PGSB_Discriminator(start_image_size, transition_channels, insert_sa_layers, pooling, last_fc_double, nc).to(device)
        [netD.extend() for _ in range(start_step)]
    if netG==None:
        netG = PGSB_Generator(n_fc, nz, start_image_size, transition_channels, insert_sa_layers, nc).to(device)
        [netG.extend() for _ in range(start_step)]
    set_device(device_type)


def set_transition_value(value):
    netD.transition_value = value
    netG.transition_value = value


def extend():
    global current_step
    netG.extend()
    netD.extend()
    current_step += 1

def set_device(_device_type):
  global device_type, netG, netD
  device_type = _device_type
  device = c_utils.set_device(device_type)
  if _device_type=='gpu':
    if netG!=None:
      netG = netG.cuda()
    if netD!=None:
      netD = netD.cuda()


def update_transition_value():
    netG.increase_transition_value(transition_rate)
    netD.increase_transition_value(transition_rate)


def load_checkpoint(path=save_checkpoint_path):

    s_dict = torch.load(path, map_location=device)
    global netG, netD, total_epochs, total_iters, total_G_losses
    global total_D_losses, n_critic, optim_D, optim_G, step_mix_style
    global beta1, beta2, criterion, lr_d, lr_g, current_step
    netG = load_model(PGSB_Generator, s_dict['generator']).to(device)
    netD = load_model(PGSB_Discriminator, s_dict['discriminator']).to(device)
    set_device(device_type)
    total_epochs, total_iters = s_dict['total_epochs'], s_dict['total_iters']
    total_G_losses, total_D_losses = s_dict['total_G_losses'], s_dict['total_D_losses']
    n_critic, beta1, beta2 = s_dict['n_critic'], s_dict['beta1'], s_dict['beta2']
    step_mix_style, criterion = s_dict['step_mix_style'], s_dict['criterion']
    lr_d, lr_g, current_step = s_dict['lr_d'], s_dict['lr_g'], s_dict['step']
    optim_D = optim.Adam(netD.parameters())
    optim_G = optim.Adam(netG.parameters())
    optim_D.load_state_dict(s_dict['optimizerD'])
    optim_G.load_state_dict(s_dict['optimizerG'])


def generate_latent_noise(b_size=batch_size, clip_value=None):
    out1 = torch.randn(b_size, nz, device=device)
    out2 = torch.randn(b_size, 1, device=device)
    if isinstance(clip_value, tuple):
        return torch.clamp(out1, *clip_value), torch.clamp(out2, *clip_value)
    else:
        return out1, out2


def train_loop(load_path=None, eval_G=1, save_fake_prefix='rkg-'):

    global total_G_losses, total_D_losses
    global total_epochs, total_iters

    real_label = 1
    fake_label = 0

    if load_path==None:
        init_models()
        init_optims()
    elif len(load_path) > 0:
        load_checkpoint(load_path)
    else:
        load_checkpoint()

    dataloader = create_image_loader_from_path(dataroot, netD.current_img_size, batch_size)
    
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):

            netD.zero_grad()
            real_sample = data[0].to(device)

            b_size = real_sample.size(0)

            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_sample).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            latent_z_noise = generate_latent_noise(b_size)
            fake = netG(*latent_z_noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1) # detach -> stopping tracing gradient record.
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake

            optim_D.step()

            if i % n_critic == 0:
                netG.zero_grad()
                #fake = netG(*generate_latent_noise())
                output = netD(fake).view(-1)
                label.fill_(real_label)
                errG = criterion(output, label)
                errG.backward(retain_graph=True)
                D_G_z2 = output.mean().item()
                optim_G.step()
                total_G_losses[current_step].append(errG.item())

            total_iters[current_step] += 1
            total_D_losses[current_step].append(errD.item())
            
            update_transition_value()

            if ((i+1) % n_show_loss == 0) or (i == len(dataloader)-1):
                 print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' 
                 % (epoch+1, num_epochs, i+1, len(dataloader), 
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        total_epochs[current_step] += 1

        if ((epoch+1) % eval_G == 0) or (epoch == num_epochs - 1):
            with torch.no_grad():
                fake = netG(*generate_latent_noise(n_images_eval, clip_value=eval_clip)).detach().cpu()
                show_images_from_tensors(fake, min_size=min_fake_size, max_size=max_fake_size)
                if save_fake_prefix != None:
                    save_images_from_tensors(fake, save_fake_prefix + str(epoch) + ".jpg", 
                    min_size=min_fake_size, max_size=max_fake_size)

        if save_checkpoint_period > 0 :
            if (epoch % save_checkpoint_period == 0) or (epoch == num_epochs-1):
                save_checkpoint()

        