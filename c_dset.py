import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


def load_images_from_path(root_path, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size), transforms.CenterCrop(image_size), 
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
        transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])
    return dset.ImageFolder(root_path, transform=transform)


def create_image_loader(dataset, batch_size, shuffle=True, num_workers=2):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
    shuffle=shuffle, num_workers=num_workers)

def create_image_loader_from_path(root_path, image_size, batch_size, shuffle=True, num_workers=2):
    return create_image_loader(load_images_from_path(root_path, image_size), batch_size, shuffle, num_workers)


def plt_images_from_tensors(image_tensors, grid_size=(8,8), min_size=0, max_size=0, title="images", padding=2):
    
    size = image_tensors.size(-1)
    if max_size > 0:
        size = min(size, max_size)
    if min_size > 0:
        size = max(size, min_size)
    if size!=image_tensors.size(-1):
        image_tensors = F.interpolate(image_tensors, size=size)
    
    
    np_image_set = np.transpose(vutils.make_grid(image_tensors[:(reduce(lambda a,b:a*b, grid_size))], 
    padding=padding, nrow=grid_size[1],normalize=True).cpu(), (1,2,0))
    dpi = 100.0#dot per inch
    plt.figure(figsize=tuple(np.array(np_image_set.size()[:2])*size/dpi))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np_image_set)


def show_images_from_tensors(image_tensors, grid_size=(8,8), min_size=0, max_size=0, title="images", padding=2):

    plt_images_from_tensors(image_tensors, grid_size, min_size, max_size, title, padding)
    plt.show()

def save_images_from_tensors(image_tensors, path, grid_size=(8,8), min_size=0, max_size=0, title="images", padding=2):

    plt_images_from_tensors(image_tensors, grid_size, min_size, max_size, title, padding)
    plt.savefig(path)
    plt.clf()
    plt.close()


if __name__=='__main__':
    size=4
    import sys
    if len(sys.argv) > 1:
        size=int(sys.argv[1])
    x=torch.ones(1,3,size,size)
    show_images_from_tensors(x, (1,1), 32, 128)