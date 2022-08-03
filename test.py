import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.datasets as dset
import numpy as np

from model import Generator, Discriminator
import yaml
import wandb
from matplotlib import pyplot as plt
from copy import deepcopy

# image_size = 28
# batch_size = 256
# nc = 1 # Number of channels in the training images. For color images this is
# nz = 128 # Size of z latent vector (i.e. size of generator input)
# ngf = 64 # Size of feature maps in generator
# ndf = 64 # Size of feature maps in discriminator
# alpha = 0.1 # loss ratio of Residual Loss, Discriminator Loss
# save_fp = './results/exp06/'
# netG_checkpoint = './results/exp06/checkpoints/G_epoch049.pth'
# netD_checkpoint = './results/exp06/checkpoints/D_epoch049.pth'


def load_config(config_file_path: str):
    with open(config_file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

config = load_config(os.path.join(os.path.dirname(__file__), "config/test_config.yaml"))
# make dirs
# get model
netD = Discriminator(config['nc'], config['ndf'])
netG = Generator(config['nz'], config['ngf'], config['nc'])

# load checkpoints
netD.load_state_dict(torch.load(config['netD_checkpoint']))
netG.load_state_dict(torch.load(config['netG_checkpoint']))

transform = transforms.Compose([transforms.Resize(config['image_size']),
                                transforms.ToTensor()])

dataset = dset.ImageFolder(root=config['dataroot'],
                            transform=transform
                            )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'])

# Decide which device we want to run on
device = 'cuda' if torch.cuda.is_available() else 'cpu'
netD.to(device)
netG.to(device)

# set loss function
criterion = torch.nn.MSELoss()

# get one real image
real, label = next(iter(dataloader))
real = real.to(device)

# get one random z vector
z_vector = torch.randn(1, config['nz'], 1, 1, device=device, requires_grad=True)

# set optimizer to update z vector
optimizer = torch.optim.Adam([z_vector])

wandb.init(project="GAN_cup_inference", entity="junshickyoon")

r_image = real.detach().cpu().numpy()
r_image = np.transpose(r_image, (0, 2, 3, 1))[0]

for i in tqdm(range(10001)):

    # generate fake from z
    fake = netG(z_vector)

    # get feature from discriminator
    f_real = netD(real).view(-1)
    f_fake = netD(fake).view(-1)

    # get loss
    lossR = criterion(real, fake)
    lossD = criterion(f_real, f_fake)
    loss = (1 - config['alpha']) * lossR + config['alpha'] * lossD

    # update z vector
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # show, save real, fake image
    if i % 100 == 0:
        
        wandb.log({
            "Epoch" : i,
            "LossR": loss.item()
        })


    if i == 10000:
        
        with torch.no_grad():

            f_image = fake.detach().cpu().numpy()
            f_image = np.transpose(f_image, (0, 2, 3, 1))


# show image
print(f'Anomaly Score: {loss.item():.3f}')