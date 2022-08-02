import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from model import Generator, Discriminator
import wandb
from PIL import Image
import numpy as np

import yaml
import os


def load_config(config_file_path: str):
    with open(config_file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

device = 'cuda' if torch.cuda.is_available() else "cpu"

print(f"Now running on {device}")

config = load_config(os.path.join(os.path.dirname(__file__), "config/train_config.yaml"))

dataset = dset.ImageFolder(root=config['data']['dataroot'],
                            transform=transforms.Compose([
                                transforms.Resize((config['image']['image_size'], config['image']['image_size'])),
                                transforms.ToTensor(),
                            ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['hyperparams']['batch_size'],
                                        shuffle=True, num_workers=config['hyperparams']['workers'])


netG = Generator(config['image']['nz'], config['image']['ngf'], config['image']['nc']).apply(weights_init)
netD = Discriminator(config['image']['nc'], config['image']['ndf']).apply(weights_init)

netG = netG.to(device)
netD = netD.to(device)

criterion = nn.BCELoss()
# Create batch of latent vectors that we will use to visualize
#  the progression of the generator

fixed_noise=torch.randn(config['hyperparams']['batch_size'], config['image']['nz'], 1, 1, device=device)

real_label = 1.
fake_label = 0.
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=config['hyperparams']['lr'], betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=config['hyperparams']['lr'], betas=(0.5, 0.999))

# Training Loop

# Lists to keep track of progress
num_epochs = config['hyperparams']['num_epochs']
img_list = []
G_losses = []
D_losses = []
iters = 0

wandb.init(project="GAN_cup", entity="junshickyoon")

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch

        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, config['image']['nz'], 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 30 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            wandb.log({
                "Loss_D":errD.item(),
                "Loss_G":errG.item(),
                "D(x)":D_x,
                "D(G(z))_1": D_G_z1,
                "D(G(z))_2": D_G_z2
            })
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        #     with torch.no_grad():
        #         fake = netG(fixed_noises[0]).detach().cpu()
        #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
           

        # iters += 1

    if epoch % 20 == 0:
        with torch.no_grad():

            fake = netG(fixed_noise).detach().cpu().numpy()
            fake = np.transpose(fake, (0, 2, 3, 1))
            img_list = [fake[i] for i in range(10)]

        wandb.log({
            "Epoch": epoch,
            "Generated imgs": [wandb.Image(image) for image in img_list]
        })

        if epoch % 100 == 0:
            # save models
            torch.save(netD.state_dict(), os.path.join(config['data']['save_path'], 'checkpoints', f'D_epoch{epoch+1:03}.pth'))
            torch.save(netG.state_dict(), os.path.join(config['data']['save_path'], 'checkpoints', f'G_epoch{epoch+1:03}.pth'))
        
        # save checkpoints
        torch.save(netD.state_dict(), os.path.join(config['data']['save_path'], 'checkpoints', f'D_checkpoint.pth'))
        torch.save(netG.state_dict(), os.path.join(config['data']['save_path'], 'checkpoints', f'G_checkpoint.pth'))

