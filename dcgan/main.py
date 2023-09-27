# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

#%matplotlib inline
import argparse
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

from generator import Generator, GeneratorBN
from discriminator import Discriminator, DiscriminatorBN

from IPython.display import HTML

# Energy consumption measurement
from carbontracker.tracker import CarbonTracker
from carbontracker import parser

# Saving lists
import cv2


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Parser arguments
parser = argparse.ArgumentParser(description="Setup model")
# Run the script with --batchnorm for batch normalization enabled and --no_batchnorm to disable
parser.add_argument('--batchnorm', action='store_true', help="Enable batch normalization?")
parser.add_argument('--no_batchnorm', dest='batchnorm', action='store_false', help="Disable batch normalization")

args = parser.parse_args()

dataroot = "data/celeba"
"""Root directory for dataset"""

workers = 2
"""Number of workers for dataloader"""

batch_size = 128
"""Batch size during training"""

image_size = 64
"""Spatial size of training images. All images will be resized to this size using a transformer."""

nc = 3
"""Number of channels in the training images. For color images, this is 3"""

nz = 100
"""Size of z latent vector (i.e., size of generator input)"""

ngf = 64
"""Size of feature maps in generator"""

ndf = 64
"""Size of feature maps in discriminator"""

num_epochs = 1
"""Number of training epochs"""

lr = 0.0002
"""Learning rate for optimizers"""

beta1 = 0.5
"""Beta1 hyperparameter for Adam optimizers"""

ngpu = 1
""" Number of GPUs available. Use 0 for CPU mode."""

batchnorm_enabled = args.batchnorm
""" Boolean that decides if batch normalization is included or not"""

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                        transform=transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers)

# Determine the total number of batches in the dataloader
total_batches = len(dataloader)

# Calculate the total number of images in your dataset
total_images = total_batches * batch_size

# Print interval for loss and stats
print_interval = 50

# Initialize variables for early stopping
improvement_threshold = 0.1  # Define the improvement threshold (10%)
consecutive_no_improvement = 0  # Counter to keep track of consecutive iterations with no improvement
window_size = 500  # Set the window size for calculating average loss

def main():
    start_time = time.time()

    # Energy consumption measurement
    tracker = CarbonTracker(epochs=num_epochs)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    netG = GeneratorBN(ngpu, nz, nc, ngf).to(device) if batchnorm_enabled else Generator(ngpu, nz, nc, ngf).to(device)
    #netG = Generator(ngpu, nz, nc, ngf).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = DiscriminatorBN(ngpu, nz, nc, ngf).to(device) if batchnorm_enabled else Discriminator(ngpu, nz, nc, ngf).to(device)
    #netD = Discriminator(ngpu, nz, nc, ngf).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training loop

    # Variables for tracking loss
    prev_avg_loss = float('inf')  # Initialize with a large value
    # Initialize variables for early stopping
    improvement = float('inf')  # Initialize with a large value
    image_counter = 0  # Counter to keep track of the number of images processed

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):

        # Initialize flag variable for early stopping
        stop_training = False
        if stop_training:
            break  # Exit the epoch processing loop

        tracker.epoch_start()
        # Initialize a list to store losses for this epoch
        epoch_losses = []
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
            noise = torch.randn(b_size, nz, 1, 1, device=device)
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
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Append the current loss to the epoch_losses list
            epoch_losses.append(errG.item())

            # Update the total number of images processed
            image_counter += 1

            # Check how the generator is doing by saving G's output on fixed_noise
            if (image_counter % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            # Check the stopping condition based on improvement in avg_loss
            if i % window_size == 0:

                # Calculate the average loss for the current window
                avg_loss = sum(epoch_losses[-window_size:]) / min(window_size, len(epoch_losses))

                # Print the improvement for this epoch, remove later
                print(f'Iteration [{i+1}/{total_images}]')
                print(f'Avg loss [{avg_loss:.4f}] - Previous avg loss: {prev_avg_loss:.4f}')

                if avg_loss > ((1 - improvement_threshold) * prev_avg_loss):
                    print("Stopping training due to insufficient improvement.")
                    stop_training = True  # Set the flag to stop training
                    break  # Exit the training loop

                prev_avg_loss = avg_loss  # Update the previous average loss
                


            # Output training stats at the defined print interval
            if (image_counter % print_interval == 0):
                # Output training stats
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\t'
                    f'Processed [{image_counter}/{total_images}] \t'
                    f'Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\t'
                    f'D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}\t'
                    )
                

        tracker.epoch_end()

    tracker.stop()
    end_time = time.time()
    print('DONE TRAINING')
    print("total time spent training:" +  str((end_time - start_time) / 60) + " minutes.")




    # Saving the model and output files
    # This is required when we want to save the images that are generated by the generator as a img file
    to_pil_image = transforms.ToPILImage()
    imgs = [np.array(to_pil_image(img)) for img in img_list]

    # Specify the directory where you want to save the images
    output_directory = 'outputs/'

    # Iterate through the list and save each image
    for i, image in enumerate(imgs):
        # Construct the file name for the image, e.g., image_0.png, image_1.png, ...
        file_name = f'image_{i}.png'
        
        # Construct the full path to save the image
        save_path = output_directory + "images/" + file_name
        
        # Save the image using cv2.imwrite()
        cv2.imwrite(save_path, image)

    # Optional: Display a message to confirm that the images were saved
    print(f'{len(imgs)} images saved in {output_directory}')

    # Save generator and discriminator
    torch.save(netG.state_dict(), './outputs/generator.pth')
    torch.save(netD.state_dict(), './outputs/discriminator.pth')

    with open(output_directory + "losses_generator.txt", "w") as file:
        for loss in G_losses:
            file.write(f"{loss}\n")

    with open(output_directory + "losses_discriminator.txt", "w") as file:
        for loss in D_losses:
            file.write(f"{loss}\n")


    # Plot results
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

    return

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    print("Random Seed: ", manualSeed)
    main()