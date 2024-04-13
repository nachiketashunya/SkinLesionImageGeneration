import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import itertools
from torch import autograd
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb


from imagePool import ImagePool
from resnetGen import ResnetGenerator
from nlayerDis import NLayerDiscriminator


# Define Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Create Dataset and Dataloader

"""
import sys
sys.path.append("/iitjhome/m23csa016/DLASS4")
TRAIN_LABELS = "Assignment_4/Train/Train_labels.csv"
TEST_LABELS = "Assignment_4/Test/Test_Labels.csv"

TRAIN_DATA_DIR = "Assignment_4/Train/Train_data"
TEST_DATA_DIR = "Assignment_4/Test/Test"

TRAIN_SKETCH_DIR = "Assignment_4/Train/Contours"
TEST_SKETCH_DIR = "Assignment_4/Test/Test_contours"

# Create Dataset
class ISICDataset(Dataset):
    def __init__(self, datadir, csvpath, sketchdir, transform=None):
        self.datadir = datadir
        self.csv = pd.read_csv(csvpath)
        self.sketchdir = sketchdir
        self.transform = transform

    def __len__(self):
        return len(self.csv[:300])

    def __getitem__(self, index):
        img_path = os.path.join(self.datadir, self.csv.iloc[index, 0] + ".jpg")
        image = Image.open(img_path)

        labels = self.csv.iloc[index, 1:].values
#         label = np.argmax(labels, axis=0)

        sketch_name = random.choice(os.listdir(self.sketchdir))
        sketch_path = os.path.join(self.sketchdir, sketch_name)
        fs, ext = os.path.splitext(sketch_path)

        while ext not in ['.jpg', '.jpeg', '.png']:
          sketch_name = random.choice(os.listdir(self.sketchdir))
          sketch_path = os.path.join(self.sketchdir, sketch_name)
          fs, ext = os.path.splitext(sketch_path)

        sketch = Image.open(sketch_path)

        if self.transform:
            image = self.transform(image)
            sketch = self.transform(sketch)
            
        x, y = int(image.size(1)), int(image.size(1) / 7)

        labels = np.array(labels, dtype=np.float32)
        labels = np.tile(labels,(x,y))

        label = torch.tensor(labels, dtype=torch.float32)

        return label, image, sketch


transform = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.ToTensor()
])

# Train Dataset and Dataloader
train_dataset = ISICDataset(TRAIN_DATA_DIR, TRAIN_LABELS, TRAIN_SKETCH_DIR, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)

# Train Dataset and Dataloader
test_dataset = ISICDataset(TEST_DATA_DIR, TEST_LABELS, TEST_SKETCH_DIR, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

# Initialize GradScaler from AMP
dscaler = GradScaler()
gscaler = GradScaler()

"""
END
"""

def show(r_img, c_limage, fake_img):
    fig, axes = plt.subplots(1, 3, figsize=(5, 6))
       
    r_img = r_img.squeeze(0)
    c_limage = c_limage.squeeze(0)
    fake_img = fake_img.squeeze(0)
    
    r_img = r_img.detach()
    r_img = F.to_pil_image(r_img)
    axes[0].imshow(r_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot the mask
    c_limage = c_limage.detach()
    c_limage = F.to_pil_image(c_limage)
    axes[1].imshow(c_limage)
    axes[1].set_title('Image & Label')
    axes[1].axis('off')

    # Plot the segmented mask
    fake_img = fake_img.detach()
    fake_img = F.to_pil_image(fake_img)
    axes[2].imshow(fake_img)
    axes[2].set_title('Generated Image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


class CGANTrainer():
    def __init__(self, rank=None):
        super().__init__()
        self.optimizers = []
        self.lamb = 10.0
        
        self.label_embed = nn.Sequential(
            nn.Embedding(7, 100),
            nn.Linear(100, 64*64)
        ).to(device)

        if rank:
            self.genA = ResnetGenerator(input_nc=3, output_nc=3).to(rank)
            self.genA = DDP(self.genA, device_ids=[rank])

            self.genB = ResnetGenerator(input_nc=3, output_nc=3).to(rank)
            self.genB = DDP(self.genB, device_ids=[rank])

            self.disA = NLayerDiscriminator(input_nc=3).to(rank)
            self.disA = DDP(self.disA, device_ids=[rank])

            self.disB = NLayerDiscriminator(input_nc=3).to(rank)
            self.disB = DDP(self.disB, device_ids=[rank])
        else:
            self.genA = ResnetGenerator(input_nc=3, output_nc=3).to(device)
            self.genB = ResnetGenerator(input_nc=3, output_nc=3).to(device)
            self.disA = NLayerDiscriminator(input_nc=3).to(device)
            self.disB = NLayerDiscriminator(input_nc=3).to(device)
            
        self.fakeA_pool = ImagePool(pool_size=50)
        self.fakeB_pool = ImagePool(pool_size=50)

        self.GANloss = nn.BCEWithLogitsLoss()
        self.cycleLoss = nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.genA.parameters(), self.genB.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    # Gradient Penalty for WGAN
    def gradient_penalty(self, dis, real, fake):
        alpha = torch.rand(real.size(0), real.size(1), 1, 1)

        alpha = alpha.expand(real.size())
        alpha = alpha.float().to(device)
        
        xhat = alpha * real + (1-alpha) * fake
        xhat = xhat.float().to(device)

        xhat = autograd.Variable(xhat, requires_grad = True)
        xhat_D = dis(xhat)
        
        grad = autograd.grad(
                    outputs=xhat_D, 
                    inputs=xhat, 
                    grad_outputs=torch.ones(xhat_D.size()).to(device),
                    create_graph=True, retain_graph=True, only_inputs=True
                )[0]
        
        penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * 0.5
        
        return penalty

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        with autocast():
            pred_real = netD(real).mean()
            # self.real_target = self.real_target.expand_as(pred_real)
            # loss_D_real = self.GANloss(pred_real, self.real_target)

            # Fake
            pred_fake = netD(fake.detach()).mean()
            # self.fake_target = self.real_target.expand_as(pred_fake)
            # loss_D_fake = self.GANloss(pred_fake, self.fake_target)
        
        # Combined loss and calculate gradients
        gp = self.gradient_penalty(netD, real, fake.detach())

        loss_D = (pred_fake - pred_real) + gp
        
        dscaler.scale(loss_D).backward()
#         loss_D.backward(retain_graph=True)

        return loss_D

    def backward_disA(self):
        """Calculate GAN loss for discriminator disA"""
        fake_B = self.fakeB_pool.query(self.fake_sketch)
        self.loss_disA = self.backward_D_basic(self.disA, self.concat_ls, fake_B)

    def backward_disB(self):
        """Calculate GAN loss for discriminator disB"""
        fake_A = self.fakeA_pool.query(self.fake_image)
        self.loss_disB = self.backward_D_basic(self.disB, self.concat_li, fake_A)
    
    # Generator Backpropagation Function
    def backward_G(self):
        """Calculate the loss for generators genA and genB"""

        # GAN loss disA(genA(image))
        with autocast():
            fake_prediction_A = self.disA(self.fake_sketch).mean()
            real_prediction_A = self.disA(self.sketch).mean()
            fake_prediction_B = self.disB(self.fake_image).mean()
            real_prediction_B = self.disB(self.image).mean()
        
        gp_A = self.gradient_penalty(self.disA, self.sketch, self.fake_sketch)
        OTdis_A = (fake_prediction_A - real_prediction_A) + gp_A

        gp_B = self.gradient_penalty(self.disB, self.image, self.fake_image)
        OTdis_B = (fake_prediction_B - real_prediction_B) + gp_B

        # Forward cycle loss || genB(genA(image)) - image ||
        self.loss_cycle_A = self.cycleLoss(self.rec_image, self.concat_li) * self.lamb
        # Backward cycle loss || genA(genB(sketch)) - sketch ||
        self.loss_cycle_B = self.cycleLoss(self.rec_sketch, self.concat_ls) * self.lamb

        # combined loss and calculate gradients
        self.loss_G = self.loss_cycle_A + self.loss_cycle_B - (OTdis_A + OTdis_B)
        
        gscaler.scale(self.loss_G).backward()
#         self.loss_G.backward(retain_graph=True)

    def train(self, dataloader, epochs=10):
        for epoch in range(1, epochs+1):
            total_dloss = 0.0
            total_gloss = 0.0
            
            b_dloss, b_gloss = 0.0, 0.0

            for index, input in tqdm(enumerate(dataloader), total=len(dataloader)):
                torch.cuda.empty_cache()
                
                self.label, self.image, self.sketch = input 
                self.sketch = torch.repeat_interleave(self.sketch, 3, dim=1)

                self.label = self.label.to(device)
                self.image = self.image.to(device)
                self.sketch = self.sketch.to(device)

#                 label_output = self.label_embed(self.label) # (32*32)
                self.label = self.label.unsqueeze(1)
    
                self.concat_li = self.image + self.label
                self.concat_ls = self.sketch + self.label
        

                self.real_target = torch.ones(self.image.size(0), 1, 1, 1).to(device)
                self.fake_target = torch.zeros(self.image.size(0), 1, 1, 1).to(device)
                
                with autocast():
                    self.fake_sketch = self.genA(self.concat_li)
                    self.rec_image = self.genB(self.fake_sketch)

                    self.fake_image = self.genB(self.concat_ls)
                    self.rec_sketch = self.genA(self.fake_image)

                # Freeze Discriminator to avoid unnecessary calculations
                self.set_requires_grad([self.disA, self.disB], False)

                # Start training Generator (genA & genB)
                self.optimizer_G.zero_grad()
                self.backward_G()
                
                gscaler.unscale_(self.optimizer_G)
                gscaler.step(self.optimizer_G)
#                 self.optimizer_G.step()

                # Start training Discriminator (disA & disB)
                self.set_requires_grad([self.disA, self.disB], True)
                self.optimizer_D.zero_grad()
                self.backward_disA()
                self.backward_disB()
                
                dscaler.unscale_(self.optimizer_D)
                dscaler.step(self.optimizer_D)
#                 self.optimizer_D.step()

                gscaler.update()
                dscaler.update()

                total_dloss += (self.loss_disA + self.loss_disB) / 2
                total_gloss += self.loss_G
                
                b_dloss += (self.loss_disA + self.loss_disB) / 2
                b_gloss += self.loss_G
                
                # Intermediate logging and visualization
                if index % 100 == 0:
                    print(f"{index}/{len(train_dataloader)} Batch Dis Loss: {b_dloss}, Batch Gen Loss: {b_gloss}\n")

                    b_dloss, b_gloss = 0.0, 0.0
        
            avg_dloss = total_dloss / len(dataloader)
            avg_gloss = total_gloss / len(dataloader)
            
            wandb.log({
                'DLoss': avg_dloss,
                'GLoss': avg_gloss
            })
            print(f"{epoch}/{epochs} Average D Loss: {avg_dloss}, Average G Loss: {avg_gloss}\n")
            show(self.image[0], self.concat_li[0], self.fake_image[0])
