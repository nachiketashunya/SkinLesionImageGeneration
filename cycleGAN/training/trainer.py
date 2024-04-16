import pandas as pd
import torch 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from tqdm import tqdm
import itertools
from torch import autograd
import wandb
import os
from torch.cuda.amp import autocast, GradScaler
import torch.nn.init as init
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

import sys 
sys.path.append("cycleGAN")
from utils.training_utils import ImagePool
from training.generator import ResnetGenerator, UnetGenerator
from training.discriminator import Discriminator
from utils.visualization_utils import show

MODEL_SAVE_DIR = "/scratch/data/m23csa016/models/"

# Define Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gscaler = GradScaler()
dscaler = GradScaler()

class CGANTrainer():
    def __init__(self,lr=0.0002, rank=None):
        super().__init__()
        self.optimizers = []
        self.lamb = 10.0

        self.label_embed = nn.Sequential(
            nn.Embedding(7, 100),
            nn.Linear(100, 64*64)
        ).to(device)

        self.fid = FrechetInceptionDistance(feature=64).to(device)
        self.inception = InceptionScore().to(device)

       
        self.genA = UnetGenerator().to(device)
        self.genB = UnetGenerator().to(device)
        self.disA = Discriminator().to(device)
        self.disB = Discriminator().to(device)

        self.genA.apply(self.weights_init)
        self.genB.apply(self.weights_init)
        self.disA.apply(self.weights_init)
        self.disB.apply(self.weights_init)

        self.fakeA_pool = ImagePool(pool_size=50)
        self.fakeB_pool = ImagePool(pool_size=50)

        self.GANloss = nn.BCEWithLogitsLoss()
        self.cycleLoss = nn.L1Loss()
        self.idLoss = nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.genA.parameters(), self.genB.parameters()), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()), lr=lr, betas=(0.5, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def save_checkpoint(self, epoch):
        # Save generator and discriminator models
        torch.save({
            'epoch': epoch,
            'genA_state_dict': self.genA.state_dict(),
            'genB_state_dict': self.genB.state_dict(),
            'disA_state_dict': self.disA.state_dict(),
            'disB_state_dict': self.disB.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict()
        }, os.path.join(MODEL_SAVE_DIR, f"epoch_{epoch}.pth"))
    
    def load_checkpoints(self, file):
        checkpoint = torch.load(file)
        
        self.genA.load_state_dict(checkpoint['genA_state_dict'])
        self.genB.load_state_dict(checkpoint['genB_state_dict'])
        self.disA.load_state_dict(checkpoint['disA_state_dict'])
        self.disB.load_state_dict(checkpoint['disB_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    def weights_init(self,m):
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

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
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_discr(self, disNet, real, fake):
        # Real
        with autocast():
            pred_real = disNet(real)
            self.real_target = self.real_target.expand_as(pred_real)
            loss_D_real = self.GANloss(pred_real, self.real_target)

            # Fake
            pred_fake = disNet(fake.detach())
            self.fake_target = self.fake_target.expand_as(pred_fake)
            loss_D_fake = self.GANloss(pred_fake, self.fake_target)

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        dscaler.scale(loss_D).backward()

        return loss_D

    def backward_disA(self):
        """Calculate GAN loss for discriminator disA"""
        fake_B = self.fakeB_pool.query(self.fake_sketch)
        self.loss_disA = self.backward_discr(self.disA, self.concat_ls, fake_B)

    def backward_disB(self):
        """Calculate GAN loss for discriminator disB"""
        fake_A = self.fakeA_pool.query(self.fake_image)
        self.loss_disB = self.backward_discr(self.disB, self.concat_li, fake_A)

    # Generator Backpropagation Function
    def backward_gener(self):
        """Calculate the loss for generators genA and genB"""

        # GAN loss disA(genA(image))
        with autocast():
            prediction = self.disA(self.fake_sketch)
            self.real_target = self.real_target.expand_as(prediction)
            self.genA_Loss = self.GANloss(prediction, self.real_target)

            # GAN loss disB(genB(sketch))
            prediction = self.disB(self.fake_image)
            self.real_target = self.real_target.expand_as(prediction)
            self.genB_Loss = self.GANloss(prediction, self.real_target)

            self.loss_cycle_A = self.cycleLoss(self.rec_image, self.concat_li) * self.lamb
            self.loss_cycle_B = self.cycleLoss(self.rec_sketch, self.concat_ls) * self.lamb

            self.idt_A = self.genA(self.concat_ls)
            self.loss_idt_A = self.idLoss(self.idt_A, self.concat_ls) * self.lamb * 0.5

            self.idt_B = self.genB(self.concat_li)
            self.loss_idt_B = self.idLoss(self.idt_B, self.concat_li) * self.lamb * 0.5

        self.loss_G = self.genA_Loss + self.genB_Loss + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        gscaler.scale(self.loss_G).backward()

    def train(self, dataloader, epochs=10):
        save_every = 10
        for epoch in range(1, epochs+1):
            total_dloss = 0.0
            total_gloss = 0.0

            b_dloss, b_gloss = 0.0, 0.0

            for index, input in tqdm(enumerate(dataloader), total=len(dataloader)):
                torch.cuda.empty_cache()

                self.label, self.image, self.sketch_unexp = input

                self.sketch = torch.cat([self.sketch_unexp] * 3, dim=1)

                self.label = self.label.to(device)
                self.image = self.image.to(device)
                self.sketch = self.sketch.to(device)

                self.concat_li = torch.cat((self.image, self.label), dim=1)
                self.concat_ls = torch.cat((self.sketch, self.label), dim=1)

                self.real_target = torch.ones(self.image.size(0), 1, 1, 1).to(device)
                self.fake_target = torch.zeros(self.image.size(0), 1, 1, 1).to(device)

                for _ in range(5):
                    with autocast():
                        self.fake_sketch = self.genA(self.concat_li)
                        self.rec_image = self.genB(self.fake_sketch)

                        self.fake_image = self.genB(self.concat_ls)
                        self.rec_sketch = self.genA(self.fake_image)

                    # Start training Discriminator (disA & disB)
                    self.set_requires_grad([self.disA, self.disB], True)
                    self.optimizer_D.zero_grad()
                    self.backward_disA()
                    self.backward_disB()

                    dscaler.unscale_(self.optimizer_D)
                    dscaler.step(self.optimizer_D)

                    dscaler.update()

                # Freeze Discriminator to avoid unnecessary calculations
                self.set_requires_grad([self.disA, self.disB], False)

                # Start training Generator (genA & genB)
                self.optimizer_G.zero_grad()
                self.backward_gener()

                gscaler.unscale_(self.optimizer_G)
                gscaler.step(self.optimizer_G)

                gscaler.update()

                total_dloss += (self.loss_disA + self.loss_disB) / 2
                total_gloss += self.loss_G

                b_dloss += (self.loss_disA + self.loss_disB) / 2
                b_gloss += self.loss_G

                # Intermediate logging and visualization
                if index % 50 == 0:
                    print(f"{index}/{len(dataloader)} Batch Dis Loss: {b_dloss}, Batch Gen Loss: {b_gloss}\n")

                    b_dloss, b_gloss = 0.0, 0.0

            if epoch % save_every == 0:
                self.save_checkpoint(epoch)

            avg_dloss = total_dloss / len(dataloader)
            avg_gloss = total_gloss / len(dataloader)

            dis_img = F.to_pil_image(self.concat_li[0, :3, :, :])
            dis_sket = F.to_pil_image(self.sketch[0, :3, :, :])
            dis_fsket = F.to_pil_image(self.fake_sketch[0, :3, :, :])
            dis_fimg = F.to_pil_image(self.fake_image[0, :3, :, :])

            wandb.log({
                'dloss': avg_dloss,
                'gloss': avg_gloss,
                'images': [
                    wandb.Image(dis_img, caption="Real Image"),
                    wandb.Image(dis_sket, caption="Sketch"),
                    wandb.Image(dis_fsket, caption="Fake Sketch"),
                    wandb.Image(dis_fimg, caption="Fake Image")
                ]
            })

            show(dis_img, dis_sket, dis_fsket, dis_fimg)

            print(f"{epoch}/{epochs} Average D Loss: {avg_dloss}, Average G Loss: {avg_gloss}\n")

    def evaluate(self, dataloader):
        f_dis = 0.0
        t_incs = 0.0
        for index, input in tqdm(enumerate(dataloader), total=len(dataloader)):
            torch.cuda.empty_cache()

            self.label, self.image, self.sketch_unexp = input

            self.sketch = torch.cat([self.sketch_unexp] * 3, dim=1)

            self.label = self.label.to(device)
            self.image = self.image.to(device)
            self.sketch = self.sketch.to(device)

            self.concat_li = torch.cat((self.image, self.label), dim=1)
            self.concat_ls = torch.cat((self.sketch, self.label), dim=1)

            self.real_target = torch.ones(self.image.size(0), 1, 1, 1).to(device)
            self.fake_target = torch.zeros(self.image.size(0), 1, 1, 1).to(device)

            with autocast():
                self.fake_sketch = self.genA(self.concat_li)
                self.rec_image = self.genB(self.fake_sketch)

                self.fake_image = self.genB(self.concat_ls)
                self.rec_sketch = self.genA(self.fake_image)


            self.inception.update(self.fake_image[:, :3, :, :].to(torch.uint8))
            incs = self.inception.compute()[0]

            t_incs += incs.item()

            self.fid.update(self.image[:, :3, :, :].to(torch.uint8), real=True)
            self.fid.update(self.fake_image[:, :3, :, :].to(torch.uint8), real=False)

            dis = self.fid.compute()
            f_dis += dis.item()

        avg_fid = f_dis / len(dataloader)
        print(f"Avg FID: {avg_fid}")

        avg_incs = t_incs / len(dataloader)
        print(f"Avg IS: {avg_incs}")

        dis_img = F.to_pil_image(self.concat_li[0, :3, :, :])
        dis_sket = F.to_pil_image(self.sketch[0, :3, :, :])
        dis_fsket = F.to_pil_image(self.fake_sketch[0, :3, :, :])
        dis_fimg = F.to_pil_image(self.fake_image[0, :3, :, :])

        wandb.log({
            'FID': avg_fid,
            'IS': avg_incs,
            'images': [
                wandb.Image(dis_img, caption="Real Image"),
                wandb.Image(dis_sket, caption="Sketch"),
                wandb.Image(dis_fsket, caption="Fake Sketch"),
                wandb.Image(dis_fimg, caption="Fake Image")
            ]
        })

        show(dis_img, dis_sket, dis_fsket, dis_fimg)
