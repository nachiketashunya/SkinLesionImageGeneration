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
from tqdm.notebook import tqdm


# Define Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Create Dataset and Dataloader

"""

TRAIN_LABELS = "data/Assignment_4/Train/Train_labels.csv"
TEST_LABELS = "data/Assignment_4/Test/Test_Labels.csv"

TRAIN_DATA_DIR = "data/Assignment_4/Train/Train_data"
TEST_DATA_DIR = "data/Assignment_4/Test/Test"

TRAIN_SKETCH_DIR = "data/Assignment_4/Train/Contours"
TEST_SKETCH_DIR = "data/Assignment_4/Test/Test_contours"

# Create Dataset
class ISICDataset(Dataset):
    def __init__(self, datadir, csvpath, sketchdir, transform=None):
        self.datadir = datadir
        self.csv = pd.read_csv(csvpath)
        self.sketchdir = sketchdir
        self.transform = transform

    def __len__(self):
        return len(self.csv.index)

    def __getitem__(self, index):
        img_path = os.path.join(self.datadir, self.csv.iloc[index, 0] + ".jpg")
        image = Image.open(img_path)

        labels = self.csv.iloc[index, 1:].values
        label = np.argmax(labels, axis=0)

        label = torch.tensor(label)

        sketch_name = random.choice(os.listdir(self.sketchdir))
        sketch_path = os.path.join(self.sketchdir, sketch_name)
        fs, ext = os.path.splitext(sketch_path)

        while ext not in ['.jpg', '.jpeg']:
          sketch_name = random.choice(os.listdir(self.sketchdir))
          sketch_path = os.path.join(self.sketchdir, sketch_name)
          fs, ext = os.path.splitext(sketch_path)

        sketch = Image.open(sketch_path)

        if self.transform:
            image = self.transform(image)
            sketch = self.transform(sketch)

        return label, image, sketch


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Train Dataset and Dataloader
train_dataset = ISICDataset(TRAIN_DATA_DIR, TRAIN_LABELS, TRAIN_SKETCH_DIR, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

# Train Dataset and Dataloader
test_dataset = ISICDataset(TEST_DATA_DIR, TEST_LABELS, TEST_SKETCH_DIR, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=2)

"""
END
"""

for batch_labels, batch_images, batch_sketches in train_dataloader:
    print("Batch Image Shape:", batch_images.shape) # (8,3,64,64)
    print("Batch Label Shape:", batch_labels.shape) # (8,7)
    print("Batch Sketch Shape:", batch_sketches.shape) # (8,1,64,64)
    break  # Only print the shape of the first batch


# Generator Class
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_embed = nn.Sequential(
            nn.Embedding(7, 100),
            nn.Linear(100, 32*32)
        )

        self.sketch = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.MaxPool2d(2), # 4*32*32
            nn.ReLU(),

            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU() # 8*32*32
        )

        self.model = nn.Sequential(
            # 9 * 32 * 32
            nn.Conv2d(9, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 8 * 16 * 16
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(4),
            nn.ReLU(),
            # 16 * 4 * 4
            nn.ConvTranspose2d(16, 64*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(),
            # 64*8 * 8 * 8
            nn.ConvTranspose2d(64*8, 64*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(),
            # 64*4 * 16 * 16
            nn.ConvTranspose2d(64*4, 64*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(),
            # 64*2 * 32 * 32
            nn.ConvTranspose2d(64*2, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # 3 * 32 * 32
        )

    def forward(self, input):
        sketch, label = input # label: (8,7)

        label_output = self.label_embed(label) # (32*32)
        label_output = label_output.view(-1, 1, 32, 32)

        sketch_output = self.sketch(sketch)
        sketch_output = sketch_output.view(-1, 8, 32, 32)

        concat = torch.cat((label_output, sketch_output), 1)

        return self.model(concat)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embed = nn.Sequential(
            nn.Embedding(7, 100),
            nn.Linear(100, 64*64)
        )

        self.model = nn.Sequential(
            # 4*64*64
            nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32*64*64
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # 64*32*32
            nn.Conv2d(64, 64*2, kernel_size=5, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(64*2, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # 64*2 * 16 * 16
            nn.Conv2d(64*2, 64*4, kernel_size=5, padding=2, stride=2, bias=False),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(64*4, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # 64*4 * 4 * 4
            nn.Conv2d(64*4, 64*8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64*8, momentum=0.1, eps=0.8),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),

            # 64*8 * 2 * 2
            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(2048, 512),
            nn.Tanh(),

            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        img, label = input

        label_output = self.label_embed(label)
        label_output = label_output.view(-1, 1, 64, 64)

        concat = torch.cat((img, label_output), dim=1)

        return self.model(concat)


gen = Generator().to(device)

disc = Discriminator()
disc = disc.to(device)

loss_fn = nn.BCELoss()
learning_rate = 0.0002

gen_opt = torch.optim.Adam(gen.parameters(), lr = learning_rate, betas=(0.5, 0.999))
disc_opt = torch.optim.Adam(disc.parameters(), lr = learning_rate, betas=(0.5, 0.999))

generator_losses = []
discriminator_losses = []
epochs = 5

# writer_real = SummaryWriter(f"tboard/real")
# writer_fake = SummaryWriter(f"tboard/fake")

step = 0

print("Training Started")
for epoch in range(1, epochs+1):
    total_dloss = 0.0
    total_gloss = 0.0

    b_dloss = 0.0
    b_gloss = 0.0

    for index, (labels, image, sketch) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        # real image shape: [64 x 3 x 64 x 64]
        image = image.to(device)
        labels = labels.to(device)
        sketch = sketch.to(device)
        # labels = labels.unsqueeze(1).long()
        # labels shape: [64 x 1]

        disc_opt.zero_grad()

        fake = gen((sketch, labels))
        # fake image output shape: [8 x 1 x 32 x 32]

        fake = fake.detach()
        # fake = torch.repeat_interleave(fake, 3, dim=1)

        fake_image_pred = disc((fake, labels))
        # fake prediction shape [8 x 1]

        real_image_pred = disc((image, labels))
        # real prediction shape [8 x 1]


        real_target = torch.ones(image.size(0), 1).to(device)
        fake_target = torch.zeros(image.size(0), 1).to(device)

        disc_real_loss = loss_fn(real_image_pred, real_target)
        disc_fake_loss = loss_fn(fake_image_pred, fake_target)
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        total_dloss += disc_loss.item()
        b_dloss += disc_loss.item()

        discriminator_losses += [disc_loss.item()]

        gen_opt.zero_grad()

        gen_loss = loss_fn(disc((fake, labels)), real_target)
        gen_loss.backward()
        gen_opt.step()

        total_gloss += gen_loss.item()
        b_gloss += gen_loss.item()

        generator_losses += [gen_loss.item()]

        if index % 100 == 0:
            step +=1

            avg_bdloss = b_dloss / 100
            avg_bgloss = b_gloss / 100

            print(f"Average D Loss: {avg_bdloss}, Average G Loss: {avg_bgloss}\n")

            b_dloss, b_gloss = 0.0, 0.0


            # grid_real = torchvision.utils.make_grid(image[:60], nrow=15,  normalize=True)
            # grid_fake = torchvision.utils.make_grid(fake[:60], nrow=15, normalize=True)

    avg_dloss = total_dloss / len(train_dataloader)
    avg_gloss = total_gloss / len(train_dataloader)

    print(f"Average D Loss: {avg_dloss}, Average G Loss: {avg_gloss}\n")