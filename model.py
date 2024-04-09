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

labels = pd.read_csv(TRAIN_LABELS)

# Create Dataset
class ISICDataset(Dataset):
    def __init__(self, datadir, csvpath, sketchdir, transform=None):
        self.datadir = datadir 
        self.csv = pd.read_csv(csvpath)
        self.sketchdir = sketchdir
        self.transform = transform 
    
    def __len__(self):
        return len(self.datadir)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.datadir, self.csv.loc[index, 'image'] + ".jpg")
        image = Image.open(img_path)

        labels = self.csv.iloc[index, 1:].values
        labels = labels.astype(np.float64)
        label = np.argmax(labels, axis=0)

        sketch_name = random.choice(os.listdir(self.sketchdir))
        sketch_path = os.path.join(self.sketchdir, sketch_name)
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
            nn.Embedding(1, 100),
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
            nn.ConvTranspose2d(9, 64*8,kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(),
            nn.ConvTranspose2d(64*8, 64*4,kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(),
            nn.ConvTranspose2d(64*4, 64*2,kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(),
            nn.ConvTranspose2d(64*2, 64*1,kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*1),
            nn.ReLU(),
            nn.ConvTranspose2d(64*1, 1,kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        
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
            nn.Embedding(1, 100),
            nn.Linear(100, 64*64)
        )
             
        self.model = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 3, 2, bias=False),
            nn.BatchNorm2d(64, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64*2, 4, 3, 2, bias=False),
            nn.BatchNorm2d(64*2, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64*2, 64*4, 4, 3, 2, bias=False),
            nn.BatchNorm2d(64*4, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64*4, 64*8, 4, 3, 2, bias=False),
            nn.BatchNorm2d(64*8, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Flatten(),
            nn.Dropout(0.4),
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
    for index, (labels, image, sketch) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        batch_size = len(image)

        # real image shape: [64 x 3 x 64 x 64]
        image = image.to(device)
        labels = labels.to(device)
        sketch = sketch.to(device)
        # labels = labels.unsqueeze(1).long()
        # labels shape: [64 x 1]

        disc_opt.zero_grad()

        fake = gen((sketch, labels))
        # fake image output shape: [8 x 1 x 32 x 32]
        
        print('Fake Image Generated')

        fake_image_pred = disc((fake.detach(), labels))
        # fake prediction shape [8 x 1]

        real_image_pred = disc((image, labels))
        # real prediction shape [8 x 1]
        
        print('Prediction Calculated')

        real_target = torch.ones(image.size(0), 1).to(device)
        fake_target = torch.zeros(image.size(0), 1).to(device)

        disc_real_loss = loss_fn(real_image_pred, real_target)
        disc_fake_loss = loss_fn(fake_image_pred, fake_target)
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        discriminator_losses += [disc_loss.item()]

        gen_opt.zero_grad()

        gen_loss = loss_fn(disc((fake, labels)), real_target)
        gen_loss.backward()
        gen_opt.step()

        generator_losses += [gen_loss.item()]

        if index %  100 == 0:
            step +=1

            grid_real = torchvision.utils.make_grid(image[:60], nrow=15,  normalize=True)
            grid_fake = torchvision.utils.make_grid(fake[:60], nrow=15, normalize=True)