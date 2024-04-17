import random 
import sys
import torch 
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import pandas as pd 
import numpy as np 
import os

# sys.path.append("/iitjhome/m23csa016/DLASS4")
sys.path.append("data")
TRAIN_LABELS = "data/Assignment_4/Train/Train_labels.csv"
TEST_LABELS = "data/Assignment_4/Test/Test_Labels.csv"

TRAIN_DATA_DIR = "data/Assignment_4/Train/Train_data"
TEST_DATA_DIR = "data/Assignment_4/Test/Test"

TRAIN_SKETCH_DIR = "data/Assignment_4/Train/Contours"
TEST_SKETCH_DIR = "data/Assignment_4/Test/Test_contours"

# sys.path.append("/iitjhome/m23csa016/DLASS4")
DATA_DIR = "/scratch/data/m23csa016/"

TRAIN_LABELS = os.path.join(DATA_DIR, "isic/Train/Train_labels.csv")
TEST_LABELS = os.path.join(DATA_DIR, "isic/Test/Test_Labels.csv")

TRAIN_DATA_DIR = os.path.join(DATA_DIR, "isic/Train/Train_data")
TEST_DATA_DIR = os.path.join(DATA_DIR, "isic/Test/Test")

TRAIN_SKETCH_DIR = os.path.join(DATA_DIR, "isic/Train/Contours")
TEST_SKETCH_DIR = os.path.join(DATA_DIR, "isic/Test/Test_contours")

# Create Dataset
class ISICDataset(Dataset):
    def __init__(self, datadir, csvpath, sketchdir, transform=None):
        self.datadir = datadir
        self.csv = pd.read_csv(csvpath)
        self.sketchdir = sketchdir
        self.transform = transform

    def __len__(self):
        return len(self.csv[:4])

    def __getitem__(self, index):
        img_path = os.path.join(self.datadir, self.csv.iloc[index, 0] + ".jpg")
        image = Image.open(img_path)

        # Apply Gaussian blur
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=2))

        # Apply sharpening
        sharpened_image = image.filter(ImageFilter.UnsharpMask)
        
        labels = self.csv.iloc[index, 1:].values

        sketch_name = random.choice(os.listdir(self.sketchdir))
        sketch_path = os.path.join(self.sketchdir, sketch_name)
        fs, ext = os.path.splitext(sketch_path)

        while ext not in ['.jpg', '.jpeg', '.png']:
          sketch_name = random.choice(os.listdir(self.sketchdir))
          sketch_path = os.path.join(self.sketchdir, sketch_name)
          fs, ext = os.path.splitext(sketch_path)

        sketch = Image.open(sketch_path)

        image = self.transform['img'](sharpened_image)
        sketch = self.transform['sketch'](sketch)
        
        true_label = np.argmax(labels)
        # Create a numpy array of zeros with shape (num_classes, 256, 256)
        encoded_label = np.zeros((7, image.size(1), image.size(1)), dtype=np.float32)

        # Set all elements in the channel corresponding to the true label to 1
        encoded_label[true_label, :, :] = 1  # Use broadcasting to set all elements

        label = torch.tensor(encoded_label, dtype=torch.float32)

        return label, image, sketch
    

def prepdata(config, transform=None):
    # Train Dataset and Dataloader
    train_dataset = ISICDataset(TRAIN_DATA_DIR, TRAIN_LABELS, TRAIN_SKETCH_DIR, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    # Train Dataset and Dataloader
    test_dataset = ISICDataset(TEST_DATA_DIR, TEST_LABELS, TEST_SKETCH_DIR, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    return train_dataloader, test_dataloader



