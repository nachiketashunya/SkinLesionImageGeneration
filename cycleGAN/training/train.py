import wandb
from torchvision.transforms import transforms

import sys 
sys.path.append("cycleGAN")
from data.customdataset import prepdata
from training.trainer import CGANTrainer 

# Training Configuration
config = {
    'batch_size': 16,
    'lr': 0.0002,
    'epochs': 30
}

# Define transformations
transform = {
        'img' : transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.4, 0.4)),
            transforms.RandomAdjustSharpness(2)
        ]), 
        'sketch': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])  
    }

train_dataloader, test_dataloader = prepdata(config, transform=transform)

# 1. Start a W&B Run
# run = wandb.init(
#             project="CycleGAN", 
#             name="Training", 
#             notes="With data augmentation and changes",
#             config=config
#         )

cgantrainer = CGANTrainer()

# Start Training
cgantrainer.train(train_dataloader, config['batch_size'])

wandb.finish()

# 2. Evaluate the Model
run = wandb.init(project="CycleGAN", name="Evaluation", notes="FID and Inception Score", config=config)

cgantrainer = CGANTrainer()
cgantrainer.load_checkpoints("models/epoch_30.pth")

cgantrainer.evaluate(test_dataloader)

wandb.finish()