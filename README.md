# CycleGAN for Sketch-to-Image Translation

This repository contains the implementation of a CycleGAN model for sketch-to-image translation incorporating label information.

## Description

The repository consists of the following components:

- `discriminator.py`: Defines the discriminator architecture, which is a PatchGAN discriminator.
- `generator.py`: Implements a UNet-based and ResNet generator for the CycleGAN.
- `image_pool.py`: Defines an image pool to store generated images during training.
- `trainer.py`: Contains the training loop for the CycleGAN model.
- `train.py`: Train the model and evaluate performance

## Dataset

The dataset used for training and evaluation is the ISIC (International Skin Imaging Collaboration) dataset. It consists of images and their corresponding labels along with unpaired sketches.

## Usage

1. **Training the CycleGAN**:
   - Run `train.py` to train the CycleGAN model. Adjust the hyperparameters as needed.
   - Specify the paths to the training and testing datasets in the `ISICDataset` class.
   - Adjust the transformations for images and sketches as required.

   
2. **Evaluation**:
   - Evaluation code is implemented along with training part.
   - Specify the paths to the testing dataset in the `ISICDataset` class.
   - Evaluate metrics are used such as FID and Inception Score.

## Results

The results of training and evaluation are logged using `wandb` (Weights & Biases) for easy visualization and tracking of metrics.

## License

This project is licensed under the [MIT License](LICENSE).
