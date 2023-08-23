#!/usr/bin/env conda_pytorch_p310
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
import math
import time
from datetime import datetime
import shutil
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import confusion_matrix
import itertools
import argparse


class ZooplanktonDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_names = [name for name in os.listdir(image_folder) if name != ".ipynb_checkpoints"]
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.image_folder, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except (UnidentifiedImageError, OSError):
            # If the image is corrupted or cannot be opened, skip it and move to the next one
            print(f"Corrupted image: {img_path}. Skipping.")
            # Optionally, you can delete the corrupted image from the folder
            os.remove(img_path)
            return self.__getitem__(index + 1)

        if self.transform:
            image = self.transform(image)
        
        return image, img_name

class Model_v2(nn.Module):
    def __init__(self, dropout):
        super(Model_v2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Adjusting the fully connected layers to fit the spatial dimensions
        # of the input image (256x256)
        self.fc1 = nn.Linear(64 * 30 * 30, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 1)  # Output layer for binary classification (1 neuron for probability)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for probability output
        
    def forward(self, x):
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(F.relu(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(-1, 64 * 30 * 30)  # Adjusted view shape
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))  # Use sigmoid activation for probability output
        return x

class Trainer:
    def __init__(self, model, test_data_loader, save_dir, with_cuda,tensorboard_dir, print_freq=1):
        self.device = torch.device("cuda:0" if with_cuda else "cpu")
        self.model = model
        self.test_data_loader = test_data_loader
        self.save_dir = save_dir
        self.print_freq = print_freq
        self.step = 0
        self.accuracy = 0
        self.tensorboard_dir = tensorboard_dir
        self.start_time = datetime.now().strftime('%b-%d_%H-%M')
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

    def predict(self, threshold=0.5):
        try:
            checkpoint = torch.load(args.chkpoint)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
        total_batches = len(self.test_data_loader)
        print(f"Total batches to process: {total_batches}")
        images = []
        predicted_labels = []
        names = []
        index = 1
        for batch_index, (batch_images, image_names) in enumerate(self.test_data_loader, 1):
            print(f"Processing Batch: {batch_index}/{total_batches}")
            index += 1
            with torch.no_grad():
                if self.device:
                    batch_images = batch_images.to(self.device)

                # Forward pass to obtain predicted probabilities for each item in the batch
                batch_outputs = self.model(batch_images)
                batch_predicted_labels = (batch_outputs >= threshold).int()  # Apply threshold for binary classification

            # Iterate over each item in the batch
            for i in range(batch_images.size(0)):
                image = vutils.make_grid(batch_images[i], nrow=1, padding=10, normalize=True)
                predicted_label = batch_predicted_labels[i].item()  # Convert to scalar value
                name = image_names[i]

                images.append(image)
                predicted_labels.append(predicted_label)
                names.append(name)
        
        print("Saving Predictions in desired folders")
        zooplankton_dir = os.path.join(self.save_dir, 'Detected')
        marine_snow_dir = os.path.join(self.save_dir, 'Not-Detected')

        # Delete directories if they exist
        if os.path.exists(zooplankton_dir):
            shutil.rmtree(zooplankton_dir)
        if os.path.exists(marine_snow_dir):
            shutil.rmtree(marine_snow_dir)

        # Create directories
        os.makedirs(zooplankton_dir)
        os.makedirs(marine_snow_dir)

        # Save the classified images into respective directories based on their predicted labels
        for i in range(len(images)):
            image = images[i]
            predicted_label = predicted_labels[i]
            image_name = names[i]
            # Save the image to the appropriate directory based on the predicted label
            if predicted_label == 1:
                save_path = os.path.join(zooplankton_dir, image_name)
            else:
                save_path = os.path.join(marine_snow_dir, image_name)

            vutils.save_image(image, save_path, normalize=True)
        print("Predictions saved successfully")
        # Log the images on TensorBoard
        for i, image in enumerate(images):
            image_name = names[i]
            label = predicted_labels[i]
            self.writer.add_image(f"Image-{i}-Predicted-{label}", image.cpu(), global_step=i)
        print("Successfully logged predictions on Tenserboard")
        # Close the SummaryWriter object
        self.writer.close()

def main(args):
    torch.backends.cuda.max_split_size_mb = 0
    batch_size = 16
    chkpoint = args.chkpoint
    dropout = 0.2
    image_folder = args.image_folder
    tensorboard_dir = args.tensorboard_dir
    save_dir = args.save_dir

    # Data transformation and DataLoader setup
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ZooplanktonDataset(image_folder=image_folder, transform=data_transform)
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model_v2(dropout)
    model.to(device)

    # Initialize Trainer and run prediction
    trainer = Trainer(
        model,
        test_data_loader=test_loader,
        save_dir=save_dir,
        with_cuda=True,
        tensorboard_dir=tensorboard_dir
    )
    print("Data loaded successfully generating predictions")
    trainer.predict(threshold=0.1)
    print("Script ran successfully")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prediction script")
    parser.add_argument('--chkpoint', type=str, help='Path to model saved checkpoint', required=True)
    parser.add_argument('--image_folder', type=str, help='Path to input image folder', required=True)
    parser.add_argument('--save_dir', type=str, help='Path to output Image folders', required=True)
    parser.add_argument('--tensorboard_dir', type=str, help='Path to save tensorboard dir', required=True)
    args = parser.parse_args()
    main(args)
