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
import datetime
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
import random as random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
from IPython.display import clear_output

#Below code is for classification task
#start
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
    def __init__(self, test_data_loader, save_dir, with_cuda, tensorboard_dir, chkpt1, chkpt2, dropout, image_folder, print_freq=1):
        self.device = torch.device("cuda:0" if with_cuda else "cpu")
        self.test_data_loader = test_data_loader
        self.save_dir = save_dir
        self.print_freq = print_freq
        self.step = 0
        self.accuracy = 0
        self.tensorboard_dir = tensorboard_dir
        self.chkpt1 =chkpt1
        self.chkpt2 = chkpt2
        self.dropout = dropout
        self.image_folder =image_folder
        self.start_time = datetime.now().strftime('%b-%d_%H-%M')
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

    def predict(self, threshold=0.5):

        try:
            checkpoint1 = torch.load(self.chkpt1, map_location=self.device)
            checkpoint2 = torch.load(self.chkpt2, map_location=self.device)
            model1 = Model_v2(self.dropout)
            model1.load_state_dict(checkpoint1['state_dict'])
            model1.to(self.device)
            model1.eval()
            model2 = Model_v2(self.dropout)
            model2.load_state_dict(checkpoint2['state_dict'])
            model2.to(self.device)
            model2.eval()
            print("Model/s loaded successfully.")
        except Exception as e:
            print(f"Error loading model/s: {e}")

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

                # Forward pass through both models
                batch_outputs1 = model1(batch_images)
                batch_outputs2 = model2(batch_images)

                # combined_outputs = torch.sqrt(batch_outputs1 * batch_outputs2 * batch_outputs3) # Geometric mean
                combined_outputs = 0.104 * batch_outputs1 + 0.896 * batch_outputs2 # weighted average
                batch_predicted_labels = (combined_outputs >= threshold).int()

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

            image_path = os.path.join(self.image_folder, image_name)  # Path to the original image
            original_image = Image.open(image_path)  # Open the original image

            # Save the image to the appropriate directory based on the predicted label
            if predicted_label == 1:
                save_path = os.path.join(zooplankton_dir, image_name)
            else:
                save_path = os.path.join(marine_snow_dir, image_name)

            # vutils.save_image(image, save_path, normalize=True)
            original_image.save(save_path)
        print("Predictions saved successfully")
        # Log the images on TensorBoard
        for i, image in enumerate(images):
            image_name = names[i]
            label = predicted_labels[i]
            self.writer.add_image(f"Image-{i}-Predicted-{label}", image.cpu(), global_step=i)
        print("Successfully logged predictions on Tenserboard")
        # Close the SummaryWriter object
        self.writer.close()
#end
#######################################################################################################################
#below code is for segmentation task
#start

class CustomDataset(Dataset):
    def __init__(self, image_files, data_dir, transform=None):
        self.image_files = image_files
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        original_image_path = os.path.join(self.data_dir, self.image_files[index])

        if os.path.exists(original_image_path):
            image_path = original_image_path
        else:
            print(original_image_path)
            raise FileNotFoundError(f"Image not found for index {index}.")

        image = Image.open(image_path).convert("RGB")  # Convert to RGB mode

        if self.transform:
            image = self.transform(image)

        return image,original_image_path

import torch
import torch.nn as nn

def init_conv_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class UNet(nn.Module):
    def __init__(self, input_channels=3, start_neurons=16, dropout_rate=0.0):
        super(UNet, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(input_channels, start_neurons, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(start_neurons)
        self.conv1_1 = nn.Conv2d(start_neurons, start_neurons, kernel_size=3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(start_neurons)
        self.pool1 = nn.MaxPool2d(4, 4)  # Updated kernel_size and stride
        self.dropout1 = nn.Dropout2d(p=dropout_rate)  # Dropout layer added

        self.conv2 = nn.Conv2d(start_neurons, start_neurons * 2, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(start_neurons * 2)
        self.conv2_1 = nn.Conv2d(start_neurons * 2, start_neurons * 2, kernel_size=3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(start_neurons * 2)
        self.pool2 = nn.MaxPool2d(4, 4)  # Updated kernel_size and stride
        self.dropout2 = nn.Dropout2d(p=dropout_rate)  # Dropout layer added

        self.conv3 = nn.Conv2d(start_neurons * 2, start_neurons * 4, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(start_neurons * 4)
        self.conv3_1 = nn.Conv2d(start_neurons * 4, start_neurons * 4, kernel_size=3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(start_neurons * 4)
        self.pool3 = nn.MaxPool2d(4, 4)  # Updated kernel_size and stride
        self.dropout3 = nn.Dropout2d(p=dropout_rate)  # Dropout layer added

        self.conv4 = nn.Conv2d(start_neurons * 4, start_neurons * 8, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(start_neurons * 8)
        self.conv4_1 = nn.Conv2d(start_neurons * 8, start_neurons * 8, kernel_size=3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(start_neurons * 8)
        self.pool4 = nn.MaxPool2d(4, 4)  # Updated kernel_size and stride
        self.dropout4 = nn.Dropout2d(p=dropout_rate)  # Dropout layer added

        # Middle
        self.middle_conv1 = nn.Conv2d(start_neurons * 8, start_neurons * 16, kernel_size=3, padding=1)
        self.middle_conv1_bn = nn.BatchNorm2d(start_neurons * 16)
        self.middle_conv1_1 = nn.Conv2d(start_neurons * 16, start_neurons * 16, kernel_size=3, padding=1)
        self.middle_conv1_1_bn = nn.BatchNorm2d(start_neurons * 16)
        self.dropout_middle = nn.Dropout2d(p=dropout_rate)  # Dropout layer added

        # Decoder
        self.deconv4 = nn.ConvTranspose2d(start_neurons * 16, start_neurons * 8, kernel_size=4, stride=4)  # Updated kernel_size and stride
        self.conv_up4_1 = nn.Conv2d(start_neurons * 16, start_neurons * 8, kernel_size=3, padding=1)
        self.conv_up4_1_bn = nn.BatchNorm2d(start_neurons * 8)
        self.conv_up4_2 = nn.Conv2d(start_neurons * 8, start_neurons * 8, kernel_size=3, padding=1)
        self.conv_up4_2_bn = nn.BatchNorm2d(start_neurons * 8)

        self.deconv3 = nn.ConvTranspose2d(start_neurons * 8, start_neurons * 4, kernel_size=4, stride=4)  # Updated kernel_size and stride
        self.conv_up3_1 = nn.Conv2d(start_neurons * 8, start_neurons * 4, kernel_size=3, padding=1)
        self.conv_up3_1_bn = nn.BatchNorm2d(start_neurons * 4)
        self.conv_up3_2 = nn.Conv2d(start_neurons * 4, start_neurons * 4, kernel_size=3, padding=1)
        self.conv_up3_2_bn = nn.BatchNorm2d(start_neurons * 4)

        self.deconv2 = nn.ConvTranspose2d(start_neurons * 4, start_neurons * 2, kernel_size=4, stride=4)  # Updated kernel_size and stride
        self.conv_up2_1 = nn.Conv2d(start_neurons * 4, start_neurons * 2, kernel_size=3, padding=1)
        self.conv_up2_1_bn = nn.BatchNorm2d(start_neurons * 2)
        self.conv_up2_2 = nn.Conv2d(start_neurons * 2, start_neurons * 2, kernel_size=3, padding=1)
        self.conv_up2_2_bn = nn.BatchNorm2d(start_neurons * 2)

        self.deconv1 = nn.ConvTranspose2d(start_neurons * 2, start_neurons, kernel_size=4, stride=4)  # Updated kernel_size and stride
        self.conv_up1_1 = nn.Conv2d(start_neurons * 2, start_neurons, kernel_size=3, padding=1)
        self.conv_up1_1_bn = nn.BatchNorm2d(start_neurons)
        self.conv_up1_2 = nn.Conv2d(start_neurons, start_neurons, kernel_size=3, padding=1)
        self.conv_up1_2_bn = nn.BatchNorm2d(start_neurons)

        # Output layer
        self.output_layer = nn.Conv2d(start_neurons, 1, kernel_size=1)

        # Initialize weights with identity matrices
        self.apply(init_conv_weights)

    def forward(self, x):
        # Encoder
        conv1 = nn.ReLU(inplace=True)(self.conv1_bn(self.conv1(x)))
        conv1 = nn.ReLU(inplace=True)(self.conv1_1_bn(self.conv1_1(conv1)))
        pool1 = self.pool1(conv1)
        pool1 = self.dropout1(pool1)  # Dropout added

        conv2 = nn.ReLU(inplace=True)(self.conv2_bn(self.conv2(pool1)))
        conv2 = nn.ReLU(inplace=True)(self.conv2_1_bn(self.conv2_1(conv2)))
        pool2 = self.pool2(conv2)
        pool2 = self.dropout2(pool2)  # Dropout added

        conv3 = nn.ReLU(inplace=True)(self.conv3_bn(self.conv3(pool2)))
        conv3 = nn.ReLU(inplace=True)(self.conv3_1_bn(self.conv3_1(conv3)))
        pool3 = self.pool3(conv3)
        pool3 = self.dropout3(pool3)  # Dropout added

        conv4 = nn.ReLU(inplace=True)(self.conv4_bn(self.conv4(pool3)))
        conv4 = nn.ReLU(inplace=True)(self.conv4_1_bn(self.conv4_1(conv4)))
        pool4 = self.pool4(conv4)
        pool4 = self.dropout4(pool4)  # Dropout added

        # Middle
        middle = nn.ReLU(inplace=True)(self.middle_conv1_bn(self.middle_conv1(pool4)))
        middle = nn.ReLU(inplace=True)(self.middle_conv1_1_bn(self.middle_conv1_1(middle)))
        middle = self.dropout_middle(middle)  # Dropout added

        # Decoder
        deconv4 = self.deconv4(middle)
        concat4 = torch.cat([deconv4, conv4], dim=1)
        conv_up4 = nn.ReLU(inplace=True)(self.conv_up4_1_bn(self.conv_up4_1(concat4)))
        conv_up4 = nn.ReLU(inplace=True)(self.conv_up4_2_bn(self.conv_up4_2(conv_up4)))

        deconv3 = self.deconv3(conv_up4)
        concat3 = torch.cat([deconv3, conv3], dim=1)
        conv_up3 = nn.ReLU(inplace=True)(self.conv_up3_1_bn(self.conv_up3_1(concat3)))
        conv_up3 = nn.ReLU(inplace=True)(self.conv_up3_2_bn(self.conv_up3_2(conv_up3)))

        deconv2 = self.deconv2(conv_up3)
        concat2 = torch.cat([deconv2, conv2], dim=1)
        conv_up2 = nn.ReLU(inplace=True)(self.conv_up2_1_bn(self.conv_up2_1(concat2)))
        conv_up2 = nn.ReLU(inplace=True)(self.conv_up2_2_bn(self.conv_up2_2(conv_up2)))

        deconv1 = self.deconv1(conv_up2)
        concat1 = torch.cat([deconv1, conv1], dim=1)
        conv_up1 = nn.ReLU(inplace=True)(self.conv_up1_1_bn(self.conv_up1_1(concat1)))
        conv_up1 = nn.ReLU(inplace=True)(self.conv_up1_2_bn(self.conv_up1_2(conv_up1)))

        # Output layer
        output = self.output_layer(conv_up1)

        return output

def load_model(directory, model, device):
    checkpoint = torch.load(directory, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # Ensure model is on the correct device
    return model


def test_model(model, test_loader, save_dir,checkpoint_dir,with_cuda):
    device = torch.device("cuda:0" if with_cuda else "cpu")
    try:
        model = load_model(checkpoint_dir, model,device=device)
        model.eval()
        print("Model/s loaded successfully.")
    except Exception as e:
        print(f"Error loading model/s: {e}")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)  # Delete the existing directory and its contents
    os.makedirs(save_dir)  # Recreate the directory
    total_batches = len(test_loader)
    with torch.no_grad():
        index = 1
        for  batch_index,(images, image_name) in enumerate(test_loader,1):
            print(f"Segmenting Batch: {batch_index}/{total_batches}")
            images = images.to(device)

            outputs = model(images)
            predicted_masks = (outputs > 0.5).float()  # Assuming binary segmentation
            
            for i in range(images.size(0)):
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 3, 1)
                plt.title("Input Image")
                plt.imshow(images[i].cpu().permute(1, 2, 0))
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.title("Predicted Mask")
                plt.imshow(predicted_masks[i].cpu().squeeze(), cmap='gray')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.title("Overlay")
                overlay = np.copy(images[i].cpu().permute(1, 2, 0))
                overlay_mask = predicted_masks[i].cpu().squeeze() == 1
                # Define overlay color (pure red in RGB format)
                overlay_color = np.array([1.0, 0.0, 0.0])  # Red: [R, G, B]
                # Apply overlay blending
                overlay[overlay_mask] = overlay[overlay_mask] * (1 - overlay_color) + overlay_color
                plt.imshow(overlay)
                plt.axis('off')
                
                # Save the overlay image
                image_filename = os.path.basename(image_name[i])
                overlay_save_path = os.path.join(save_dir, f"{image_filename}")
                plt.savefig(overlay_save_path, bbox_inches='tight', pad_inches=0, dpi=300)
                
                plt.show()
                plt.close()

def main(args):
    use_cuda = args.with_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Device is: ",device)
    if args.is_onlysegmentation:
        print("Skipping classifiction")
        print("Grerating segmentations...")
        segmented_save_dir = args.segmented_save_dir
        segmented_data_dir = args.segmented_data_dir
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image_files = sorted([f for f in os.listdir(segmented_data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        test_dataset = CustomDataset(image_files, segmented_data_dir, transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
        model_segment = UNet(input_channels=3, start_neurons=16, dropout_rate=0.1).to(device)
        test_model(model_segment, test_loader, segmented_save_dir, checkpoint_dir=args.chkpoint_segmentation,with_cuda=use_cuda)
        print("Segmentation Done")
        print("Script ran successfully")
    else:
        torch.backends.cuda.max_split_size_mb = 0
        batch_size = 16
        chkpt1 = args.chkpoint1
        chkpt2 = args.chkpoint2
        dropout = 0.2
        image_folder = args.image_folder
        tensorboard_dir = args.tensorboard_dir
        save_dir = args.save_dir
        segmented_save_dir = args.segmented_save_dir
        segmented_data_dir = args.segmented_data_dir
        # Data transformation and DataLoader setup
        from torchvision import transforms
        data_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = ZooplanktonDataset(image_folder=image_folder, transform=data_transform)
        test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        # Model setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # Initialize Trainer and run prediction
        trainer = Trainer(
            test_data_loader=test_loader,
            save_dir=save_dir,
            with_cuda=use_cuda,
            tensorboard_dir=tensorboard_dir,
            chkpt1= chkpt1,
            chkpt2=chkpt2,
            dropout=dropout,
            image_folder=image_folder
        )
        print("Data loaded successfully generating predictions")
        trainer.predict(threshold=0.1)

        print("Classification ran successfully")
        print("Grerating segmentations...")
        image_files = sorted([f for f in os.listdir(segmented_data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        test_dataset = CustomDataset(image_files, segmented_data_dir, transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
        model_segment = UNet(input_channels=3, start_neurons=16, dropout_rate=0.1).to(device)
        test_model(model_segment, test_loader, segmented_save_dir, checkpoint_dir=args.chkpoint_segmentation,with_cuda=True)
        print("Segmentation Done")
        print("Script ran successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prediction script")
    parser.add_argument('--chkpoint1', type=str, help='Path to model saved checkpoint', required=True)
    parser.add_argument('--chkpoint2', type=str, help='Path to model saved checkpoint', required=True)
    parser.add_argument('--chkpoint_segmentation', type=str, help='Path to segmentation checkpoint', required=True)
    parser.add_argument('--image_folder', type=str, help='Path to input image folder', required=True)
    parser.add_argument('--save_dir', type=str, help='Path to output Image folders', required=True)
    parser.add_argument('--segmented_save_dir', type=str, help='Path to saved segmented predictions', required=True)
    parser.add_argument('--segmented_data_dir', type=str, help='Path to saved detected predictions', required=True)
    parser.add_argument('--tensorboard_dir', type=str, help='Path to save tensorboard dir', required=True)
    parser.add_argument('--with_cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--is_onlysegmentation', action='store_true', help='want only segmentation result')
    args = parser.parse_args()
    main(args)
