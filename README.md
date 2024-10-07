# Semantic Segmentation of Wetland Areas Using Black and White Historical Aerial Images 

This repository contains code for the **CNN Wetland** project, which involves processing satellite or raster images using a U-Net model for semantic segmentation. The project implements a variety of functions for image preprocessing, prediction, and model evaluation, along with the use of distributed training techniques. Below is a detailed breakdown of the key components.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Functions](#functions)
    - [PreProcessing_BW](#preprocessing_bw)
    - [UNET_BW](#unet_bw)
    - [Prediction](#prediction)
    - [plot_predictions](#plot_predictions)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## Project Overview

The **CNN Wetland Project** leverages the U-Net architecture to segment wetland areas from satellite images. The core functionalities include preprocessing raster images, training a model using PyTorch Lightning, and evaluating predictions with visualizations.

## Functions

### PreProcessing_BW
**File:** `PreProcessing_BW.py`

This function preprocesses input raster images and labels by performing the following:
- Resampling images using GDAL and converting them into PyTorch tensors.
- Applying padding and splitting the image into patches for further processing.
- Removing invalid patches based on specific conditions (e.g., empty patches).
- Augmenting data and saving it for training the U-Net model.

#### Parameters:
- `filename`: Path to the input raster image.
- `label`: Path to the label file for segmentation.

### UNET_BW
**File:** `UNET_BW.py`

This class implements the U-Net model using PyTorch Lightning. It consists of an encoder-decoder structure with a pre-trained encoder and trains the model using the binary cross-entropy loss function for binary segmentation tasks.

#### Key Features:
- Supports metric tracking for accuracy, F1 score, and Intersection over Union (IoU).
- Uses data augmentation techniques during training.
- Can split datasets into training, validation, and testing sets.

#### Parameters:
- `num_classes`: Number of output classes for segmentation.
- `learning_rate`: Learning rate for training.
- `dataset`: Dataset to be used for training.
- `batch_size`: Batch size for training.
- `encoder`: Pre-trained encoder for U-Net.

### Prediction
**File:** `Prediction.py`

This function runs predictions using a trained U-Net model. It performs patch-based prediction and aggregates the results by rotating patches for augmentation. The results are saved as a georeferenced TIFF image.

#### Parameters:
- `args`: A tuple containing `Image`, `model_chkpt`, `opdir`, and `device_queue` for parallel device usage.

### plot_predictions
**File:** `plot_predictions.py`

This function visualizes the predictions of the U-Net model. It generates plots with the original image overlaid with both ground truth and predicted labels, providing a clear comparison between the two.

#### Parameters:
- `model`: Trained U-Net model for prediction.
- `dataloader`: Dataloader containing test images and labels.
- `num_images`: Number of images to display in the visualization.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/CNN_Wetland.git
   cd CNN_Wetland
