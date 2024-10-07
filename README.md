# Semantic Segmentation of Wetland Areas Using Black and White Historical Aerial Images 
## Table of Contents
1. [Project Overview](#project-overview)
2. [Background](#background)
3. [Data Sources](#data-sources)
4. [Preprocessing](#preprocessing)
5. [Training & Refinement](#training--refinement)
6. [Tools Used](#tools-used)
7. [Metrics](#metrics)
8. [Functions](#functions)
    - [PreProcessing_BW](#preprocessing_bw)
    - [UNET_BW](#unet_bw)
    - [Prediction](#prediction)
    - [plot_predictions](#plot_predictions)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Contributing](#contributing)
12. [License](#license)

## Project Overview

This repository contains code for the **CNN Wetland** project, which involves processing satellite or raster images using a U-Net model for semantic segmentation. The project implements a variety of functions for image preprocessing, prediction, and model evaluation, along with the use of distributed training techniques. Below is a detailed breakdown of the key components.

## Background

Urban development and human activities significantly stress various ecosystems, including wetlands. Wetlands are crucial for natural carbon storage but can become carbon emitters if disturbed or drained. Understanding the historical changes in wetlands due to development is essential for effective future monitoring. One of the major challenges in understanding the extent of wetland loss lies in the lack of historical data, making it difficult to establish a baseline for comparison with current conditions. Historical aerial imagery offers a valuable resource for overcoming data gaps, providing high-resolution visuals that can be used to analyze past wetland extents and changes over time. Despite their usefulness, historical aerial images are limited by their spectral information, posing challenges for precise environmental analysis. This study aims to develop a model capable of segmenting wetland areas from historical aerial images, which can be scaled for monitoring human activities and their impact on wetlands.

## Data Sources

### Image Source
- Norwegian mapping authority provides historical orthophotos along different time spans of Norway.

### Labels
- The Natural History Museum at NTNU holds fieldwork maps from various periods across Norwegian catchments. These maps were matched with corresponding aerial images and manually annotated to create labeled datasets.

![Figure 1](/Logos/Training_Data.jpg)

## Preprocessing

The annotated label data and aerial images were preprocessed as follows:

- **Clipping**: Images were divided and folded into 512x512 pixel patches.
- **Filtering**: Patches without foreground labels (wetland areas) or containing no data labels were excluded.
- **Augmentation**: The images, originally at 20cm pixel resolution, were resampled to 0.5m resolution and then rotated by 90, 180, and 270 degrees.
- **Parallel Processing**: Preprocessing was distributed across 64 CPU cores and 12 multi-threads.


## Training & Refinement

- **Model**: Processed patches were trained using the U-Net architecture [3] with EfficientNetB07 as the encoder [4].
- **Training & Metrics**: The model was trained for 200 epochs using the Adam optimizer and BCE Binary Loss function, distributed across 3 NVIDIA A1000 GPUs. The following additional metrics were monitored: Accuracy, F1 Score, and IoU.
- **Prediction**: The trained model was used to predict 4 additional aerial image sets, which were refined using reference fieldwork maps. These refined labels were then added to the initial dataset.

## Tools Used

- **GDAL**: Used for Geoprocessing.
- **Pytorch & Torch Lightning**: Used for CNN training and parallel GPU computing.

![Figure 2](/Logos/Mapping%20Diagram.jpg)

## Metrics

The final training results were:

- **Accuracy**: 91.1%
- **F1 Score**: 87.9%
- **IoU**: 78.6%
- **Training Loss**: 0.2
- **Validation Loss**: 0.184
- **Test Loss**: 0.182

![Figure 3](/Logos/Testing.png)

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
