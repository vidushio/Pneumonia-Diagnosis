
# Pneumonia Classification

## Introduction

Pneumonia is a prevalent respiratory infection that affects millions of people globally. Timely and accurate diagnosis is crucial for effective treatment and improved patient outcomes. Chest X-rays are commonly used in the diagnosis of pneumonia, and the advent of deep learning techniques has opened avenues for automating the classification of X-ray images.

This project focuses on the development of a pneumonia detection model using deep learning. Leveraging Convolutional Neural Networks (CNNs), the model is trained to analyze chest X-ray images and classify them as either normal or indicative of pneumonia. The use of transfer learning, specifically a pre-trained model, enhances the model's ability to extract relevant features from the images.

Objectives

Develop a deep learning model for pneumonia detection using chest X-ray images.
Utilize transfer learning to leverage a pre-trained CNN model.
Train the model on a labeled dataset comprising normal and pneumonia X-ray images.
Evaluate the model's performance on separate validation and test sets.
Provide an inference script for making predictions on new X-ray images.

Dataset

The Chest X-ray Images (Pneumonia) dataset from Kaggle serves as the foundation for training and evaluating the model. This dataset includes labeled X-ray images for training, validation, and testing, enabling the model to learn patterns associated with pneumonia.

Dataset Information

Dataset Path: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Training Set Path: chest_xray_data/train

Validation Set Path: chest_xray_data/val

Test Set Path: chest_xray_data/test

Dataset Size

Training Set Size: 5216 images

Validation Set Size: 16 images

Test Set Size: 624 images

The training script employs this dataset to fine-tune the pre-trained model, optimizing it for pneumonia detection.




## Libraries and tools
PyTorch

Import Statement: import torch
Description: PyTorch is an open-source machine learning library known for its dynamic computational graph, facilitating flexible model development and training.

TorchVision

Import Statement: from torchvision import transforms as T, datasets
Description: TorchVision, part of PyTorch, provides datasets, model architectures, and image transformations, simplifying computer vision tasks.

NumPy

Import Statement: import numpy as np
Description: NumPy is a fundamental library for scientific computing, offering support for large arrays and matrices along with mathematical functions.

Matplotlib

Import Statement: import matplotlib.pyplot as plt
Description: Matplotlib is a versatile 2D plotting library in Python, widely used for creating static, animated, and interactive visualizations.

TQDM

Import Statement: from tqdm.notebook import tqdm
Description: TQDM is a fast progress bar library for Python and command-line interfaces, providing a visual indication of task progress.

Timm

Import Statement: import timm
Description: Timm (PyTorch Image Models) offers pre-trained models for image classification and computer vision tasks, streamlining model importation.

TorchSummary

Import Statement: from torchsummary import summary
Description: TorchSummary is a library providing a concise summary of PyTorch model architectures, including layer information and parameter counts.

os

Import Statement: import os
Description: The os module provides a cross-platform way to interact with the operating system, offering functions for file and directory operations, path manipulation, and more.

DataLoader

Import Statement: from torch.utils.data import DataLoader
Description: DataLoader is a PyTorch utility that efficiently loads and iterates over batches of data from a given dataset during model training or evaluation.

make_grid

Import Statement: from torchvision.utils import make_grid
Description: make_grid is a function within the torchvision library that creates a grid of images from a batch, simplifying the visualization of multiple images in a single display.
## Project workflow

### Dataset Loading and Transformation:

Utilized TorchVision library for common image transformations and dataset loading.
Defined custom transformations, including resizing, random rotation, ToTensor, and normalization.
Loaded the dataset using ImageFolder class from TorchVision, organizing images into classes based on subdirectories.

### Data Batching:

Employed DataLoader class from PyTorch for efficient loading and iteration over batches of data during training and evaluation.
Applied shuffling to introduce randomness during training.

### Model Fine-tuning:

Imported a pre-trained model using timm library for transfer learning.
Froze certain layers by setting requires_grad to False, modifying the classifier to adapt it to the target task.

### Training Loop:

Implemented a custom trainer class with methods for training and validation batch loops.
Trained the model for multiple epochs, monitoring and saving the model with the minimum validation loss.

### Evaluation and Prediction:

Loaded the trained model and performed evaluation on the test set.
Implemented a function to visualize model predictions on individual test images.

### Model Interpretability:

Created a function to visualize model predictions and ground truth for better interpretability.

### Tools and Libraries Used:

Utilized PyTorch for neural network modeling and training.

Incorporated TorchVision for image-related tasks.

Employed NumPy for numerical operations.

Used Matplotlib for data visualization.

Leveraged TQDM for progress bars during training.

Integrated Timm for pre-trained models and architectures.

Utilized TorchSummary for model architecture summaries.

Utilized os for interaction with the operating system.

Implemented DataLoader and make_grid for efficient data handling.


## Result
The trained model was evaluated on the test set to assess its performance on unseen data. The results on the test set are as follows:

Test Accuracy: 87.98%

Test Loss: 0.2959

These metrics indicate that the model achieved a high accuracy on the test set, demonstrating its effectiveness in classifying pneumonia from chest X-ray images.

Visualizing Model Predictions:
An example from the test set (image at index 423) was selected to visualize the model's predictions. The model correctly predicted the class as "PNEUMONIA," and the visualization includes the ground truth label, predicted class probabilities, and the corresponding image.

<img width="587" alt="image" src="https://github.com/vidushio/Diabetes-Prediction/assets/140071981/a4999793-c606-4f00-b364-78755a9f0bab">



These results showcase the model's ability to make accurate predictions on individual images, contributing to its overall high accuracy on the test set.
