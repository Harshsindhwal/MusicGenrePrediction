# Music Genre Classification using Convolutional Neural Networks

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
This project aims to classify music genres using audio files. We have utilized Convolutional Neural Networks (CNNs) to train a model on the GTZAN Genre Classification dataset, which consists of 10 different genres. The model achieves high accuracy in predicting the genre of a given audio file.

## Dataset
The dataset used for this project is the GTZAN Genre Classification dataset, which includes 1,000 audio tracks, each 30 seconds long. The dataset is divided into 10 genres, each containing 100 tracks:

- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

The dataset includes:
- `genres_original`: Audio files for each genre.
- `images_original`: Visual representations of the audio files.
- Two CSV files with extracted features (mean and variance over multiple features) for both 30-second and 3-second splits.

## Feature Extraction
We used the `librosa` library to extract the following features from the audio files:
- Mel-Frequency Cepstral Coefficients (MFCCs)
- Chroma Feature
- Spectral Contrast
- Spectral Rolloff
- Zero Crossing Rate

These features are then scaled and encoded for use in our model.

## Model Architecture
We implemented a Convolutional Neural Network (CNN) with the following architecture:
- Input layer
- Convolutional layers with ReLU activation
- Max Pooling layers
- Fully connected (Dense) layers
- Dropout layers to prevent overfitting
- Output layer with Softmax activation

## Training the Model
The model was trained using the Adam optimizer and sparse categorical cross-entropy loss function. The training was performed over 600 epochs with a batch size of 32.

## Evaluation
The model achieved an accuracy of 92.93% on the test set. The accuracy could potentially be increased by further tuning the hyperparameters and increasing the number of epochs.

## Usage
To use this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/music-genre-classification.git
   cd music-genre-classification
