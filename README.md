# Music Genre Classification using Convolutional Neural Networks

## Introduction
This project aims to classify music genres using audio files. We have utilized Convolutional Neural Networks (CNNs) to train a model on the GTZAN Genre Classification dataset, which consists of 10 different genres. The model achieves high accuracy in predicting the genre of a given audio file.

## Dataset
The dataset used for this project is the GTZAN Genre Classification dataset, which includes 1,000 audio tracks, each 30 seconds long. The dataset is divided into 10 genres, each containing 100 tracks:
You can download the data set through this link
[https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification]

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
The model achieved an accuracy of 72.93% on the test set. The accuracy could potentially be increased by further tuning the hyperparameters and increasing the number of epochs.

## Results
The model achieved a test accuracy of 72.93%. Detailed results and confusion matrix can be found in the results directory.

## Future Work
- Further tuning of hyperparameters.
- Exploring different model architectures.
- Adding more genres and expanding the dataset.
- Implementing real-time genre classification.
