# Speaker Classification CNN (LibriSpeech)

## Overview
This project implements a speaker classification pipeline using the LibriSpeech dataset. The code defines a PyTorch Dataset class for handling the dataset, extracts Mel spectrogram features, and trains a Convolutional Neural Network (CNN) model for speaker identification.

---

## Features
- **Custom Dataset:** Efficiently loads and preprocesses the LibriSpeech dataset, including audio resampling, downsampling, and padding.
- **Mel Spectrograms:** Extracts Mel spectrogram features using the `torchaudio` library.
- **CNN Classifier:** A CNN architecture designed for speaker classification with support for multi-class outputs.
- **Training and Testing:** Splits the dataset into training and testing subsets for model evaluation.

---

## Requirements
- Python 3.8+
- PyTorch 1.12+
- Torchaudio 0.12+
- Numpy
- Matplotlib
- Tqdm
- Google Colab (Optional, for cloud execution)

---

## Dataset
The model uses the **LibriSpeech** dataset, specifically the `train-clean-100` subset.

---

## Code Highlights

### 1. LibriSpeechDataset Class
The `LibriSpeechDataset` class processes the dataset by:
- Extracting audio file paths, transcriptions, and speaker IDs.
- Resampling, downsampling, and clamping audio signals to ensure uniformity.
- Transforming audio signals into Mel spectrograms.

### 2. CNN Classifier
The `ClassifierCNN` is a four-layer CNN architecture:
- Convolutional layers with ReLU activations and max pooling.
- A fully connected layer for classification.
- Softmax activation for multi-class output.

### 3. Training Pipeline
- Uses `Subset` to create training and testing splits.
- Trains the CNN model using Cross-Entropy Loss and Adam optimizer.
- Tracks training loss for each epoch.

### 4. Evaluation
- Computes the model's accuracy on the test set.
- Visualizes the training loss progression.

---

## Acknowledgments

- LibriSpeech Dataset: https://www.openslr.org/12/  
- PyTorch: https://pytorch.org/  
- Torchaudio: https://pytorch.org/audio/  

