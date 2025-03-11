# Sound-genre-identification

Input: Any music

Output: Genre of the music

# Music Genre Classification Using Deep Learning

## Overview
This project uses deep learning to classify music genres based on audio features extracted from songs. It leverages the **GTZAN Dataset** and the **VGGish model** from TensorFlow Hub to extract audio embeddings, followed by a Convolutional Neural Network (CNN) for classification.

## Features
- Downloads the **GTZAN music genre dataset** from Kaggle.
- Extracts audio embeddings using **Google's VGGish model**.
- Uses **Convolutional Neural Networks (CNNs)** for classification.
- Evaluates model performance using **accuracy and confusion matrix**.

## Dataset
- **GTZAN Dataset**: A widely used dataset for music genre classification.
- **Genres**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock.

## Prerequisites
Ensure you have the following dependencies installed before running the script:

```sh
pip install kaggle tensorflow tensorflow-hub librosa pandas numpy scikit-learn matplotlib seaborn visualkeras
```

## Steps to Run the Project

### 1. Download the GTZAN Dataset
To download datasets from Kaggle, you need to set up authentication:

```sh
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

Download the dataset:
```sh
!kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
!unzip gtzan-dataset-music-genre-classification.zip
```

### 2. Extract Features
- Load an audio file from the dataset.
- Convert it into a spectrogram.
- Extract features using **VGGish**.

### 3. Train the CNN Model
- The dataset is split into training and testing sets.
- A CNN model is designed with **Conv2D, MaxPooling, BatchNormalization, and Dropout layers**.
- The model is trained using categorical cross-entropy loss and Adam optimizer.

### 4. Evaluate the Model
- The trained model is evaluated on the test set.
- A confusion matrix is generated to analyze classification performance.

## Example Output
```
Processing folder blues: 100%|██████████| 100/100 [00:05<00:00, 18.29it/s]
Processing folder jazz: 100%|██████████| 100/100 [00:06<00:00, 15.89it/s]
...
Test Accuracy: 92.3%
```

## Model Architecture
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 43, 128, 64)       640       
max_pooling2d (MaxPooling2D) (None, 21, 64, 64)        0         
batch_normalization (BatchN  (None, 21, 64, 64)        256       
dropout (Dropout)            (None, 21, 64, 64)        0         
...
=================================================================
Total params: 1,234,567
Trainable params: 1,234,567
Non-trainable params: 0
_________________________________________________________________
```

## Visualization
- **Waveform and Spectrogram of a Sample Audio**.
- **Confusion Matrix** for genre classification.

## License
This project is open-source and available for modification and distribution.

