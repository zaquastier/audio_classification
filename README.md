# Audio Classification using CNNs on UrbanSound8K

This project aims to classify urban sounds using various Convolutional Neural Network (CNN) architectures. The dataset used for this project is the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html).

## Objectives

* Understand **audio signal processing for deep learning**, specifically using mel spectrograms.
* Implement and experiment with different **CNN architectures for audio classification**.
* Learn and apply **Git and programming best practices**.
* Investigate the impact of various **hyperparameters on the model's performance**.
* Explore different **audio augmentation techniques** and their effects on model accuracy.

## Dataset
The UrbanSound8K dataset is utilized in this project. It contains 8732 labeled sound excerpts of urban sounds from 10 classes:

* Air Conditioner
* Car Horn
* Children Playing
* Dog bark
* Drilling
* Engine Idling
* Gun Shot
* Jackhammer
* Siren
* Street Music

Each audio file in this dataset is less than 4 seconds.

## Methodology

### Audio Signal Processing

**Mel spectrograms** are employed to convert audio signals into visual representations, which are then fed into the CNN model for training.

### Audio Normalization

To maintain consistency across the dataset and improve the model's training efficiency all audio files are converted to **mono** to ensure a single channel. Each audio clip is treated to have the **same duration**. This ensures that the input to the model is of a consistent shape.

### Model Architecture

Various **Convolutional Neural Network (CNN) architectures** are experimented with for classifying the audio signals. Detailed architecture configurations and layers will be updated as the project progresses.

### Hyperparameter Tuning

The project will delve into the impact of various hyperparameters such as:

* Batch size
* Number of epochs
* Duration of audio clips
* Audio augmentation techniques


## Getting Started

### Installation

Clone the repository:

```
git clone https://github.com/zaquastier/audio_classification.git
```

Navigate to the project directory and install the required packages:

```
cd audio_classification
pip install -r requirements.txt
```

### Download and extract the dataset:

```
python download.py
```

### Run the main script to train the models(this will be updated as the project progresses):

```
python main.py
```

## Results

TODO

