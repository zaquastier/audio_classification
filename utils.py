from preprocess import *
import numpy as np
import math
import keras

#do test as well without augmentation
class AudioDataTrainGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, filepaths, labels, indices, batch_size=32, n_classes=10, duration=DURATION, shuffle=True):
        self.batch_size = batch_size
        self.labels = np.array([labels[i] for i in indices])
        self.filepaths = np.array([filepaths[i] for i in indices])
        self.num_samples = len(self.filepaths)
        self.indices = np.arange(self.num_samples)
        np.random.shuffle(self.indices)
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.duration = duration


    def __len__(self):
        return math.ceil(len(self.filepaths) / self.batch_size)

    def __getitem__(self, index):
        batch_indexes = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        filepaths_batch = self.filepaths[batch_indexes]
        
        X_batch = []
        y_batch = self.labels[batch_indexes]

        for i, filepath in enumerate(filepaths_batch):
            audio_data = open_file(filepath, duration=self.duration)
            audio_data = time_augmentation(audio_data, duration=self.duration)
            sgram = mel_spectrogram(audio_data)
            sgram = frequency_augmentation(sgram)

            X_batch.append(sgram)
        X_batch = np.array(X_batch)

        return X_batch, y_batch
    
class AudioDataTestGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, filepaths, labels, indices, batch_size=32, n_classes=10, duration=DURATION, shuffle=False):
        self.batch_size = batch_size
        self.labels = np.array([labels[i] for i in indices])
        self.filepaths = np.array([filepaths[i] for i in indices])
        self.num_samples = len(self.filepaths)
        self.indices = np.arange(self.num_samples)
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.duration = duration


    def __len__(self):
        return math.ceil(len(self.filepaths) / self.batch_size)

    def __getitem__(self, index):
        batch_indexes = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        filepaths_batch = self.filepaths[batch_indexes]
        
        X_batch = []
        y_batch = self.labels[batch_indexes]

        for filepath in filepaths_batch:
            audio_data = open_file(filepath, duration=self.duration)
            sgram = mel_spectrogram(audio_data)

            X_batch.append(sgram)
        X_batch = np.array(X_batch)

        return X_batch, y_batch
    
    