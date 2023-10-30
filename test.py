import keras
import os
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from preprocess import *
import tensorflow as tf

import matplotlib.pyplot as plt

def logits_to_categorical(predictions):
    res = []
    for logits in predictions:
        idx = np.argmax(logits)
        predicted_classes = np.zeros(len(logits))
        predicted_classes[idx] = 1
        res.append(predicted_classes)
    return np.array(res)

def confusion_matrix(y_true, y_pred, labels):
    conf_matrix = [np.zeros(len(labels)) for _ in range(len(labels))]
    for i in range(len(y_true)):
        true_idx = np.argmax(y_true[i])
        pred_idx = np.argmax(y_pred[i])
        conf_matrix[pred_idx][true_idx] +=1
    
    return np.array(conf_matrix)

if __name__ == '__main__':
    model = keras.models.load_model('model_1_epoch_15_batch_size_128_duration_3.0.h5')

    classes = np.array([
        "air_conditioner", 
        "car_horn", 
        "children_playing", 
        "dog_bark", 
        "drilling", 
        "engine_idling", 
        "gun_shot", 
        "jackhammer", 
        "siren", 
        "street_music"
    ])

    dataset_df = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
    filepaths = []
    results = []
    for i, row in dataset_df.iterrows():
        filepath = os.path.join('UrbanSound8K/audio', 'fold'+str(row['fold']), row['slice_file_name'])
        file = open_file(filepath, duration=3.0)
        sgram = mel_spectrogram(file)
        sgram = tf.convert_to_tensor([sgram])
        sgram = tf.expand_dims(sgram, axis=-1)
        res = model(sgram)
        results.append(res[0])


    results = np.array(results)
    results = logits_to_categorical(results)
    audio_class = dataset_df['classID'].to_numpy()
    audio_class = keras.utils.to_categorical(audio_class)

    print(audio_class.shape)
    print(audio_class[0])
    print(results.shape)
    print(results[0])

    conf_matrix = confusion_matrix(y_true=audio_class, y_pred=results, labels=classes)
    print(conf_matrix)
    print(conf_matrix.shape)
    plt.matshow(conf_matrix)
    plt.colorbar()
    plt.show()
