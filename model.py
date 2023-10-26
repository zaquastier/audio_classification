import tensorflow as tf
import keras 

from config import *

SGRAM_SHAPE=[N_MELS, int(DURATION*SR) // HOP_LENGTH + 1]

def create_cnn_model_v0(num_classes=10, duration=DURATION):
    sgram_shape=[N_MELS, int(duration*SR) // HOP_LENGTH + 1]

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(16, (4, 4), input_shape=(sgram_shape[0], sgram_shape[1], 1), activation='ReLU'))
    model.add(keras.layers.Conv2D(16, (4, 4), activation='ReLU'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(16, (4, 4), activation='ReLU'))
    model.add(keras.layers.Conv2D(16, (4, 4), activation='ReLU'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(16, activation='ReLU'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_cnn_model_v1(num_classes=10, duration=DURATION):
    sgram_shape=[N_MELS, int(duration*SR) // HOP_LENGTH + 1]
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (4, 4), input_shape=(sgram_shape[0], sgram_shape[1], 1), activation='ReLU'))
    model.add(keras.layers.Conv2D(32, (4, 4), activation='ReLU'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv2D(32, (4, 4), activation='ReLU'))
    model.add(keras.layers.Conv2D(32, (4, 4), activation='ReLU'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='ReLU'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='ReLU'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model