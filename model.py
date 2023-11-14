import tensorflow as tf
import keras 

from config import *

def create_cnn_model_test(num_classes=10, duration=DURATION): # small model to debug train.py
    sgram_shape=[N_MELS, int(duration*SR) // HOP_LENGTH + 1]

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(16, (4, 4), input_shape=(sgram_shape[0], sgram_shape[1], 1), activation='ReLU'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(16, activation='ReLU'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

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

def create_cnn_model_v2(num_classes=10, duration=DURATION):
    sgram_shape=[N_MELS, int(duration*SR) // HOP_LENGTH + 1]
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (4, 4), input_shape=(sgram_shape[0], sgram_shape[1], 1), activation='ReLU'))
    model.add(keras.layers.Conv2D(32, (4, 4), activation='ReLU'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv2D(16, (4, 4), activation='ReLU'))
    model.add(keras.layers.Conv2D(16, (4, 4), activation='ReLU'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='ReLU'))
    model.add(keras.layers.Dense(128, activation='ReLU'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='ReLU'))
    model.add(keras.layers.Dense(128, activation='ReLU'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_cnn_model_v3(num_classes=10, duration=DURATION):
    sgram_shape=[N_MELS, int(duration*SR) // HOP_LENGTH + 1]
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (5, 5), input_shape=(sgram_shape[0], sgram_shape[1], 1), activation='ReLU'))
    model.add(keras.layers.Conv2D(32, (5, 5), activation='ReLU'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(16, (4, 4), activation='ReLU'))
    model.add(keras.layers.Conv2D(16, (4, 4), activation='ReLU'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='ReLU'))
    model.add(keras.layers.Dense(256, activation='ReLU'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='ReLU'))
    model.add(keras.layers.Dense(256, activation='ReLU'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
def create_cnn_model_v4(num_classes=10, duration=DURATION): # model 3 with more regularization
    sgram_shape=[N_MELS, int(duration*SR) // HOP_LENGTH + 1]
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (5, 5), input_shape=(sgram_shape[0], sgram_shape[1], 1), activation='ReLU'))
    model.add(keras.layers.Conv2D(32, (5, 5), activation='ReLU'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv2D(16, (4, 4), activation='ReLU'))
    model.add(keras.layers.Conv2D(16, (4, 4), activation='ReLU'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='ReLU'))
    model.add(keras.layers.Dense(256, activation='ReLU'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='ReLU'))
    model.add(keras.layers.Dense(256, activation='ReLU'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_cnn_model_v5(num_classes=10, duration=DURATION):
    sgram_shape=[N_MELS, int(duration*SR) // HOP_LENGTH + 1]
    model = keras.Sequential()

    # Conv block 1
    model.add(keras.layers.Conv2D(32, (5, 5), input_shape=(sgram_shape[0], sgram_shape[1], 1), activation='ReLU'))
    model.add(keras.layers.Conv2D(32, (5, 5)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

    # Conv block 2
    model.add(keras.layers.Conv2D(16, (5, 5), activation='ReLU'))
    model.add(keras.layers.Conv2D(16, (5, 5)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

    # MLP
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_cnn_model_v6(num_classes=10, duration=DURATION): # model 3 with residual
    sgram_shape=[N_MELS, int(duration*SR) // HOP_LENGTH + 1]
    input = keras.Input(shape=(sgram_shape[0], sgram_shape[1], 1))
    x = keras.layers.Conv2D(32, (5, 5), padding='same', activation='ReLU')(input)

    indent = x
    for i in range (5):
        x = keras.layers.Conv2D(32, (5, 5), padding='same', activation='ReLU')(x)
        x = keras.layers.add([x, indent])
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)

    indent = x
    for i in range (5):
        x = keras.layers.Conv2D(32, (4, 4), padding='same', activation='ReLU')(x)
        x = keras.layers.add([x, indent])
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation='ReLU')(x)
    x = keras.layers.Dense(256, activation='ReLU')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='ReLU')(x)
    x = keras.layers.Dense(256, activation='ReLU')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=input, outputs=x)

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_cnn_model_v7(num_classes=10, duration=DURATION): # model 3 with residual
    sgram_shape=[N_MELS, int(duration*SR) // HOP_LENGTH + 1]
    input = keras.Input(shape=(sgram_shape[0], sgram_shape[1], 1))
    x = keras.layers.Conv2D(32, (5, 5), padding='same', activation='ReLU')(input)

    indent = x
    for i in range (5):
        x = keras.layers.Conv2D(32, (5, 5), padding='same', activation='ReLU')(x)
        x = keras.layers.add([x, indent])
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)

    indent = x
    for i in range (5):
        x = keras.layers.Conv2D(32, (4, 4), padding='same', activation='ReLU')(x)
        x = keras.layers.add([x, indent])
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation='ReLU')(x)
    x = keras.layers.Dense(256, activation='ReLU')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='ReLU')(x)
    x = keras.layers.Dense(256, activation='ReLU')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=input, outputs=x)

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
