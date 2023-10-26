from utils import *
from config import *
from model import *

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm 

if __name__ == '__main__':
    dataset_df = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
    filepaths = []
    for i, row in dataset_df.iterrows():
        filepaths.append(os.path.join('UrbanSound8K/audio', 'fold'+str(row['fold']), row['slice_file_name']))
    dataset_df['filepath'] = filepaths
    filepaths = filepaths
    filepaths = np.array(filepaths)
    audio_class = dataset_df['classID'].to_numpy()
    audio_class = keras.utils.to_categorical(audio_class, 10)

    models = {'model_1': create_cnn_model_v1}

    epochs = [5, 10, 15, 20]
    batch_sizes = [8, 16, 32, 64, 128]
    durations = [1.0, 1.5, 2.0, 3.0, 4.0]

    # epochs = [5]
    # batch_sizes = [8, 16, 32, 64, 128]
    # durations = [1.5]

    # simple train test split
    train_data, test_data, train_labels, test_labels = train_test_split(filepaths, audio_class, test_size=0.2)
    train_idx = np.arange(0, len(train_data))
    val_idx = np.arange(0, len(test_data))

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    

    for name, model_function in models.items():
        print(f"Trainig {name}")
        current_model_train_accuracies = []
        current_model_val_accuracies = []
        current_model_train_losses = []
        current_model_val_losses = []

        for epoch in epochs:
            print(f"Number of epochs: {epoch}")
            current_epoch_train_accuracies = []
            current_epoch_val_accuracies = []
            current_epoch_train_losses = []
            current_epoch_val_losses = []

            for batch_size in batch_sizes:
                print(f"Batch size: {batch_size}")

                current_batch_size_train_accuracies = []
                current_batch_size_val_accuracies = []
                current_batch_size_train_losses = []
                current_batch_size_val_losses = []


                for duration in tqdm(durations):

                    print(f"File duration: {duration}")

                    train_gen = AudioDataTrainGenerator(train_data, train_labels, train_idx, batch_size=batch_size, duration=duration)
                    val_gen = AudioDataTestGenerator(test_data, test_labels, val_idx, batch_size=batch_size, duration=duration)
                    model = model_function(duration=duration)
                    history = model.fit(train_gen, validation_data=val_gen, epochs=epoch, verbose=1, shuffle=False)
                    # Append metrics to the lists
                    current_batch_size_train_accuracies.append(history.history['accuracy'])
                    current_batch_size_val_accuracies.append(history.history['val_accuracy'])
                    current_batch_size_train_losses.append(history.history['loss'])
                    current_batch_size_val_losses.append(history.history['val_loss'])

                    model.save(f"{name}_epoch_{epoch}_batch_size_{batch_size}_duration_{duration}.h5")


                current_epoch_train_accuracies.append(current_batch_size_train_accuracies)
                current_epoch_val_accuracies.append(current_batch_size_val_accuracies)
                current_epoch_train_losses.append(current_batch_size_train_losses)
                current_epoch_val_losses.append(current_batch_size_val_losses)

            current_model_train_accuracies.append(current_epoch_train_accuracies)
            current_model_val_accuracies.append(current_epoch_val_accuracies)
            current_model_train_losses.append(current_epoch_train_losses)
            current_model_val_losses.append(current_epoch_val_losses)

        train_accuracies.append(current_model_train_accuracies)
        val_accuracies.append(current_model_val_accuracies)
        train_losses.append(current_model_train_losses)
        val_losses.append(current_model_val_losses)

    ## 10-fold cross validation

    # for train_idx, val_idx in kf.split(filepaths):
    #     print(f"Training on fold {fold_num}/{num_folds}")
    #     # Create data generators for training and validation
    #     train_gen = AudioDataTrainGenerator(filepaths, audio_class, train_idx)
    #     val_gen = AudioDataTrainGenerator(filepaths, audio_class, val_idx)

    #     # Define and compile your model (this ensures a fresh model for each fold)
    #     model = create_cnn_model(num_class)  # Define your model architecture

    #     # Train the model
    #     for epoch in tqdm(range(epochs), desc=f"Epochs (Fold {fold_num}/{num_folds})"):
    #         history = model.fit(train_gen, validation_data=val_gen, epochs=1, verbose=0, shuffle=False)
    #         # Append metrics to the lists
    #         train_accuracies.append(history.history['accuracy'])
    #         val_accuracies.append(history.history['val_accuracy'])
    #         train_losses.append(history.history['loss'])
    #         val_losses.append(history.history['val_loss'])

    #     fold_num += 1


    for i, (name, model_function) in enumerate(models.items()):
        for j in range(len(epochs)):
            for k in range(len(batch_sizes)):
                for l in range(len(durations)):
                    print(f"Model {name}, epochs: {epochs[j]}, batch size: {batch_sizes[k]}, duration: {durations[l]}")
                    print(train_accuracies[i][j][k][l], val_accuracies[i][j][k][l], train_losses[i][j][k][l], val_losses[i][j][k][l])
                    
    train_accuracies, val_accuracies, train_losses, val_losses

    # run different models, with different complexity, durations, different epochs, different batch sizes, save all that, and create a paper discussing accuracy of different models, also do cross validation during the night optimize for gpu before github, change learning rate?