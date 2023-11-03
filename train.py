from utils import *
from config import *
from model import *

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import argparse
import json

available_models = {'model_1': create_cnn_model_v1, 'model_2': create_cnn_model_v1}

if __name__ == '__main__':
    
    # Parse model parameters

    parser = argparse.ArgumentParser(description='Train model. Use a comma-separated list for multiple values.')
    parser.add_argument('--model', type=str, required=True, help='Name of the model (available: model_1).')
    parser.add_argument('--batch_size', type=str, required=True, help='Batch size to train model.')
    parser.add_argument('--duration', type=str, required=True, help='Duration of audio files (files are 4 minutes max).')
    parser.add_argument('--number_of_epochs', type=str, required=True, help='Number of epochs.')

    args = parser.parse_args()

    number_of_epochs = [int(epoch) for epoch in args.number_of_epochs.split(',')]
    batch_sizes = [int(batch_size) for batch_size in args.batch_size.split(',')]
    durations = [float(duration) for duration in args.duration.split(',')]

    model_names = [model for model in args.model.split(',')]
    models = {model_name: available_models[model_name] for model_name in model_names}

    # Load dataset

    dataset_df = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
    filepaths = []
    for i, row in dataset_df.iterrows():
        filepaths.append(os.path.join('UrbanSound8K/audio', 'fold'+str(row['fold']), row['slice_file_name']))
    dataset_df['filepath'] = filepaths
    filepaths = filepaths
    filepaths = np.array(filepaths)
    audio_class = dataset_df['classID'].to_numpy()
    audio_class = keras.utils.to_categorical(audio_class, 10)


    # simple train test split
    train_data, test_data, train_labels, test_labels = train_test_split(filepaths, audio_class, test_size=0.2)
    train_idx = np.arange(0, len(train_data))
    val_idx = np.arange(0, len(test_data))

    results_json = {"experiment": []}

    for name, model_function in models.items():
        print(f"Trainig {name}")

        for epoch in number_of_epochs:
            print(f"Number of epochs: {epoch}")

            for batch_size in batch_sizes:
                print(f"Batch size: {batch_size}")

                for duration in tqdm(durations):
                    print(f"File duration: {duration}")

                    train_gen = AudioDataTrainGenerator(train_data, train_labels, train_idx, batch_size=batch_size, duration=duration)
                    val_gen = AudioDataTestGenerator(test_data, test_labels, val_idx, batch_size=batch_size, duration=duration)
                    model = model_function(duration=duration)
                    history = model.fit(train_gen, validation_data=val_gen, epochs=epoch, verbose=1, shuffle=False)

                    experiment = {
                        "model_name": name,
                        "number_of_epochs": epoch,
                        "batch_size": batch_size,
                        "duration": duration,
                        "train_accuracy": history.history['accuracy'],
                        "test_accuracy": history.history['val_accuracy'],
                        "train_loss": history.history['loss'],
                        "test_loss": history.history['val_loss']
                    }

                    results_json['experiment'].append(experiment)

                    model.save(f"{name}_epoch_{epoch}_batch_size_{batch_size}_duration_{duration}.h5")

    json_object = json.dumps(results_json, indent=4)

    with open("results.json", "w") as res:
        res.write(json_object)
        