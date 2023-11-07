from utils import *
from config import *
from model import *

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import argparse
import json
from datetime import datetime


available_models = {
    'model_0': create_cnn_model_v0, 
    'model_1': create_cnn_model_v1, 
    'model_2': create_cnn_model_v2, 
    'model_3': create_cnn_model_v3,
    'model_4': create_cnn_model_v4,
    'model_5': create_cnn_model_v5
}

if __name__ == '__main__':
    
    # Parse model parameters

    parser = argparse.ArgumentParser(description='Train model. Use a comma-separated list for multiple values.')
    parser.add_argument('--model', type=str, required=True, help='Name of the model (available: model_1).')
    parser.add_argument('--batch_size', type=str, required=True, help='Batch size to train model.')
    parser.add_argument('--duration', type=str, required=True, help='Duration of audio files (files are 4 minutes max).')
    parser.add_argument('--number_of_epochs', type=str, required=True, help='Number of epochs.')

    parser.add_argument('--no_aug', action='store_true', help='Train without data augmentation.')
    parser.add_argument('--freq_aug', action='store_true', help='Train with data augmentation on spectrogram.')
    parser.add_argument('--time_aug', action='store_true', help='Train with data augmentation on raw wave.')
    parser.add_argument('--both_aug', action='store_true', help='Train on both data augmentation techniques.')
    parser.add_argument('--all', action='store_true', help='Train with and without data augmentation.')

    parser.add_argument('--output', type=str, help='Name of the output file.')


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

    # Define augmentation configurations
    augment_configs = [
        {'time_augment': False, 'freq_augment': False, 'suffix': 'no_aug'},
        {'time_augment': True, 'freq_augment': False, 'suffix': 'time_aug'},
        {'time_augment': False, 'freq_augment': True, 'suffix': 'freq_aug'},
        {'time_augment': True, 'freq_augment': True, 'suffix': 'both_aug'},
    ]

    # Loop over models, epochs, batch sizes, and durations
    for name, model_function in models.items():
        print(f"Training {name}")

        for epoch in number_of_epochs:
            print(f"Number of epochs: {epoch}")

            for batch_size in batch_sizes:
                print(f"Batch size: {batch_size}")

                for duration in tqdm(durations):
                    print(f"\nFile duration: {duration}")

                    # Loop over augmentation configurations
                    for aug_config in augment_configs:
                        # Check if the current configuration should be trained
                        if args.all or getattr(args, aug_config['suffix']):
                            print(aug_config['suffix'])

                            # Set up generators with the current augmentation configuration
                            train_gen = AudioDataTrainGenerator(
                                train_data, train_labels, train_idx, batch_size=batch_size,
                                duration=duration, time_augment=aug_config['time_augment'],
                                freq_augment=aug_config['freq_augment']
                            )
                            val_gen = AudioDataTestGenerator(
                                test_data, test_labels, val_idx, batch_size=batch_size,
                                duration=duration
                            )

                            # Initialize and train the model
                            model = model_function(duration=duration)
                            history = model.fit(
                                train_gen, validation_data=val_gen, epochs=epoch,
                                verbose=1, shuffle=False
                            )

                            # Save experiment results
                            experiment = {
                                "model_name": name,
                                "number_of_epochs": epoch,
                                "batch_size": batch_size,
                                "duration": duration,
                                "time_augment": str(aug_config['time_augment']),
                                "freq_augment": str(aug_config['freq_augment']),
                                "train_accuracy": history.history['accuracy'],
                                "test_accuracy": history.history['val_accuracy'],
                                "train_loss": history.history['loss'],
                                "test_loss": history.history['val_loss']
                            }
                            results_json['experiment'].append(experiment)

                            # Save the model
                            model_filename = f"{name}_epoch_{epoch}_batch_size_{batch_size}_duration_{duration}_{aug_config['suffix']}.keras"
                            model.save(model_filename)
           

    json_object = json.dumps(results_json, indent=4)

    if args.output:
        with open(args.output, "w") as res:
            res.write(json_object)       
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')

        with open(f"results_{timestamp}.json", "w") as res:
            res.write(json_object)
        