import json
import csv
import os

os.chdir('results')

# Load the data from the JSON file
with open('parameter_testing.json', 'r') as file:
    data = json.load(file)

# Prepare to write to the CSV file
with open('parameter_testing.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the CSV header
    writer.writerow(["model_name", "number_of_epochs", "batch_size", "duration", "time_augment", "freq_augment", "train_accuracy", "test_accuracy", "train_loss", "test_loss"])

    # Iterate over the experiments in the JSON data
    for experiment in data["experiment"]:
        writer.writerow([
            experiment["model_name"],
            experiment["number_of_epochs"],
            experiment["batch_size"],
            experiment["duration"],
            experiment["time_augment"],
            experiment["freq_augment"],
            experiment["train_accuracy"][-1],  # Last value of train_accuracy array
            experiment["test_accuracy"][-1],   # Last value of test_accuracy array
            experiment["train_loss"][-1],      # Last value of train_loss array
            experiment["test_loss"][-1]        # Last value of test_loss array
        ])