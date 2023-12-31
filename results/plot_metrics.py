import json
import argparse
import matplotlib.pyplot as plt

def label_name(model_name, time_aug, freq_aug):
    time_aug = True if time_aug == "True" else False
    freq_aug = True if freq_aug == "True" else False

    label = model_name + ' - '
    augmentation = ''
    if not time_aug and not freq_aug:
        augmentation = 'no data augmentation'
    if time_aug and not freq_aug:
        augmentation = 'time data augmentation'
    if not time_aug and freq_aug:
        augmentation = 'frequency data augmentation'
    if time_aug and freq_aug:
        augmentation = 'both data augmentation'

    return label + augmentation

def plot_curves(data, 
                batch_sizes, 
                durations, 
                number_of_epochs, 
                split_curves, 
                hide_legend, 
                no_aug, 
                time_aug, 
                freq_aug, 
                both_aug):
    # Filter data based on provided arguments
    data = data['experiment']

    if batch_sizes != 'all':
        data = [entry for entry in data if entry['batch_size'] in batch_sizes]
    if number_of_epochs != 'all':
        data = [entry for entry in data if entry['number_of_epochs'] in number_of_epochs]
    if durations != 'all':
        data = [entry for entry in data if float(entry['duration']) in durations]
    if not no_aug:
        data = [entry for entry in data if (entry['time_augment'] == "True" or entry['freq_augment'] == "True")]
    if not time_aug:
        data = [entry for entry in data if (entry['time_augment'] == "False" or entry['freq_augment'] == "True")]
    if not freq_aug:
        data = [entry for entry in data if (entry['time_augment'] == "True" or entry['freq_augment'] == "False")]
    if not both_aug:
        data = [entry for entry in data if (entry['time_augment'] == "False" or entry['freq_augment'] == "False")]


    if split_curves:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot train accuracy
        for entry in data:
            label = label_name(entry['model_name'], entry['time_augment'], entry['freq_augment'])
            axs[0, 0].plot(entry['train_accuracy'], label=label)
        axs[0, 0].set_title('Train Accuracy')
        if not hide_legend:
            axs[0, 0].legend()

        # Plot test accuracy
        for entry in data:
            label = label_name(entry['model_name'], entry['time_augment'], entry['freq_augment'])
            axs[0, 1].plot(entry['test_accuracy'], label=label)
        axs[0, 1].set_title('Test Accuracy')
        if not hide_legend:
            axs[0, 1].legend()

        # Plot train loss
        for entry in data:
            label = label_name(entry['model_name'], entry['time_augment'], entry['freq_augment'])
            axs[1, 0].plot(entry['train_loss'], label=label)
        axs[1, 0].set_title('Train Loss')
        if not hide_legend:
            axs[1, 0].legend()

        # Plot test loss
        for entry in data:
            label = label_name(entry['model_name'], entry['time_augment'], entry['freq_augment'])
            axs[1, 1].plot(entry['test_loss'], label=label)
        axs[1, 1].set_title('Test Loss')
        if not hide_legend:
            axs[1, 1].legend()

    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        for entry in data:
            label = label_name(entry['model_name'], entry['time_augment'], entry['freq_augment'])
            ax1.plot(entry['train_accuracy'], label=f"Train Acc - {label}")
            ax1.plot(entry['test_accuracy'], label=f"Test Acc - {label}")
            ax2.plot(entry['train_loss'], label=f"Train Loss - {label}", linestyle='--')
            ax2.plot(entry['test_loss'], label=f"Test Loss - {label}", linestyle='--')

        ax1.set_title('Accuracy')
        ax2.set_title('Loss')
        if not hide_legend:
            ax1.legend()
        if not hide_legend:
            ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot curves from JSON data.')
    parser.add_argument('--batch_size', type=str, required=True, help='Batch size to filter by (8, 16-, 32, 64, 128).')
    parser.add_argument('--duration', type=str, required=True, help='Duration to filter by (1, 1.5, 2, 3 or 4 minutes). Use "all" for all durations.')
    parser.add_argument('--number_of_epochs', type=str, required=True, help='Number of epochs to filter by (5, 10, 15, 20). Use a comma-separated list for multiple values.')
    parser.add_argument('--split_curves', action='store_true', help='Option to split train and test curves into separate plots.')
    parser.add_argument('--hide_legend', action='store_true', help='Hide legend.')
    parser.add_argument('--no_aug', action='store_true', help='Show training without data augmentation.')
    parser.add_argument('--time_aug', action='store_true', help='Show training with time data augmentation.')
    parser.add_argument('--freq_aug', action='store_true', help='Show training with frequency data augmentation.')
    parser.add_argument('--both_aug', action='store_true', help='Show training with both data augmentation techniques.')

    args = parser.parse_args()

    # Load JSON data
    with open('model_6_more_epochs.json', 'r') as f:
        data = json.load(f)

    number_of_epochs = 'all' if args.number_of_epochs == 'all' else [float(epoch) for epoch in args.number_of_epochs.split(',')]
    batch_sizes = 'all' if args.batch_size == 'all' else [float(batch_size) for batch_size in args.batch_size.split(',')]
    durations = 'all' if args.duration == 'all' else [float(duration) for duration in args.duration.split(',')]
    
    plot_curves(data, 
                batch_sizes, 
                durations, 
                number_of_epochs, 
                args.split_curves, 
                args.hide_legend,
                args.no_aug,
                args.time_aug,
                args.freq_aug,
                args.both_aug)