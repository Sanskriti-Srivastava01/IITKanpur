import os
import mne
import matplotlib.pyplot as plt

# Path to processed_data folder (assumes it's in the root directory)
processed_folder = os.path.join(os.path.dirname(__file__), '..', 'results')

# Check if folder exists
if not os.path.exists(processed_folder):
    raise FileNotFoundError(f"Folder '{processed_folder}' does not exist. Please check the path.")

# Get all epoch files
epoch_files = [f for f in os.listdir(processed_folder) if f.endswith('-epo.fif')]
if not epoch_files:
    raise ValueError(f"No '-epo.fif' files found in '{processed_folder}'.")

# Load and visualize each file
for epo_file in epoch_files:
    file_path = os.path.join(processed_folder, epo_file)
    print(f"\n--- Visualizing {epo_file} ---")
    epochs = mne.read_epochs(file_path, preload=True)

    # Plot an overview of the epochs
    epochs.plot(block=False)

    # Plot ERP per condition
    for condition in epochs.event_id:
        evoked = epochs[condition].average()
        fig = evoked.plot(spatial_colors=True, show=False)
        plt.suptitle(f"ERP - {condition} - {epo_file}")
        plt.show()

    # Plot event distribution
    mne.viz.plot_events(epochs.events, sfreq=epochs.info['sfreq'], event_id=epochs.event_id, show=True)
    plt.title(f"Event Distribution - {epo_file}")
    plt.show()
