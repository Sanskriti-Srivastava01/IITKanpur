# Complete optimized batch processing code for EEGLAB .set files

import os
import mne
import numpy as np
import concurrent.futures

dataset_folder = "Downloads/eegdata_all"
output_folder = "processed_data1"
os.makedirs(output_folder, exist_ok=True)

def process_set_file(set_file):
    print(f"\n{'='*40}\nProcessing: {set_file}\n{'='*40}")
    try:
        # Load EEG data and metadata
        raw = mne.io.read_raw_eeglab(os.path.join(dataset_folder, set_file), preload=True)

        # Basic preprocessing
        raw.filter(0.5, 40., fir_design='firwin')
        raw.notch_filter(50.)

        # Extract events and remove duplicates
        events, event_id = mne.events_from_annotations(raw)
        _, unique_idx = np.unique(events[:, 0], return_index=True)
        events = events[unique_idx]

        print("Found events:", event_id)

        # Epoch creation
        epochs = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=-0.2, tmax=0.8, baseline=(None, 0),
            preload=True, reject_by_annotation=True
        )
        print(epochs)

        # Save processed data
        epochs.save(f"{output_folder}/{set_file}-epo.fif", overwrite=True)

        # Plotting removed for speed
        return True

    except Exception as e:
        print(f"Error processing {set_file}: {str(e)}")
        return False

set_files = [f for f in os.listdir(dataset_folder) if f.endswith('.set')]
print(f"\nFound {len(set_files)} .set files:")
for idx, f in enumerate(set_files, 1):
    print(f"{idx}. {f}")

# Use ThreadPoolExecutor for parallel processing
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_set_file, set_files))

success_count = sum(results)
print(f"\nProcessing complete. Successfully processed {success_count}/{len(set_files)} files.")
