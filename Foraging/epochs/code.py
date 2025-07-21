import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# --- User Parameters ---
set_folder = "Downloads/eegdata_all"
ced_file = "Downloads/live_amp_32.ced"
output_folder = "processed_results"
os.makedirs(output_folder, exist_ok=True)

# --- Load CED file ---
ced_data = np.genfromtxt(ced_file, dtype=None, encoding='utf-8', names=True)
ced_ch_names = [str(ch) for ch in ced_data['labels']]
ced_positions = np.column_stack((ced_data['X'], ced_data['Y'], ced_data['Z']))

# --- Manual mapping for the 5 non-standard channels ---
non_standard_channels = ['F9', 'P9', 'Oz', 'P10', 'F10']
remaining_ced_channels = ['IO', 'FT9', 'FT10', 'TP9', 'TP10']
manual_mapping = dict(zip(non_standard_channels, remaining_ced_channels))

def create_stay_leave_epochs(events, event_id, raw, tmin=-1.0, tmax=0.0, baseline=(None, 0)):
    c_code = event_id.get('c')
    n_code = event_id.get('N')
    if c_code is None or n_code is None:
        raise ValueError("Both 'C' and 'N' must be present in event_id.")

    c_indices = np.where(events[:, 2] == c_code)[0]
    leave_events = []
    stay_events = []

    for idx in c_indices:
        if idx + 1 < len(events) and events[idx + 1, 2] == n_code:
            leave_events.append(events[idx])
        else:
            stay_events.append(events[idx])

    leave_events = np.array(leave_events)
    stay_events = np.array(stay_events)

    epoch_params = dict(tmin=tmin, tmax=tmax, baseline=baseline, preload=True, reject_by_annotation=True)
    stay_epochs = mne.Epochs(raw, stay_events, event_id={'Stay': c_code}, **epoch_params) if len(stay_events) > 0 else None
    leave_epochs = mne.Epochs(raw, leave_events, event_id={'Leave': c_code}, **epoch_params) if len(leave_events) > 0 else None

    print(f"Created Stay epochs: {len(stay_epochs) if stay_epochs else 0}")
    print(f"Created Leave epochs: {len(leave_epochs) if leave_epochs else 0}")

    return stay_epochs, leave_epochs

def plot_condition(epochs, condition, set_file, output_folder):
    if epochs is None or len(epochs) == 0:
        print(f"{set_file}: No {condition} epochs found.")
        return

    evoked = epochs.average()
    mean_data = np.mean(np.abs(evoked.data), axis=0)
    peak_idx = np.argmax(mean_data)
    peak_time = evoked.times[peak_idx]

    # Create two axes: one for the topomap, one for the colorbar
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[10, 1])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    # Pass both axes as a tuple
    evoked.plot_topomap(times=[peak_time], axes=(ax, cax), show=False)
    ax.set_title(f"{condition} - Peak at {peak_time:.3f}s")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{set_file[:-4]}_{condition}_topo.png"))
    plt.close(fig)
    print(f"{set_file}: {condition} topomap saved at {peak_time:.3f}s.")

# --- Process each .set file ---
set_files = [f for f in os.listdir(set_folder) if f.endswith('.set')]

for set_file in set_files:
    try:
        print(f"\nProcessing {set_file}...")
        raw = mne.io.read_raw_eeglab(os.path.join(set_folder, set_file), preload=True)
        
        # 1. Manual renaming for non-standard channels
        raw.rename_channels(manual_mapping)
        
        # 2. Case-insensitive renaming for other channels
        rename_map = {ch: ced_ch for ch in raw.ch_names for ced_ch in ced_ch_names if ch.lower() == ced_ch.lower()}
        raw.rename_channels(rename_map)
        
        # 3. Remove non-EEG channels
        raw.pick_types(eeg=True)
        
        # 4. Set montage
        montage = mne.channels.make_dig_montage(
            ch_pos=dict(zip(ced_ch_names, ced_positions)),
            coord_frame='head'
        )
        raw.set_montage(montage)
        
        # 5. Event processing
        events, event_id = mne.events_from_annotations(raw)
        _, unique_idx = np.unique(events[:, 0], return_index=True)
        events = events[unique_idx]
        
        if 'c' not in event_id or 'N' not in event_id:
            print(f"{set_file}: Missing 'C' or 'N' events")
            continue
        
        # 6. Create Stay/Leave epochs
        stay_epochs, leave_epochs = create_stay_leave_epochs(events, event_id, raw, tmin=-1, tmax=0)
        
        # 7. Plot and save
        plot_condition(stay_epochs, "Stay", set_file, output_folder)
        plot_condition(leave_epochs, "Leave", set_file, output_folder)
        
        # 8. Save epochs
        if stay_epochs is not None:
            stay_epochs.save(os.path.join(output_folder, f"{set_file[:-4]}-stay-epo.fif"), overwrite=True)
        if leave_epochs is not None:
            leave_epochs.save(os.path.join(output_folder, f"{set_file[:-4]}-leave-epo.fif"), overwrite=True)

    except Exception as e:
        print(f"Error processing {set_file}: {str(e)}")
