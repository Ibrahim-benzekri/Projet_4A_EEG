import pandas as pd
import mne
import time

# Step 1: Load the CSV file from the disk
csv_file = 'C:\\Users\\PC\\Desktop\\EEG\\ENregistrement_artefacts\\ARTF\\Oculaire2.csv'  # to replace with our file path
data = pd.read_csv(csv_file)

# Step 2: Extract EEG data columns (EEG-ch1 to EEG-ch10)
eeg_columns = ['EEG-ch1', 'EEG-ch2', 'EEG-ch3', 'EEG-ch4', 'EEG-ch5',
               'EEG-ch6', 'EEG-ch7', 'EEG-ch8', 'EEG-ch9', 'EEG-ch10']

eeg_data = data[eeg_columns].to_numpy().T

# Sampling frequency
sfreq = 256

# Limiter les données aux secondes 1s
limited_data = eeg_data.iloc[:sfreq]

# Step 3: Define channel info
channel_names = eeg_columns
channel_types = ['eeg'] * len(channel_names)

info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)

# Create RawArray
raw = mne.io.RawArray(limited_data, info)

# Step 4: Map  channel names to standard 10-20 names
# We need to define which of our channels correspond to which standard 10-20 electrodes.
channel_mapping = {
    'EEG-ch1': 'FC3',
    'EEG-ch2': 'FCz',
    'EEG-ch3': 'FC4',
    'EEG-ch4': 'C3',
    'EEG-ch5': 'Cz',
    'EEG-ch6': 'C4',
    'EEG-ch7': 'CP3',
    'EEG-ch8': 'CPz',
    'EEG-ch9': 'CP4',
    'EEG-ch10': 'A2',
}

# Rename channels
raw.rename_channels(channel_mapping)

# Set the standard 10-20 montage
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

#starting time:
start_time = time.time()

# Step 5: The band-pass filter
low_cutoff = 1
high_cutoff = 40.0
raw.filter(l_freq=low_cutoff, h_freq=high_cutoff)

# Step 6: Referencing (average reference)
raw.set_eeg_reference(ref_channels=["A2"])

# Step 7: ICA
ica = mne.preprocessing.ICA(n_components=8, random_state=0)
ica.fit(raw)
ica.exclude = [0,1,7]  # Exclure les composantes
ica.apply(raw)

#temps de calcul
processing_time = (time.time() - start_time ) * 1000

print(f"Processing time : {processing_time:.2f} milliseconds")


