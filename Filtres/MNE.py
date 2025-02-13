import pandas as pd
import mne
from mne.preprocessing import create_eog_epochs

# Step 1: Load the CSV file from the disk
csv_file = 'C:\\Users\\PC\\Desktop\\EEG\\ENregistrement_artefacts\\Oculaire.csv'  # to replace with our file path
data = pd.read_csv(csv_file)

# Step 2: Extract EEG data columns (EEG-ch1 to EEG-ch10)
eeg_columns = ['EEG-ch1', 'EEG-ch2', 'EEG-ch3', 'EEG-ch4', 'EEG-ch5',
               'EEG-ch6', 'EEG-ch7', 'EEG-ch8', 'EEG-ch9', 'EEG-ch10']

eeg_data = data[eeg_columns].to_numpy().T

# Sampling frequency
sfreq = 256

# Step 3: Define channel info
channel_names = eeg_columns
channel_types = ['eeg'] * len(channel_names)

info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)

# Create RawArray
raw = mne.io.RawArray(eeg_data, info)

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

# Step 5: The band-pass filter
low_cutoff = 8
high_cutoff = 30.0
raw.filter(l_freq=low_cutoff, h_freq=high_cutoff)

# Step 6: Referencing (average reference)
raw.set_eeg_reference(ref_channels=["A2"])


"""
# Step 7: ICA
ica = mne.preprocessing.ICA(n_components=8, random_state=0)
ica.fit(raw)

#visualisation des composants
#ica.plot_components()

#elimination des composants en les comparants par rapport a la reference A2.
#ica.find_bads_eog(raw,'A2',threshold=2)
ica.exclude = [1,2,3,7]  # Exclure les composantes
ica.apply(raw)

#ica.plot_sources(raw)

ica.plot_properties(raw, picks=0)
ica.plot_properties(raw, picks=1)
ica.plot_properties(raw, picks=2)
ica.plot_properties(raw, picks=3)
ica.plot_properties(raw, picks=4)
ica.plot_properties(raw, picks=5)
ica.plot_properties(raw, picks=6)
ica.plot_properties(raw, picks=7)


filtered_data = raw.get_data().T  # Transpose back to samples x channels
filtered_df = pd.DataFrame(filtered_data, columns=channel_names)
filtered_df.to_csv('C:\\Users\\PC\\Desktop\\EEG\\ENregistrement_artefacts\\ICA\\Oculaire.csv', index=False)

"""

