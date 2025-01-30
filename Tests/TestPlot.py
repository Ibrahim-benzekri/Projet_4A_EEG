import pandas as pd
import numpy as np
import mne

# Step 1: Load the CSV file from the disk
csv_file = 'C:\\Users\\PC\\Desktop\\EEG\\Codes\\EEG.csv'  # to replace with our file path
data = pd.read_csv(csv_file)

# Sampling frequency: The sampling rate, is indicated in the documentation of the headset, it's 256 SPS (Samples Per Second).
#This means the headset records 256 data points per second for each EEG channel.
sfreq = 256

discard = 7 * sfreq  # 1st 7 seconds to move
max_records = 42752

# erasing surplus records if necessary
if len(data) > max_records:
    data = data.iloc[:max_records].reset_index(drop=True)

# erasing the first 7s records
data = data.iloc[discard:].reset_index(drop=True)


# Step 2: Extract EEG data columns (EEG-ch1 to EEG-ch10)
#Non-EEG columns :( steady_timestamp, sequence, battery, and flag ) are ignored.

eeg_columns = ['EEG-ch1', 'EEG-ch2', 'EEG-ch3', 'EEG-ch4', 'EEG-ch5',
               'EEG-ch6', 'EEG-ch7', 'EEG-ch8', 'EEG-ch9', 'EEG-ch10']

#data[eeg_columns] for extracting only the 10 channel data from the csv file
#.to_numpy() converts the extracted DataFrame into a NumPy array which has a shape of (samples, channels).
# .T Transposes the NumPy array, swapping rows and columns, the shape becomes (channels, samples), which is the required format for MNE library.
eeg_data = data[eeg_columns].to_numpy().T



# Step 3: Define channel info
channel_names = eeg_columns
channel_types = ['eeg'] * len(channel_names)

info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
# preparred raw data is storred now in raw variable and ready to be passed to the filter
raw = mne.io.RawArray(eeg_data, info)

# Step 4: The band-pass filter

#Motor imagery tasks, such as imagining or performing right-hand or left-hand movements,
#are associated with mu rhythms (8–12 Hz) and beta rhythms (13–30 Hz) in the EEG signal.

low_cutoff = 8.0
high_cutoff = 30.0
raw.filter(l_freq=low_cutoff, h_freq=high_cutoff)

raw.plot();
