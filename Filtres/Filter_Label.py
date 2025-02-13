import pandas as pd
import mne

# Function to filter EEG data
def filter_eeg_data(input_file, output_file, sfreq=256, l_freq=8, h_freq=30):
    data = pd.read_csv(input_file)
    discard = 7 * sfreq  # 1st 7 seconds to move
    max_records = 42752

    # erasing surplus records if necessary
    if len(data) > max_records:
        data = data.iloc[:max_records].reset_index(drop=True)

    # erasing the first 7s records
    data = data.iloc[discard:].reset_index(drop=True)

    # Step 2: Extract EEG data columns (EEG-ch1 to EEG-ch10)
    # Non-EEG columns :( steady_timestamp, sequence, battery, and flag ) are ignored.

    eeg_columns = ['EEG-ch1', 'EEG-ch2', 'EEG-ch3', 'EEG-ch4', 'EEG-ch5',
                   'EEG-ch6', 'EEG-ch7', 'EEG-ch8', 'EEG-ch9', 'EEG-ch10']

    # data[eeg_columns] for extracting only the 10 channel data from the csv file
    # .to_numpy() converts the extracted DataFrame into a NumPy array which has a shape of (samples, channels).
    # .T Transposes the NumPy array, swapping rows and columns, the shape becomes (channels, samples), which is the required format for MNE library.
    eeg_data = data[eeg_columns].to_numpy().T

    # Step 3: Define channel info
    channel_names = eeg_columns
    channel_types = ['eeg'] * len(channel_names)

    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
    # preparred raw data is storred now in raw variable and ready to be passed to the filter
    raw = mne.io.RawArray(eeg_data, info)

    # Step 4: The band-pass filter

    # Motor imagery tasks, such as imagining or performing right-hand or left-hand movements,
    # are associated with mu rhythms (8–12 Hz) and beta rhythms (13–30 Hz) in the EEG signal.
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    #  Save the filtered data back to a CSV
    filtered_data = raw.get_data().T  # Transpose back to samples x channels
    filtered_df = pd.DataFrame(filtered_data, columns=channel_names)
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtering complete and saved to {output_file}.")

# Function to add labels to EEG data
def add_labels(input_file, output_file, sfreq=256):
    filtered_data = pd.read_csv(input_file)

    # Parameters
    rest = 10 * sfreq  # 10 seconds of rest
    action = 5 * sfreq  # 5 seconds for each action

    # Label computation
    labels = []
    n_records = len(filtered_data)
    time_counter = 0

    while time_counter < n_records:
        # Add periods of rest
        if time_counter + rest <= n_records:
            labels.extend(["Repos"] * rest)
            time_counter += rest
        else:
            labels.extend(["Repos"] * (n_records - time_counter))
            break

        # Add "Gauche"
        if time_counter + action <= n_records:
            labels.extend(["Gauche"] * action)
            time_counter += action
        else:
            labels.extend(["Gauche"] * (n_records - time_counter))
            break

        # Add periods of rest
        if time_counter + rest <= n_records:
            labels.extend(["Repos"] * rest)
            time_counter += rest
        else:
            labels.extend(["Repos"] * (n_records - time_counter))
            break

        # Add "Droite"
        if time_counter + action <= n_records:
            labels.extend(["Droite"] * action)
            time_counter += action
        else:
            labels.extend(["Droite"] * (n_records - time_counter))
            break

    # Verify that the length of labels matches the number of rows in the DataFrame
    assert len(labels) == len(filtered_data), "Error: Label length does not match the number of records."

    # Add the "Label" column to the DataFrame
    filtered_data['Label'] = labels

    # Save the new CSV file
    filtered_data.to_csv(output_file, index=False)

    print(f"File with labels created: {output_file}")
def process_eeg_file(input_file, filtered_file, labeled_file):
    # Step 1: Filter the data
    filter_eeg_data(input_file, filtered_file)

    # Step 2: Add labels to the filtered data
    add_labels(filtered_file, labeled_file)

    print(f"Processing completed. Final output saved to: {labeled_file}")

# Example usage
if __name__ == "__main__":
    input_file = "../EEG.csv"
    filtered_file = "Filtered/filtered_eeg_data.csv"
    labeled_file = "Filtered/labeled_eeg_data.csv"

    process_eeg_file(input_file, filtered_file, labeled_file)
