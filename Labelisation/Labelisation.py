import pandas as pd

# Load the filtered CSV file
filtered_data = pd.read_csv('notre fichier filtré')

# Parameters
sfreq = 256  # Sampling frequency (256 samples/second)
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
filtered_data.to_csv('filtered_eeg_data_with_labels.csv', index=False)

print("File with labels created: 'filtered_eeg_data_with_labels.csv'")
