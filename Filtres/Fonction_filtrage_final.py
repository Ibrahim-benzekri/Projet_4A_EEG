import pandas as pd
import mne
import time
import os

# Paramètres globaux
SFREQ = 256
EEG_COLUMNS = ['EEG-ch1', 'EEG-ch2', 'EEG-ch3', 'EEG-ch4', 'EEG-ch5',
               'EEG-ch6', 'EEG-ch7', 'EEG-ch8', 'EEG-ch9', 'EEG-ch10']

CHANNEL_TYPES = ['eeg'] * len(EEG_COLUMNS)
CHANNEL_MAPPING = {
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
LOW_CUTOFF = 8.0  # Fréquence de coupure basse
HIGH_CUTOFF = 30.0  # Fréquence de coupure haute
REF_CHANNEL = "A2"  # Canal de référence
N_COMPONENTS_ICA = 8  # Nombre de composantes pour ICA
EOG_THRESHOLD = 2.0  # Seuil pour l'identification des artéfacts
MONTAGE_NAME = "standard_1020"  # Nom du montage EEG
output_csv = "C:\\Users\\PC\\Desktop\\EEG\\Filtré\\processed_eeg_final.csv"
n_simples = 256

def preprocess_eeg_data(eeg_data_batch):


    # Vérification des dimensions d'entrée
    if eeg_data_batch.shape != (len(EEG_COLUMNS), n_simples):
        raise ValueError(f"Les données EEG doivent avoir une forme de ({len(EEG_COLUMNS)}, {n_simples})")

    # Création de l'objet Info de MNE
    info = mne.create_info(ch_names=EEG_COLUMNS, sfreq=SFREQ, ch_types=CHANNEL_TYPES)

    # Création du RawArray
    raw = mne.io.RawArray(eeg_data_batch, info)

    # Renommage des canaux selon le standard 10-20
    raw.rename_channels(CHANNEL_MAPPING)

    # Application du montage standard 10-20
    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    raw.set_montage(montage)

    # Filtrage passe-bande (IIR pour éviter les distorsions sur petits signaux)
    raw.filter(l_freq=LOW_CUTOFF, h_freq=HIGH_CUTOFF)

    # Re-référencement à REF_CHANNEL
    raw.set_eeg_reference(ref_channels="average")

    # ICA pour suppression des artéfacts
    ica = mne.preprocessing.ICA(n_components=N_COMPONENTS_ICA, random_state=0)
    ica.fit(raw)

    # Détection automatique des artéfacts (ex : clignements des yeux)
    eog_indices, _ = ica.find_bads_eog(raw, ch_name=REF_CHANNEL, threshold=EOG_THRESHOLD)
    ica.exclude = eog_indices  # Exclusion des composantes détectées
    ica.apply(raw)


    # Extraction des données filtrées
    filtered_data = raw.get_data().T
    filtered_df = pd.DataFrame(filtered_data, columns=EEG_COLUMNS)

    # Ajout au fichier CSV
    file_exists = os.path.exists(output_csv)
    filtered_df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
