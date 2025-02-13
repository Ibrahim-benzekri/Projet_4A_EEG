import numpy as np
import pandas as pd

from Filtres.Fonction_filtrage_final import preprocess_eeg_data

# Définition des colonnes EEG
eeg_columns = ['EEG-ch1', 'EEG-ch2', 'EEG-ch3', 'EEG-ch4', 'EEG-ch5',
               'EEG-ch6', 'EEG-ch7', 'EEG-ch8', 'EEG-ch9', 'EEG-ch10']

def main():
    # Charger le fichier CSV
    csv_file = 'C:\\Users\\PC\\Desktop\\EEG\\ENregistrement_artefacts\\ARTF\\Oculaire2.csv'
    data = pd.read_csv(csv_file)

    # Extraction des données EEG sous forme de numpy array (chx échantillons)
    eeg_data = data[eeg_columns].to_numpy().T  # (10, N)

    # Sélection des 256 premiers échantillons
    limited_data = eeg_data[:, :256]  # (10, 256)

    print("Shape of limited_data:", limited_data.shape)  # Vérification

    # Lancement du pré-traitement
    preprocess_eeg_data(limited_data)

if __name__ == "__main__":
    main()
