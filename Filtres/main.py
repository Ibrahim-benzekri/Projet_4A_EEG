import numpy as np

from Filtres.Fonction_filtrage_final import preprocess_eeg_data


def main():

    # Génération de données EEG fictives
    fake_eeg_data = np.random.randn(10, 256)

    # Lancement du pré-traitement
    preprocess_eeg_data(fake_eeg_data)

if __name__ == "__main__":
    main()
