import pandas as pd
import matplotlib.pyplot as plt

# Chemin du fichier CSV filtré
filtered_csv_file = 'C:\\Users\\PC\\Desktop\\EEG\\ENregistrement_artefacts\\Passe-bande\\musculaire2.csv'

# Charger les données filtrées
filtered_data = pd.read_csv(filtered_csv_file)

# Liste des colonnes EEG
eeg_columns = ['EEG-ch1', 'EEG-ch2', 'EEG-ch3', 'EEG-ch4', 'EEG-ch5',
               'EEG-ch6', 'EEG-ch7', 'EEG-ch8', 'EEG-ch9']

# Fréquence d'échantillonnage
sfreq = 256  # Échantillons par seconde

# Nombre d'échantillons correspondant aux 7 premières secondes
n_samples = int(7 * sfreq)

# Limiter les données aux 7 premières secondes
limited_data = filtered_data.iloc[n_samples:2*n_samples]

# Créer une figure pour les tracés
plt.figure(figsize=(15, 10))

# Temps limité aux 7 premières secondes
time = [i / sfreq for i in range(n_samples)]

# Tracer chaque canal EEG dans un subplot séparé
for i, channel in enumerate(eeg_columns):
    plt.subplot(len(eeg_columns), 1, i + 1)
    plt.plot(time, limited_data[channel], label=channel)
    plt.ylabel('Amplitude (µV)')
    plt.legend(loc='upper right')
    plt.grid(True)
    if i == 0:
        plt.title('Signaux EEG filtrés (7 premières secondes)')
    if i == len(eeg_columns) - 1:
        plt.xlabel('Temps (s)')

# Ajuster l'espacement entre les sous-graphiques
plt.tight_layout()
plt.show()
