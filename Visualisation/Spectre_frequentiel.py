import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chemin du fichier CSV filtré
filtered_csv_file = 'C:\\Users\\PC\\Desktop\\EEG\\Codes\\Filtred_Tests\\test1.csv'

# Charger les données filtrées
filtered_data = pd.read_csv(filtered_csv_file)

# Fréquence d'échantillonnage
sfreq = 256  # Échantillons par seconde

# Nombre d'échantillons correspondant aux 7 premières secondes
n_samples = int(10 * sfreq)

# Limiter les données aux 7 premières secondes
limited_data = filtered_data.iloc[4*n_samples:5 * n_samples]

# Sélectionner le canal à analyser
channel_to_plot = 'EEG-ch4'

# Vérifier si le canal existe dans les données
if channel_to_plot not in filtered_data.columns:
    raise ValueError(f"Le canal {channel_to_plot} n'existe pas dans les données.")

# Extraire les données du canal sélectionné
signal = limited_data[channel_to_plot].values

# Calculer la FFT du signal (transformé de Fourier)
fft_values = np.fft.fft(signal)
fft_frequencies = np.fft.fftfreq(len(signal), d=1/sfreq)  # d=1/sfreq correspond à la période d'échantillonnage

# Garder uniquement les fréquences positives
positive_freqs = fft_frequencies[:len(fft_frequencies)//2]
positive_fft = np.abs(fft_values[:len(fft_values)//2])

# Tracer le spectre fréquentiel
plt.figure(figsize=(12, 6))
plt.plot(positive_freqs, positive_fft, color='blue')
plt.title(f'Spectre fréquentiel du signal filtré ({channel_to_plot})', fontsize=16)
plt.xlabel('Fréquence (Hz)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.grid(True)
plt.xlim(0, 50)  # Limite l'affichage aux fréquences utiles (0-50 Hz)
plt.tight_layout()
plt.show()
