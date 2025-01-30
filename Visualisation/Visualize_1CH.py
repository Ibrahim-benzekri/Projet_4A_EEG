import pandas as pd
import matplotlib.pyplot as plt

# Chemin du fichier CSV filtré
filtered_csv_file = 'C:\\Users\\PC\\Desktop\\EEG\\ENregistrement_artefacts\\filtered_lesDaux.csv'

# Charger les données filtrées
filtered_data = pd.read_csv(filtered_csv_file)

# Fréquence d'échantillonnage
sfreq = 256  # Échantillons par seconde

# Nombre d'échantillons correspondant aux 7 premières secondes
n_samples = int(7 * sfreq)

# Limiter les données aux secondes 7 à 14
limited_data = filtered_data.iloc[:n_samples]

# Sélectionner le canal à afficher
channel_to_plot = 'EEG-ch4'  # Remplacez par le canal souhaité

# Temps limité aux 7 premières secondes
time = [i / sfreq for i in range(n_samples)]

# Vérifier si le canal existe dans les données
if channel_to_plot not in filtered_data.columns:
    raise ValueError(f"Le canal {channel_to_plot} n'existe pas dans les données.")

# Créer le graphique
plt.figure(figsize=(15, 6))
plt.plot(time, limited_data[channel_to_plot], label=channel_to_plot, color='blue')
plt.title(f'Signal EEG [Les deux bruits] pour le canal {channel_to_plot} (pendant 7 secondes d\'enregistrement )', fontsize=16)
plt.xlabel('Temps (s)', fontsize=14)
plt.ylabel('Amplitude (µV)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
