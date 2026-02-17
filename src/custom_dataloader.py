import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def create_sequences(df, window, feature_cols):
    X, y = [], []
    for _, group in df.groupby("series_id"):
        values = group[feature_cols].values
        
        for i in range(len(values) - window):
            X.append(values[i:i+window])
            y.append(values[i+window][0])
    
    return np.array(X), np.array(y)

class SwimDataset(Dataset):
    def __init__(self, df, seq_len, feature_cols):
        print("Preparing dataset in memory for...")
        self.X, self.y = create_sequences(df, seq_len, feature_cols)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        print(f"Dataset ready: {len(self.X)} sequences loaded into RAM.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Accès instantané
        return self.X[idx], self.y[idx]

def create_sequences_tft(df, window, feature_cols, target_idx=0, horizon=3):
    X, y = [], []
    # On groupe par nageur/série pour ne pas mélanger les historiques
    for _, group in df.groupby("series_id"):
        values = group[feature_cols].values
        
        # On doit s'arrêter assez tôt pour avoir 'horizon' valeurs devant nous
        # Si len = 15, window = 10, horizon = 3 :
        # i peut aller jusqu'à 15 - 10 - 3 = 2 (indices 0, 1, 2)
        for i in range(len(values) - window - horizon + 1):
            # Input : les 'window' lignes précédentes
            X.append(values[i : i + window])
            
            # Target : les 'horizon' valeurs futures de la colonne cible
            # On prend uniquement la colonne target (souvent l'index 0 : perf_temps_sec)
            target_values = values[i + window : i + window + horizon, target_idx]
            y.append(target_values)
    
    return np.array(X), np.array(y)

class SwimDatasetTFT(Dataset):
    def __init__(self, df, seq_len, feature_cols, horizon=3):
        print(f"Preparing Multi-Horizon ({horizon}) dataset in memory...")
        # target_idx=0 car 'perf_temps_sec' est la première de FEATURE_COLS
        self.X, self.y = create_sequences_tft(df, seq_len, feature_cols, target_idx=0, horizon=horizon)
        if len(self.X) == 0:
            raise ValueError("Aucune séquence générée. Vérifiez que vos groupes ont assez de données (window + horizon).")

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        print(f"Dataset ready: {len(self.X)} sequences loaded into RAM.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]