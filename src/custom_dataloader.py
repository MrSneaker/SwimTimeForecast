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

class SwimDatasetTFTV2(Dataset):
    def __init__(self, df, past_cols, future_cols, static_cols, target_col, seq_len=20, horizon=3):
        self.seq_len = seq_len
        self.horizon = horizon
        
        # Définition stricte des colonnes par type
        self.past_cols = past_cols
        self.future_cols = future_cols
        self.static_cols = static_cols
        self.target_col = target_col

        print(f"--- Preparing SwimDatasetV2 (Horizon: {horizon}) ---")
        self.data_list = self._create_tft_sequences(df)
        print(f"Dataset ready: {len(self.data_list)} sequences generated.")

    def _create_tft_sequences(self, df):
        sequences = []
        # On groupe par series_id (chaque nageur/parcours)
        for _, group in df.groupby("series_id"):
            group_vals = group.reset_index(drop=True)
            n_rows = len(group_vals)
            
            # Il faut assez de données pour le passé + le futur
            if n_rows < (self.seq_len + self.horizon):
                continue
                
            for i in range(n_rows - self.seq_len - self.horizon + 1):
                # 1. Fenêtre Passée (0 à seq_len)
                past_chunk = group_vals.iloc[i : i + self.seq_len]
                
                # 2. Fenêtre Future (seq_len à seq_len + horizon)
                future_chunk = group_vals.iloc[i + self.seq_len : i + self.seq_len + self.horizon]
                
                # 3. Target (valeurs réelles sur l'horizon)
                target_vals = future_chunk[self.target_col].values
                
                # On stocke les données sous forme de dictionnaire pour un accès rapide
                sample = {
                    'past': {col: past_chunk[col].values.astype(np.float32) for col in self.past_cols},
                    'future': {col: future_chunk[col].values.astype(np.float32) for col in self.future_cols},
                    'static': {col: past_chunk[col].iloc[0].astype(np.float32) for col in self.static_cols},
                    'target': target_vals.astype(np.float32)
                }
                sequences.append(sample)
        return sequences

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        s = self.data_list[idx]
        
        # Conversion en tenseurs PyTorch avec les dimensions attendues par TFT :
        # Past/Future: [Sequence_Length, 1]
        # Static: [1]
        
        past_tensor_dict = {
            k: torch.tensor(v).unsqueeze(-1) for k, v in s['past'].items()
        }
        
        future_tensor_dict = {
            k: torch.tensor(v).unsqueeze(-1) for k, v in s['future'].items()
        }
        
        # Pour les statiques, on prend la valeur seule (le TFT les projettera)
        static_tensor_dict = {
            k: torch.tensor([v]) for k, v in s['static'].items()
        }
        
        # Target: [Horizon]
        target = torch.tensor(s['target'])
        
        # Format d'output pour la boucle d'entraînement
        return {
            'past': past_tensor_dict,
            'future': future_tensor_dict,
            'static': static_tensor_dict
        }, target