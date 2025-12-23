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
        print("Preparing dataset in memory...")
        self.X, self.y = create_sequences(df, seq_len, feature_cols)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        print(f"Dataset ready: {len(self.X)} sequences loaded into RAM.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Accès instantané
        return self.X[idx], self.y[idx]