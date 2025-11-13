import torch
from torch.utils.data import Dataset, DataLoader

class SwimDataset(Dataset):
    def __init__(self, df, seq_len, feature_cols, target_col):
        self.df = df
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.groups = list(df.groupby("nageur_id"))

    def __len__(self):
        return sum(max(0, len(g) - self.seq_len) for _, g in self.groups)

    def __getitem__(self, idx):
        # pas d’accès aléatoire efficace, donc on itère (utilise batch_size=1 pour un premier test)
        raise NotImplementedError("Utilise plutôt un DataLoader avec generator() pour stream")

def swim_generator(df, seq_len, feature_cols, target_col):
    for nageur_id, group in df.groupby("nageur_id"):
        group = group.sort_values("perf_date")
        if len(group) <= seq_len:
            continue
        data_values = torch.tensor(group[feature_cols].values, dtype=torch.float32)
        target_values = torch.tensor(group[target_col].values, dtype=torch.float32)
        for i in range(seq_len, len(group)):
            yield data_values[i-seq_len:i], target_values[i]
