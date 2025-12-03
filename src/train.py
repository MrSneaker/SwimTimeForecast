from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from quantile_loss import QuantileLoss

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEQ_LEN = 10
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = [
    "perf_nage_encoded",
    "nageur_sexe_encoded",
    "nageur_age_mois_scaled",
    "perf_distance_encoded",
    "perf_bassin_encoded",
    "mois_saison_sin",
    "mois_saison_cos",
    "perf_temps_sec",
]
TARGET_COL = "perf_temps_sec"

data = pd.read_csv("../data/performances_cleaned.csv")

# Split par nageur
nageurs = data["nageur_id"].unique()
np.random.shuffle(nageurs)
split = int(0.8 * len(nageurs))

train_ids, test_ids = nageurs[:split], nageurs[split:]

train_df_full = data[data["nageur_id"].isin(train_ids)]
test_df       = data[data["nageur_id"].isin(test_ids)]

train_nageurs = train_df_full["nageur_id"].unique()
np.random.shuffle(train_nageurs)

val_ratio = 0.1
val_split = int((1 - val_ratio) * len(train_nageurs))

train_ids_final = train_nageurs[:val_split]
val_ids         = train_nageurs[val_split:]

train_df = train_df_full[train_df_full["nageur_id"].isin(train_ids_final)]
val_df   = train_df_full[train_df_full["nageur_id"].isin(val_ids)]

scaler_y = StandardScaler()
train_df["target_scaled"] = scaler_y.fit_transform(train_df[[TARGET_COL]])
val_df["target_scaled"]   = scaler_y.transform(val_df[[TARGET_COL]])
test_df["target_scaled"]  = scaler_y.transform(test_df[[TARGET_COL]])

TRAIN_FRACTION = 0.30
VAL_FRACTION   = 0.30
TEST_FRACTION  = 0.30

train_df_small = train_df.sample(frac=TRAIN_FRACTION, random_state=42)
val_df_small   = val_df.sample(frac=VAL_FRACTION, random_state=42)
test_df_small  = test_df.sample(frac=TEST_FRACTION, random_state=42)

min_length = SEQ_LEN + 1  # +1 because we need target after the sequence

train_series_lengths = train_df_small.groupby("series_id").size()
train_valid_series = train_series_lengths[train_series_lengths >= min_length].index
val_series_lengths = val_df_small.groupby("series_id").size()
val_valid_series = val_series_lengths[val_series_lengths >= min_length].index
test_series_lengths = test_df_small.groupby("series_id").size()
test_valid_series = test_series_lengths[test_series_lengths >= min_length].index

# Filtrer le DataFrame
train_df_filtered = train_df_small[train_df_small["series_id"].isin(train_valid_series)].copy()
val_df_filtered   = val_df_small[val_df_small["series_id"].isin(val_valid_series)].copy()
test_df_filtered  = test_df_small[test_df_small["series_id"].isin(test_valid_series)].copy()

print(f"Train: {len(train_df_filtered)} | Val: {len(val_df_filtered)} | Test: {len(test_df_filtered)}")
            
def create_sequences(df, window=SEQ_LEN):
    X, y = [], []

    for _, group in df.groupby("series_id"):
        values = group[[
            "perf_temps_sec",
            "nageur_age_mois_scaled",
            "perf_nage_encoded",
            "perf_distance_encoded",
            "perf_bassin_encoded",
            "mois_saison_sin",
            "mois_saison_cos",
            "nageur_sexe_encoded"
        ]].values
        
        for i in range(len(values) - window):
            X.append(values[i:i+window])
            y.append(values[i+window][0])  # perf_temps_sec next step
    return np.array(X), np.array(y)

class SwimIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, df, seq_len):
        self.df = df
        self.seq_len = seq_len

    def __iter__(self):
        X, y = create_sequences(self.df, self.seq_len)
        for seq, target in zip(X, y):
            yield torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return sum(max(0, len(g) - self.seq_len) for _, g in self.df.groupby("series_id"))


# train_loader = DataLoader(SwimIterableDataset(train_df, SEQ_LEN), batch_size=BATCH_SIZE)
# val_loader   = DataLoader(SwimIterableDataset(val_df,   SEQ_LEN), batch_size=BATCH_SIZE)
# test_loader  = DataLoader(SwimIterableDataset(test_df,  SEQ_LEN), batch_size=BATCH_SIZE)

train_loader = DataLoader(SwimIterableDataset(train_df_filtered, SEQ_LEN), batch_size=BATCH_SIZE)
val_loader   = DataLoader(SwimIterableDataset(val_df_filtered,   SEQ_LEN), batch_size=BATCH_SIZE)
test_loader  = DataLoader(SwimIterableDataset(test_df_filtered,  SEQ_LEN), batch_size=BATCH_SIZE)

for X_batch, y_batch in train_loader:
    print(X_batch.shape, y_batch.shape)
    break

class SwimLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]       # last timestep
        out = self.fc(out)        # shape: (batch, 3)
        return out
    
model = SwimLSTM(input_dim=len(FEATURE_COLS)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

EPOCHS = 5

train_history = []
val_history   = []

quantiles = [0.1, 0.5, 0.9]
criterion_list = [QuantileLoss(q) for q in quantiles]

def quantile_loss(preds, true):
    return (
        criterion_list[0](preds[:, 0], true) +
        criterion_list[1](preds[:, 1], true) +
        criterion_list[2](preds[:, 2], true)
    )

for epoch in range(EPOCHS):

    # -------------------- TRAIN --------------------
    model.train()
    total_train = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train")

    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        preds = model(X_batch)              # shape (batch, 3)
        loss = quantile_loss(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_train += loss.item() * len(X_batch)
        pbar.set_postfix(train_loss=loss.item())

    avg_train = total_train / len(train_loader.dataset)
    train_history.append(avg_train)

    # -------------------- VALIDATION --------------------
    model.eval()
    total_val = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            preds = model(X_batch)
            loss = quantile_loss(preds, y_batch)
            total_val += loss.item() * len(X_batch)

    avg_val = total_val / len(val_loader.dataset)
    val_history.append(avg_val)

    print(f"Epoch {epoch+1} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

plt.figure(figsize=(8,4))
plt.plot(train_history, label="Train loss")
plt.plot(val_history, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Pinball loss (sum of quantiles)")
plt.title("Training curves")
plt.legend()
plt.grid()
plt.show()

# Sauvegarde
torch.save(model.state_dict(), "../models/swim_lstm.pt")
print("Modèle sauvegardé")

model.eval()
preds_list = []
trues_list = []

with torch.no_grad():
    for X_batch, y_batch in tqdm(test_loader, desc="Testing"):
        X_batch = X_batch.to(DEVICE)
        preds = model(X_batch).cpu().numpy()  # shape (batch, 3)

        preds_list.append(preds)
        trues_list.append(y_batch.numpy())

preds = np.vstack(preds_list)
trues = np.concatenate(trues_list)

q10 = scaler_y.inverse_transform(preds[:, 0:1]).squeeze()
q50 = scaler_y.inverse_transform(preds[:, 1:2]).squeeze()
q90 = scaler_y.inverse_transform(preds[:, 2:3]).squeeze()
true = scaler_y.inverse_transform(trues.reshape(-1,1)).squeeze()

q10 = np.minimum(q10, q50)
q90 = np.maximum(q90, q50)

plt.figure(figsize=(14,6))
plt.plot(true, label="True")
plt.plot(q50, label="Median q50")
plt.fill_between(np.arange(len(q50)), q10, q90, alpha=0.2, label="90% band")
plt.legend()
plt.grid()
plt.title("Quantile LSTM prediction")
plt.show()
