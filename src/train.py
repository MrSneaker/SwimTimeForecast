from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEQ_LEN = 30
BATCH_SIZE = 256
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

train_df = data[data["nageur_id"].isin(train_ids)]
test_df  = data[data["nageur_id"].isin(test_ids)]

print(f"Train : {len(train_df)} lignes | Test : {len(test_df)} lignes")

scaler_y = StandardScaler()
train_df["target_scaled"] = scaler_y.fit_transform(train_df[[TARGET_COL]])
test_df["target_scaled"]  = scaler_y.transform(test_df[[TARGET_COL]])

def swim_sequence_generator(df, seq_len=SEQ_LEN):
    for nageur_id, group in df.groupby("nageur_id"):
        group = group.sort_values("perf_date")
        if len(group) <= seq_len:
            continue

        data_values = group[FEATURE_COLS].values.astype(np.float32)
        target_values = group["target_scaled"].values.astype(np.float32)

        for i in range(seq_len, len(group)):
            seq = data_values[i - seq_len:i]
            target = target_values[i]
            yield torch.tensor(seq), torch.tensor(target)

class SwimIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, df, seq_len):
        self.df = df
        self.seq_len = seq_len
    def __iter__(self):
        for seq, target in swim_sequence_generator(self.df, self.seq_len):
            yield seq, target
    def __len__(self):
        return sum(max(0, len(g) - self.seq_len) for _, g in self.df.groupby("nageur_id"))

train_loader = DataLoader(SwimIterableDataset(train_df, SEQ_LEN), batch_size=BATCH_SIZE)
test_loader  = DataLoader(SwimIterableDataset(test_df,  SEQ_LEN), batch_size=BATCH_SIZE)


class SwimLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # dernière sortie temporelle
        out = self.fc(out)
        return out.squeeze()

model = SwimLSTM(input_dim=len(FEATURE_COLS)).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for X_batch, y_batch in progress:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"→ Epoch {epoch+1} | Loss moyenne : {avg_loss:.6f}")

# Sauvegarde
torch.save(model.state_dict(), "../models/swim_lstm.pt")
print("Modèle sauvegardé")

# Évaluation
model.eval()
preds, trues = [], []

with torch.no_grad():
    for X_batch, y_batch in tqdm(test_loader, desc="Évaluation"):
        X_batch = X_batch.to(DEVICE)
        y_pred = model(X_batch).cpu().numpy()
        preds.extend(y_pred)
        trues.extend(y_batch.numpy())

preds = np.array(preds)
trues = np.array(trues)


from sklearn.metrics import mean_absolute_error, root_mean_squared_error

preds = np.array(preds).reshape(-1, 1)
trues = np.array(trues).reshape(-1, 1)

preds_real = scaler_y.inverse_transform(preds)
trues_real = scaler_y.inverse_transform(trues)


rmse = root_mean_squared_error(trues_real, preds_real)
mae = mean_absolute_error(trues_real, preds_real)

print(f"\nRésultats de test :")
print(f"  RMSE = {rmse:.4f}")
print(f"  MAE  = {mae:.4f}")


plt.figure(figsize=(10,5))
plt.scatter(trues[:500], preds[:500], alpha=0.5)
plt.plot([min(trues), max(trues)], [min(trues), max(trues)], 'r--')
plt.xlabel("Valeur réelle (perf_temps_sec)")
plt.ylabel("Prédiction")
plt.title("Comparaison des prédictions vs valeurs réelles")
plt.grid(True)
plt.tight_layout()
plt.show()
