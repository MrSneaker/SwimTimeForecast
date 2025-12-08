import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from .quantile_loss import QuantileLoss

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =============================================================
# Model Definition
# =============================================================
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
        return self.fc(out[:, -1, :])  # last timestep -> 3 quantiles output


# =============================================================
# Dataset utilities
# =============================================================
SEQ_LEN = 10
FEATURE_COLS = ["perf_temps_sec","nageur_age_mois_scaled","perf_nage_encoded",
             "perf_distance_encoded","perf_bassin_encoded","mois_saison_sin",
             "mois_saison_cos","nageur_sexe_encoded"]
TARGET_COL = "perf_temps_sec"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_sequences(df, window=SEQ_LEN):
    X, y = [], []
    for _, group in df.groupby("series_id"):
        values = group[FEATURE_COLS].values
        
        for i in range(len(values) - window):
            X.append(values[i:i+window])
            y.append(values[i+window][0])
    
    return np.array(X), np.array(y)


class SwimIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, df, seq_len):
        self.df = df
        self.seq_len = seq_len
        self.n_samples = sum(max(0, len(g)-seq_len) for _, g in df.groupby("series_id"))

    def __iter__(self):
        X, y = create_sequences(self.df, self.seq_len)
        for seq, target in zip(X, y):
            yield torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return self.n_samples


# =============================================================
# Training FUNCTION (callable from main.py)
# =============================================================

def train(EPOCHS=5, BATCH_SIZE=128, TRAIN_FRACTION=0.30, VAL_FRACTION=0.30):
    print("\nTRAIN MODE SELECTED\n")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv("../data/performances_cleaned.csv")

    # ---------------- split by swimmers ------------- #
    nageurs = np.random.permutation(data["nageur_id"].unique())
    split = int(0.8 * len(nageurs))

    train_ids = nageurs[:split]
    test_ids  = nageurs[split:]

    train_df_full = data[data["nageur_id"].isin(train_ids)]
    test_df       = data[data["nageur_id"].isin(test_ids)]
    
    os.makedirs("../data/test_data", exist_ok=True)
    test_df.to_csv("../data/test_data/test_df.csv", index=False)
    print(f"Saved test data: {len(test_df)} rows")

    train_nageurs = np.random.permutation(train_df_full["nageur_id"].unique())
    val_split = int(0.9 * len(train_nageurs))

    train_df = train_df_full[train_df_full["nageur_id"].isin(train_nageurs[:val_split])]
    val_df   = train_df_full[train_df_full["nageur_id"].isin(train_nageurs[val_split:])]
    
    # === Scale target column ===
    scaler_y = StandardScaler()
    scaler_y.fit(train_df[[TARGET_COL]])  # fit only on train_df

    train_df[TARGET_COL] = scaler_y.transform(train_df[[TARGET_COL]])
    val_df[TARGET_COL]   = scaler_y.transform(val_df[[TARGET_COL]])

    os.makedirs("../models", exist_ok=True)
    with open("../models/target_scaler.pkl", "wb") as f:
        pickle.dump(scaler_y, f)


    # Random subset
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=42)
    val_df   = val_df.sample(frac=VAL_FRACTION, random_state=42)

    # remove short series
    min_len = SEQ_LEN + 1
    train_df = train_df[train_df.groupby("series_id").series_id.transform('count')>=min_len]
    val_df   = val_df[val_df.groupby("series_id").series_id.transform('count')>=min_len]

    train_loader = DataLoader(SwimIterableDataset(train_df, SEQ_LEN), batch_size=BATCH_SIZE)
    val_loader   = DataLoader(SwimIterableDataset(val_df,   SEQ_LEN), batch_size=BATCH_SIZE)

    print(f"Training samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # ---------------- model + training ---------------- #
    model = SwimLSTM(input_dim=len(FEATURE_COLS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    quantiles = [0.1,0.5,0.9]
    loss_fn = [QuantileLoss(q) for q in quantiles]

    def quantile_loss(pred, y):
        return sum(loss_fn[i](pred[:,i], y) for i in range(3))

    train_hist, val_hist = [], []

    for epoch in range(EPOCHS):
        model.train()
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for Xb, yb in pbar:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = quantile_loss(pred, yb)
            loss.backward()
            optimizer.step()

            total += loss.item()*len(Xb)
            pbar.set_postfix(loss=loss.item())

        avg_train = total/len(train_loader.dataset)
        train_hist.append(avg_train)

        # ---- validation ----
        model.eval()
        total=0
        with torch.no_grad():
            for Xb,yb in val_loader:
                Xb,yb = Xb.to(DEVICE), yb.to(DEVICE)
                loss = quantile_loss(model(Xb), yb)
                total+=loss.item()*len(Xb)

        avg_val = total/len(val_loader.dataset)
        val_hist.append(avg_val)

        print(f"Epoch {epoch+1} | Train {avg_train:.4f} | Val {avg_val:.4f}")


    # ---------------- Save model ---------------- #
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/swim_lstm.pt")
    print("\nModel saved at: ../models/swim_lstm.pt")

    # optional loss curve
    plt.plot(train_hist,label="train")
    plt.plot(val_hist,label="val")
    plt.legend(); plt.grid(); plt.title("Training Curves")
    plt.show()

    print("\nTraining Complete.\n")
