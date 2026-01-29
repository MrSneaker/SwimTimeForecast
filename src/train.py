import os
import json
import pickle
from regex import E
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from .custom_dataloader import SwimDataset

from .quantile_loss import QuantileLoss
from .early_stopper import EarlyStopper

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =============================================================
# Model Definition
# =============================================================
class SwimLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.4):
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


# =============================================================
# Training FUNCTION (callable from main.py)
# =============================================================

def train(EPOCHS=5, BATCH_SIZE=128, TRAIN_FRACTION=0.30, VAL_FRACTION=0.30, use_optimized=False):
    print("\nTRAIN MODE SELECTED\n")
    
    if use_optimized:
        print("Loading optimized hyperparameters from best_params.json")
        with open("../models/best_params.json", "r") as f:
            best_params = json.load(f)
        BATCH_SIZE = best_params["batch_size"]
        LR = best_params["lr"]
        HIDDEN_DIM = best_params["hidden_dim"]
        NUM_LAYERS = best_params["num_layers"]
        DROPOUT = best_params["dropout"]
        print(f"Using parameters: Batch Size={BATCH_SIZE}, LR={LR}, Hidden Dim={HIDDEN_DIM}, Num Layers={NUM_LAYERS}, Dropout={DROPOUT}")
    else:
        LR = 5e-4
        HIDDEN_DIM = 256
        NUM_LAYERS = 4
        DROPOUT = 0.4

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv("../data/performances_cleaned.csv")
    
    print(f"Données avant filtrage : {data.shape[0]} lignes")

    # On définit les bornes (ici on garde ce qui est entre 0.5% et 99.5%)
    q_low = data[TARGET_COL].quantile(0.001)
    q_high = data[TARGET_COL].quantile(0.999)

    # On filtre
    data_filtered = data[(data[TARGET_COL] > q_low) & (data[TARGET_COL] < q_high)]
    
    removed_count = data.shape[0] - data_filtered.shape[0]
    data = data_filtered
    
    print(f"Seuils appliqués : < {q_low:.2f}s et > {q_high:.2f}s")
    print(f"Données après filtrage : {data.shape[0]} lignes ({removed_count} supprimées)")

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

    train_dataset = SwimDataset(train_df, SEQ_LEN, FEATURE_COLS)
    val_dataset   = SwimDataset(val_df, SEQ_LEN, FEATURE_COLS)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,         
        num_workers=4,          
        pin_memory=True,        
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Training samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # ---------------- model + training ---------------- #
    model = SwimLSTM(input_dim=len(FEATURE_COLS), hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    # model = torch.compile(model, mode="reduce-overhead") # Uncomment if using PyTorch 2.0+ with python<=3.11
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    quantiles = [0.1,0.5,0.9]
    loss_fn = [QuantileLoss(q) for q in quantiles]

    def quantile_loss(pred, y):
        return sum(loss_fn[i](pred[:,i], y) for i in range(3))

    train_hist, val_hist = [], []
    early_stopper = EarlyStopper(patience=7, min_delta=0.0001)

    best_model_path = "../models/swim_lstm_best.pt"

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        
        stop_training, save_model = early_stopper.check(avg_val)

        if save_model:
            torch.save(model.state_dict(), best_model_path)
            print(f"   >>> New Best Model Saved (Val Loss: {avg_val:.4f})")
        if stop_training:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        val_hist.append(avg_val)

        print(f"Epoch {epoch+1} | Train {avg_train:.4f} | Val {avg_val:.4f}")


    # ---------------- Save model ---------------- #
    os.makedirs("../models", exist_ok=True)
    
    if os.path.exists(best_model_path):
        print("Loading best model for final save...")
        model.load_state_dict(torch.load(best_model_path))
        torch.save(model.state_dict(), "../models/swim_lstm.pt")
    print("\nModel saved at: ../models/swim_lstm.pt")

    plt.plot(train_hist,label="train")
    plt.plot(val_hist,label="val")
    plt.legend(); plt.grid(); plt.title("Training Curves")
    date_now = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    training_id = f"E{EPOCHS}_B{BATCH_SIZE}_H{HIDDEN_DIM}_L{NUM_LAYERS}_D{DROPOUT}_LR{LR}_{date_now}"
    plt.savefig(f"../model_result/training_curve_{training_id}.png")
    plt.show()

    print("\nTraining Complete.\n")
