import os
import json
import pickle
from pyexpat import model
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from .SwimTFT import SwimTFT

from .custom_dataloader import SwimDataset, SwimDatasetTFT
from .quantile_loss import MultiHorizonQuantileLoss, QuantileLoss, QuantilesLoss
from .early_stopper import EarlyStopper

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =============================================================
# Model V1: Standard LSTM
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
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)
        # On prend juste le dernier état caché
        return self.fc(out[:, -1, :])

# =============================================================
# Model V2: LSTM + Multi-Head Self-Attention
# =============================================================
class SwimAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.4, n_heads=4):
        super().__init__()
        
        # 1. Encodeur LSTM (Capture la séquentialité locale)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        
        # 2. Multi-Head Attention (Capture les relations globales)
        # Note : hidden_dim doit être divisible par n_heads
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        
        # 3. Normalisation (Aide à la convergence)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 4. Tête de sortie
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        # x: [batch, seq_len, features]
        
        # Sortie LSTM : [batch, seq_len, hidden_dim]
        lstm_out, _ = self.lstm(x) 
        
        # Self-Attention : Query=Key=Value=lstm_out
        # attn_output : [batch, seq_len, hidden_dim]
        # attn_weights : [batch, seq_len, seq_len] (non utilisé ici)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        
        # Connexion Résiduelle + Normalisation
        # Permet de garder l'info séquentielle du LSTM tout en ajoutant le contexte de l'attention
        out = self.norm(lstm_out + attn_out)
        
        # On prend le dernier timestep qui contient maintenant le contexte enrichi par l'attention
        return self.fc(out[:, -1, :])


# =============================================================
# Dataset utilities
# =============================================================
FEATURE_COLS = ["perf_temps_sec","nageur_age_mois_scaled","perf_nage_encoded",
             "perf_distance_encoded","perf_bassin_encoded","mois_saison_sin",
             "mois_saison_cos","nageur_sexe_encoded"]
TARGET_COL = "perf_temps_sec"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================
# Training FUNCTION
# =============================================================
def train(EPOCHS=5, BATCH_SIZE=128, TRAIN_FRACTION=0.30, VAL_FRACTION=0.30, 
          use_optimized=False, model_version="v2"):
    
    SEQ_LEN = 20 if model_version.lower() == "v3" else 10  # TFT nécessite plus de contexte pour les horizons multiples
    
    print(f"\nTRAIN MODE SELECTED | VERSION: {model_version.upper()}\n")
    
    # --- Hyperparams setup ---
    if use_optimized:
        print("Loading optimized hyperparameters from best_params.json")
        try:
            with open("../models/best_params.json", "r") as f:
                best_params = json.load(f)
            BATCH_SIZE = best_params.get("batch_size", BATCH_SIZE)
            LR = best_params.get("lr", 5e-4)
            HIDDEN_DIM = best_params.get("hidden_dim", 256)
            NUM_LAYERS = best_params.get("num_layers", 4)
            DROPOUT = best_params.get("dropout", 0.4)
        except FileNotFoundError:
            print("Warning: best_params.json not found, using defaults.")
            LR, HIDDEN_DIM, NUM_LAYERS, DROPOUT = 5e-4, 256, 4, 0.4
    else:
        LR = 5e-4
        HIDDEN_DIM = 256 # Dois être divisible par n_heads pour la V2
        NUM_LAYERS = 4
        DROPOUT = 0.2
    
    print(f"Params: Batch={BATCH_SIZE}, LR={LR}, Hidden={HIDDEN_DIM}, Layers={NUM_LAYERS}, Drop={DROPOUT}")

    # --- Data Loading ---
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv("../data/performances_cleaned.csv")

    # --- Extreme values filtering ---
    print(f"Données avant filtrage : {data.shape[0]} lignes")
    q_low = data[TARGET_COL].quantile(0.001)
    q_high = data[TARGET_COL].quantile(0.999)
    data = data[(data[TARGET_COL] > q_low) & (data[TARGET_COL] < q_high)]
    print(f"Données après filtrage : {data.shape[0]} lignes")

    # --- Split Data ---
    nageurs = np.random.permutation(data["nageur_id"].unique())
    split = int(0.8 * len(nageurs))
    train_ids, test_ids = nageurs[:split], nageurs[split:]

    train_df_full = data[data["nageur_id"].isin(train_ids)]
    test_df       = data[data["nageur_id"].isin(test_ids)]
    
    os.makedirs("../data/test_data", exist_ok=True)
    test_df.to_csv("../data/test_data/test_df.csv", index=False)

    train_nageurs = np.random.permutation(train_df_full["nageur_id"].unique())
    val_split = int(0.9 * len(train_nageurs))
    train_df = train_df_full[train_df_full["nageur_id"].isin(train_nageurs[:val_split])]
    val_df   = train_df_full[train_df_full["nageur_id"].isin(train_nageurs[val_split:])]
    
    # --- Scaling ---
    scaler_y = StandardScaler()
    scaler_y.fit(train_df[[TARGET_COL]])
    train_df[TARGET_COL] = scaler_y.transform(train_df[[TARGET_COL]])
    val_df[TARGET_COL]   = scaler_y.transform(val_df[[TARGET_COL]])

    os.makedirs("../models", exist_ok=True)
    with open("../models/target_scaler.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    # Subsampling & Cleanup
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=42)
    val_df   = val_df.sample(frac=VAL_FRACTION, random_state=42)
    
    min_len = SEQ_LEN + 1
    train_df = train_df[train_df.groupby("series_id").series_id.transform('count')>=min_len]
    val_df   = val_df[val_df.groupby("series_id").series_id.transform('count')>=min_len]

    if model_version.lower() == "v3":
        train_loader = DataLoader(SwimDatasetTFT(train_df, SEQ_LEN, FEATURE_COLS), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader   = DataLoader(SwimDatasetTFT(val_df, SEQ_LEN, FEATURE_COLS), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    else:
        train_loader = DataLoader(SwimDataset(train_df, SEQ_LEN, FEATURE_COLS), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader   = DataLoader(SwimDataset(val_df, SEQ_LEN, FEATURE_COLS), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # =============================================================
    # Model Selection Logic
    # =============================================================
    if model_version.lower() == "v2":
        n_heads = 4
        if HIDDEN_DIM % n_heads != 0:
            print(f" >> Hidden Dim {HIDDEN_DIM} non divisible par {n_heads}")
            HIDDEN_DIM = (HIDDEN_DIM // n_heads) * n_heads
            print(f" >> Ajustement Hidden Dim pour MultiHead: {HIDDEN_DIM}")
            
        model = SwimAttention(
            input_dim=len(FEATURE_COLS), 
            hidden_dim=HIDDEN_DIM, 
            num_layers=NUM_LAYERS, 
            dropout=DROPOUT,
            n_heads=n_heads
        ).to(DEVICE)
        save_name = "swim_attention_v2"
    elif model_version.lower() == "v3":
        model = SwimTFT(
            input_dim=len(FEATURE_COLS), 
            hidden_dim=HIDDEN_DIM, 
            num_layers=NUM_LAYERS, 
            dropout=DROPOUT,
            n_heads=4
        ).to(DEVICE)
        save_name = "swim_tft_v3"
    else:
        model = SwimLSTM(
            input_dim=len(FEATURE_COLS), 
            hidden_dim=HIDDEN_DIM, 
            num_layers=NUM_LAYERS, 
            dropout=DROPOUT
        ).to(DEVICE)
        save_name = "swim_lstm_v1"

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Quantile Loss Setup
    quantiles = [0.1, 0.5, 0.9]
    loss_fn = None
    if model_version.lower() == "v3":
        # For TFT, we expect predictions to be of shape (batch_size, horizon, 3)
        loss_fn = MultiHorizonQuantileLoss(quantiles=quantiles)
    else:
        loss_fn = QuantilesLoss(quantiles=quantiles)

    # --- Training Loop ---
    early_stopper = EarlyStopper(patience=7, min_delta=0.0001)
    best_model_path = f"../models/{save_name}_best.pt"
    train_hist, val_hist = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [{model_version.upper()}]")
        
        for Xb, yb in pbar:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            if model_version.lower() == "v3":
                pred, feat_weights, attn_weights = model(Xb)  # TFT retourne aussi les poids (non utilisés ici)
            else:
                pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * len(Xb)
            pbar.set_postfix(loss=loss.item())

        avg_train = total_train_loss / len(train_loader.dataset)
        train_hist.append(avg_train)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                if model_version.lower() == "v3":
                    pred, feat_weights, attn_weights = model(Xb)  # TFT retourne aussi les poids (non utilisés ici)
                else:
                    pred = model(Xb)
                loss = loss_fn(pred, yb)
                total_val_loss += loss.item() * len(Xb)

        avg_val = total_val_loss / len(val_loader.dataset)
        val_hist.append(avg_val)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        stop, save = early_stopper.check(avg_val)
        if save:
            torch.save(model.state_dict(), best_model_path)
            print(f"   >>> Best Model Saved: {best_model_path}")
        if stop:
            print("Early stopping triggered.")
            break

    # Save final
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        final_path = f"../models/{save_name}.pt"
        torch.save(model.state_dict(), final_path)
        print(f"\nFinal model saved to: {final_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_hist, label="Train Loss")
    plt.plot(val_hist, label="Val Loss")
    plt.title(f"Training Curve - {model_version.upper()}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../model_result/curve_{save_name}.png")
    plt.show()