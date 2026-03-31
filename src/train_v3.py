import os
import pickle
import optuna
from streamlit import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Imports locaux
from .SwimTFT import SwimTFT
from .custom_dataloader import SwimDataset, SwimDatasetTFT
from .quantile_loss import MultiHorizonQuantileLoss
from .early_stopper import EarlyStopper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_COLS = ["perf_temps_sec","nageur_age_mois_scaled","perf_nage_encoded",
             "perf_distance_encoded","perf_bassin_encoded","mois_saison_sin",
             "mois_saison_cos","nageur_sexe_encoded", "days_since_last_log"]
TARGET_COL = "perf_temps_sec"

# =============================================================
# 1. DATA LOADING (Fonction réutilisable)
# =============================================================
def load_and_prep_data(seq_len=20, horizon=3, sample_frac=1.0):
    """Charge, nettoie, scale et prépare les DataLoaders une seule fois."""
    print(f"\n>>> Loading Data (Seq: {seq_len}, Horizon: {horizon})...")
    
    # Chemin relatif robuste
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../data/performances_cleaned.csv")
    
    df = pd.read_csv(data_path)
    
    # Filtrage et Subsampling
    if sample_frac < 1.0:
        ids = df["nageur_id"].unique()
        selected_ids = np.random.choice(ids, int(len(ids) * sample_frac), replace=False)
        df = df[df["nageur_id"].isin(selected_ids)]

    # Split Train/Val/Test par ID nageur
    nageurs = df["nageur_id"].unique()
    np.random.shuffle(nageurs)
    
    # 80% Train, 20% Val (On ignore le Test pour l'optimisation)
    split_idx = int(0.8 * len(nageurs))
    train_ids = nageurs[:split_idx]
    val_ids = nageurs[split_idx:]
    
    train_df = df[df["nageur_id"].isin(train_ids)].copy()
    val_df = df[df["nageur_id"].isin(val_ids)].copy()

    # Scaling Target
    scaler_y = StandardScaler()
    scaler_y.fit(train_df[["perf_temps_sec"]])
    train_df["perf_temps_sec"] = scaler_y.transform(train_df[["perf_temps_sec"]])
    val_df["perf_temps_sec"] = scaler_y.transform(val_df[["perf_temps_sec"]])

    # Sauvegarde du scaler
    os.makedirs(os.path.join(base_dir, "../models"), exist_ok=True)
    with open(os.path.join(base_dir, "../models/target_scaler.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)

    # Filtrage longueur minimale
    min_len = seq_len + horizon
    train_df = train_df[train_df.groupby("series_id")["series_id"].transform('count') >= min_len]
    val_df = val_df[val_df.groupby("series_id")["series_id"].transform('count') >= min_len]

    print(f"Data Ready: Train={len(train_df)} rows, Val={len(val_df)} rows")
    return train_df, val_df, len(train_df.columns)

# =============================================================
# 2. TRAINING ROUTINE
# =============================================================
def run_training(params, train_df, val_df, trial=None, save_model=False, plot_loss=False):
    """
    params: dict des hyperparamètres (lr, hidden_dim, etc.)
    trial: objet optuna.trial (optionnel, pour le pruning)
    """
    # Hyperparams extraction
    BATCH_SIZE = params.get("batch_size", 64)
    LR = params.get("lr", 1e-3)
    HIDDEN_DIM = params.get("hidden_dim", 64)
    NUM_LAYERS = params.get("num_layers", 2)
    DROPOUT = params.get("dropout", 0.2)
    N_HEADS = params.get("n_heads", 4)
    EPOCHS = params.get("epochs", 10)
    
    # Création Datasets (Rapide car données déjà en RAM)
    train_ds = SwimDatasetTFT(train_df, seq_len=20, feature_cols=FEATURE_COLS, horizon=3)
    val_ds = SwimDatasetTFT(val_df, seq_len=20, feature_cols=FEATURE_COLS, horizon=3) # Fix feature cols logic if needed

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Modèle TFT V3
    model = SwimTFT(
        input_dim=8, # Hardcodé temporairement, à ajuster selon tes FEATURE_COLS réelles
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        n_heads=N_HEADS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = MultiHorizonQuantileLoss()
    early_stopper = EarlyStopper(patience=5, min_delta=0.001)
    val_losses = []
    train_losses = []

    # Boucle d'entraînement
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # Train Step
        for Xb, yb in pbar:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            
            preds, _, _ = model(Xb)
            loss = criterion(preds, yb)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Clipping important pour TFT
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))
        print(f"Epoch {epoch+1} | Train Loss : {train_loss / len(train_loader):.5f}")

        # Val Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                preds, _, _ = model(Xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.5f}")
        
        # --- OPTUNA PRUNING ---
        if trial:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Early Stopping check
        stop, _ = early_stopper.check(avg_val_loss)
        if stop:
            break
    if save_model:
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/swim_tft_v3.pt"))
    if plot_loss:
        plt.figure(figsize=(10, 5))
        plt.plot(val_losses, label="Validation Loss")
        plt.plot(train_losses, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation and Training Loss Over Epochs")
        plt.legend()
        plt.grid()
        plt.show()
        
    return avg_val_loss # C'est ce que Optuna va minimiser

# =============================================================
# 3. LEGACY WRAPPER
# =============================================================
def train(EPOCHS=10, BATCH_SIZE=64, use_optimized=False):
    """Fonction wrapper pour lancer un entraînement manuel."""
    # Charge tout le dataset
    t_df, v_df, _ = load_and_prep_data(seq_len=20)
    
    # Si on veut les hyperparams optimisés, on les charge depuis le fichier JSON
    if use_optimized:
        import json
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, "../models/best_params.json"), "r") as f:
            best_params = json.load(f)
        print(f">>> Using Optimized Hyperparameters: {best_params}")
        EPOCHS = best_params.get("epochs", EPOCHS)
        BATCH_SIZE = best_params.get("batch_size", BATCH_SIZE)
    
    params = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_dim": 256,
        "lr": 5e-4,
        "num_layers": 4,
        "dropout": 0.3,
        "n_heads": 8
    }
    
    final_loss = run_training(params, t_df, v_df, save_model=True, plot_loss=True)
    print(f"Training finished. Final Val Loss: {final_loss:.5f}")

if __name__ == "__main__":
    train()