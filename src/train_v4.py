import os
import pickle
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Imports locaux
from .custom_dataloader import SwimDatasetTFTV2
from .quantile_loss import MultiHorizonQuantileLossV2
from .early_stopper import EarlyStopper
from .SwimTFTv2 import TemporalFusionTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================
# DEFINITION DES FEATURES (Logique TFT)
# =============================================================
PAST_COLS = ["perf_temps_sec", "days_since_last_log", "perf_nage_encoded"]
# La présence de "age" dans les features futures est discutable, mais on peut laisser pour voir si le modèle l'utilise
FUTURE_COLS = [
    "mois_saison_sin", "mois_saison_cos", 
    "perf_distance_encoded", "perf_bassin_encoded", 
    "nageur_age_mois_scaled"
]
STATIC_COLS = ["nageur_sexe_encoded"]

TARGET_COL = "perf_temps_sec"
QUANTILES = [0.1, 0.5, 0.9]

# =============================================================
# 1. DATA LOADING
# =============================================================
def load_and_prep_data(seq_len=20, horizon=3, sample_frac=1.0):
    """
    Charge les données, effectue un split par nageur (80/10/10),
    sauvegarde le set de test brut et scale le train/val.
    """
    print(f"\n>>> Loading Data (Seq: {seq_len}, Horizon: {horizon})...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../data/performances_cleaned.csv")
    df = pd.read_csv(data_path)
    
    if sample_frac < 1.0:
        ids = df["nageur_id"].unique()
        selected_ids = np.random.choice(ids, int(len(ids) * sample_frac), replace=False)
        df = df[df["nageur_id"].isin(selected_ids)]

    # Split par nageurs pour éviter de mélanger le passé/futur d'un même athlète
    nageurs = df["nageur_id"].unique()
    np.random.shuffle(nageurs)
    
    # 80% Train, 10% Val, 10% Test
    train_split = int(0.8 * len(nageurs))
    val_split = int(0.9 * len(nageurs))
    
    train_ids = nageurs[:train_split]
    val_ids = nageurs[train_split:val_split]
    test_ids = nageurs[val_split:]
    
    train_df = df[df["nageur_id"].isin(train_ids)].copy()
    val_df = df[df["nageur_id"].isin(val_ids)].copy()
    test_df = df[df["nageur_id"].isin(test_ids)].copy()

    # --- Sauvegarde du set de TEST (Brut) ---
    test_dir = os.path.join(base_dir, "../data/test_data")
    os.makedirs(test_dir, exist_ok=True)
    test_path = os.path.join(test_dir, "test_df_v4.csv")
    test_df.to_csv(test_path, index=False)
    print(f"Test set saved ({len(test_df)} rows) to: {test_path}")

    # --- Scaling de la Target (Train & Val seulement) ---
    scaler_y = StandardScaler()
    scaler_y.fit(train_df[[TARGET_COL]])
    
    train_df[TARGET_COL] = scaler_y.transform(train_df[[TARGET_COL]])
    val_df[TARGET_COL] = scaler_y.transform(val_df[[TARGET_COL]])

    # Sauvegarde du scaler pour l'utiliser dans test.py
    model_dir = os.path.join(base_dir, "../models")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "target_scaler.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)

    return train_df, val_df

# =============================================================
# 2. TRAINING ROUTINE
# =============================================================
def run_training(params, train_df, val_df, trial=None, save_model=False, plot_loss=False):
    BATCH_SIZE = params.get("batch_size", 64)
    LR = params.get("lr", 1e-3)
    HIDDEN_DIM = params.get("hidden_dim", 64)
    DROPOUT = params.get("dropout", 0.2)
    N_HEADS = params.get("n_heads", 4)
    EPOCHS = params.get("epochs", 10)
    SEQ_LEN = params.get("seq_len", 20)
    HORIZON = params.get("horizon", 3)
    
    best_model_path = os.path.join(os.path.dirname(__file__), "../models/swim_tft_v4_best.pt")
    
    train_ds = SwimDatasetTFTV2(train_df, past_cols=PAST_COLS, future_cols=FUTURE_COLS, static_cols=STATIC_COLS, target_col=TARGET_COL, seq_len=SEQ_LEN, horizon=HORIZON)
    val_ds = SwimDatasetTFTV2(val_df, past_cols=PAST_COLS, future_cols=FUTURE_COLS, static_cols=STATIC_COLS, target_col=TARGET_COL, seq_len=SEQ_LEN, horizon=HORIZON)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Configuration des dimensions (1 par feature car on utilise des GRNs séparés)
    past_config = {col: 1 for col in PAST_COLS}
    future_config = {col: 1 for col in FUTURE_COLS}
    static_config = {col: 1 for col in STATIC_COLS}

    model = TemporalFusionTransformer(
        number_of_past_inputs=SEQ_LEN,
        horizon=HORIZON,
        embedding_size_inputs=1, 
        hidden_dimension=HIDDEN_DIM,
        dropout=DROPOUT,
        number_of_heads=N_HEADS,
        past_inputs=past_config,
        future_inputs=future_config,
        static_inputs=static_config,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        quantiles=QUANTILES
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = MultiHorizonQuantileLossV2(quantiles=QUANTILES)
    early_stopper = EarlyStopper(patience=5, min_delta=0.0005)
    
    val_losses, train_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_inputs, yb in pbar:
            # Préparation des dictionnaires pour le modèle
            past_dicts = {k: v.to(DEVICE) for k, v in batch_inputs['past'].items()}
            future_dicts = {k: v.to(DEVICE) for k, v in batch_inputs['future'].items()}
            static_dicts = {k: v.to(DEVICE) for k, v in batch_inputs['static'].items()}
            yb = yb.to(DEVICE)
            
            optimizer.zero_grad()
            preds, _ = model(past_inputs=past_dicts, future_inputs=future_dicts, static_inputs=static_dicts)
            
            # Loss sur les quantiles
            loss = criterion(preds, yb)
            loss.backward()
            
            # Gradient clipping très bas pour stabiliser le TFT
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_inputs, yb in val_loader:
                p_d = {k: v.to(DEVICE) for k, v in batch_inputs['past'].items()}
                f_d = {k: v.to(DEVICE) for k, v in batch_inputs['future'].items()}
                s_d = {k: v.to(DEVICE) for k, v in batch_inputs['static'].items()}
                yb = yb.to(DEVICE)
                
                preds, _ = model(past_inputs=p_d, future_inputs=f_d, static_inputs=s_d)
                val_loss += criterion(preds, yb).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        if trial:
            import optuna
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        stop, save = early_stopper.check(avg_val_loss)
        if save:
            torch.save(model.state_dict(), best_model_path)
            print(f"   >>> Best Model Saved: {best_model_path}")
        if stop:
            print("Early stopping triggered.")
            break
            
    if save_model:
        model_path = os.path.join(os.path.dirname(__file__), "../models/swim_tft_v4.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    if plot_loss:
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")
        plt.legend(); plt.show()
        plt.savefig(os.path.join(os.path.dirname(__file__), "../model_result/loss_curve_v4.png"))
        
    return avg_val_loss

# =============================================================
# 3. WRAPPER (Lancement manuel ou via main.py)
# =============================================================
def train(EPOCHS=10, BATCH_SIZE=64, use_optimized=False):
    t_df, v_df = load_and_prep_data(seq_len=20, horizon=3)
    
    params = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_dim": 256,
        "lr": 5e-4,
        "dropout": 0.3,
        "n_heads": 8,
        "seq_len": 20,
        "horizon": 3
    }

    if use_optimized:
        try:
            with open("../models/best_params_v4.json", "r") as f:
                import json
                params.update(json.load(f))
        except FileNotFoundError:
            print("Optimized params not found, using defaults.")
    
    run_training(params, t_df, v_df, save_model=True, plot_loss=True)

if __name__ == "__main__":
    train()