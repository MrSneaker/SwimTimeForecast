import json
import optuna
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# Local imports
try:
    from .quantile_loss import QuantileLoss
    from .train import SwimLSTM
    from .custom_dataloader import SwimDataset
except ImportError:
    from quantile_loss import QuantileLoss
    from train import SwimLSTM
    from custom_dataloader import SwimDataset

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =============================================================
# CONFIGURATION
# =============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_COLS = ["perf_temps_sec", "nageur_age_mois_scaled", "perf_nage_encoded",
                "perf_distance_encoded", "perf_bassin_encoded", "mois_saison_sin",
                "mois_saison_cos", "nageur_sexe_encoded"]
TARGET_COL = "perf_temps_sec"

SAMPLE_FRACTION = 0.05
MIN_SWIMMERS = 500
NB_EPOCHS_OPT = 6

# =============================================================
# DATA PREPARATION
# =============================================================
print(f">>> Loading Data... (Target Fraction: {SAMPLE_FRACTION})")
full_df = pd.read_csv("../data/performances_cleaned.csv")

unique_ids = full_df["nageur_id"].unique()
target_n = max(int(len(unique_ids) * SAMPLE_FRACTION), min(MIN_SWIMMERS, len(unique_ids)))

# Stratified Subsampling
splitter = StratifiedShuffleSplit(n_splits=1, train_size=target_n, random_state=42)
profiles = full_df.groupby("nageur_id")["perf_distance_encoded"].first().reset_index()

try:
    for _, idx in splitter.split(profiles, profiles["perf_distance_encoded"]):
        selected_ids = profiles.iloc[idx]["nageur_id"].values
except ValueError:
    selected_ids = np.random.choice(unique_ids, size=target_n, replace=False)

subset_df = full_df[full_df["nageur_id"].isin(selected_ids)].copy()

# Split
nageurs = subset_df["nageur_id"].unique()
np.random.shuffle(nageurs)
split_idx = int(0.8 * len(nageurs))
train_df_raw = subset_df[subset_df["nageur_id"].isin(nageurs[:split_idx])].copy()
val_df_raw = subset_df[subset_df["nageur_id"].isin(nageurs[split_idx:])].copy()

# Scale
scaler_y = StandardScaler()
scaler_y.fit(train_df_raw[[TARGET_COL]])
train_df_raw[TARGET_COL] = scaler_y.transform(train_df_raw[[TARGET_COL]])
val_df_raw[TARGET_COL]   = scaler_y.transform(val_df_raw[[TARGET_COL]])

# Pre-load Datasets
SEQ_LEN = 10
train_ds = SwimDataset(train_df_raw, SEQ_LEN, FEATURE_COLS)
val_ds   = SwimDataset(val_df_raw, SEQ_LEN, FEATURE_COLS)
print(f">>> Optuna Dataset Ready: {len(train_ds)} train seqs | {len(val_ds)} val seqs")


# =============================================================
# OPTUNA OBJECTIVE
# =============================================================
def objective(trial):
    # Hyperparams
    param = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "dropout": trial.suggest_float("dropout", 0.2, 0.6), # Encouraging higher dropout for uncertainty
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_int("batch_size", 64, 512, step=32),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    }

    train_loader = DataLoader(train_ds, batch_size=param["batch_size"], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=param["batch_size"], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    model = SwimLSTM(len(FEATURE_COLS), param["hidden_dim"], param["num_layers"], param["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=param["lr"], weight_decay=param["weight_decay"])
    
    # Quantile Loss
    loss_fns = [QuantileLoss(q) for q in [0.1, 0.5, 0.9]]

    for epoch in range(NB_EPOCHS_OPT):
        print(f"Trial {trial.number} | Epoch {epoch+1}/{NB_EPOCHS_OPT}")
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = sum(loss_fns[i](preds[:, i], yb) for i in range(3))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validation for Pruning (Approximate on first batch to save time if needed, or full)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_loss += sum(loss_fns[i](model(Xb)[:, i], yb) for i in range(3)).item()
        
        avg_val = val_loss / len(val_loader)
        trial.report(avg_val, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # --- METRICS & COVERAGE PENALTY ---
    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            preds_list.append(model(Xb.to(DEVICE)).cpu().numpy())
            trues_list.append(yb.numpy())
            
    preds = np.vstack(preds_list)
    trues = np.concatenate(trues_list)

    # Inverse Transform
    q10 = scaler_y.inverse_transform(preds[:, 0:1]).squeeze()
    q50 = scaler_y.inverse_transform(preds[:, 1:2]).squeeze()
    q90 = scaler_y.inverse_transform(preds[:, 2:3]).squeeze()
    true_vals = scaler_y.inverse_transform(trues.reshape(-1, 1)).squeeze()

    # Fix crossing
    q10 = np.minimum(q10, q50)
    q90 = np.maximum(q90, q50)

    # 1. Winkler Score (Quality of Intervals)
    alpha = 0.2
    width = q90 - q10
    below = np.maximum(q10 - true_vals, 0)
    above = np.maximum(true_vals - q90, 0)
    winkler = np.mean(width + (2/alpha) * (below + above))

    # 2. Coverage Penalty (HARD CONSTRAINT)
    in_bound = (true_vals >= q10) & (true_vals <= q90)
    coverage = np.mean(in_bound)
    coverage_gap = abs(coverage - 0.80) 
    
    # We penalize massively if coverage is far from 80%
    # If gap is 0.18 (62%), penalty adds ~3.6 to score
    penalty = coverage_gap * 20.0 

    final_score = (winkler / 10.0) + penalty
    
    return final_score


if __name__ == "__main__":
    pruner = optuna.pruners.HyperbandPruner(min_resource=2, max_resource=NB_EPOCHS_OPT, reduction_factor=3)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    
    print("Starting Optimization...")
    study.optimize(objective, n_trials=30, show_progress_bar=True, n_jobs=1)

    print("\n" + "="*40)
    print("BEST PARAMS")
    print("="*40)
    print(study.best_params)
    print(f"Best Score: {study.best_value:.4f}")

    with open("../models/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    print("Best parameters saved to ../models/best_params.json")
    
    try:
        os.makedirs("../model_result", exist_ok=True)
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("../model_result/optuna_history.png")
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image("../model_result/optuna_importance.png")
        print("Visualization saved.")
    except Exception as e:
        print(f"Could not save plots: {e}")