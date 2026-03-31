import pickle
from sympy import use
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from .train import SwimAttention, SwimLSTM                
from .custom_dataloader import SwimDataset, SwimDatasetTFT    

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
TARGET_COL = "perf_temps_sec"

FEATURE_COLS = [
    "perf_temps_sec","nageur_age_mois_scaled","perf_nage_encoded",
    "perf_distance_encoded","perf_bassin_encoded","mois_saison_sin",
    "mois_saison_cos","nageur_sexe_encoded", "days_since_last_log"
]

PAST_COLS = ["perf_temps_sec", "days_since_last_log", "perf_nage_encoded"]
# La présence de "age" dans les features futures est discutable, mais on peut laisser pour voir si le modèle l'utilise
FUTURE_COLS = [
    "mois_saison_sin", "mois_saison_cos", 
    "perf_distance_encoded", "perf_bassin_encoded", 
    "nageur_age_mois_scaled"
]
STATIC_COLS = ["nageur_sexe_encoded"]

# =============================================================
#  TEST FUNCTION
# =============================================================

def test(save_figures=True, show_figures=True, model_version="v4", use_optimized=False):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # --- Configuration par version ---
    if model_version.lower() in ["v3", "v4"]:
        SEQ_LEN = 20
        HORIZON = 3
        print(f"\nTEST MODE SELECTED | VERSION: {model_version.upper()} (TFT Multi-Horizon)")
    else:
        SEQ_LEN = 10
        HORIZON = 1
        print(f"\nTEST MODE SELECTED | VERSION: {model_version.upper()}")

    # Dossier de résultats
    result_dir = "../model_result"
    os.makedirs(result_dir, exist_ok=True)

    # ------------------ Load dataset ------------------ #
    test_csv_path = "../data/test_data/test_df.csv"
    if model_version.lower() == "v4":
        test_csv_path = "../data/test_data/test_df_v4.csv"
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Saved test data not found: {test_csv_path}")

    test_df = pd.read_csv(test_csv_path)
    print(f"Loaded test data: {len(test_df)} rows")

    # Scale target
    with open("../models/target_scaler.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    test_df[TARGET_COL] = scaler_y.transform(test_df[[TARGET_COL]])

    # Filtrage séries longues
    min_len = SEQ_LEN + HORIZON
    valid_series = test_df.groupby("series_id").size()
    valid_series = valid_series[valid_series >= min_len].index
    test_df = test_df[test_df["series_id"].isin(valid_series)]

    # Choix du Dataset
    if model_version.lower() == "v4":
        from .custom_dataloader import SwimDatasetTFTV2
        test_dataset = SwimDatasetTFTV2(test_df, past_cols=PAST_COLS, future_cols=FUTURE_COLS, static_cols=STATIC_COLS, target_col=TARGET_COL, seq_len=SEQ_LEN, horizon=HORIZON)
    elif model_version.lower() == "v3":
        test_dataset = SwimDatasetTFT(test_df, seq_len=SEQ_LEN, feature_cols=FEATURE_COLS, horizon=HORIZON)
    else:
        test_dataset = SwimDataset(test_df, seq_len=SEQ_LEN, feature_cols=FEATURE_COLS)
        
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # ------------------ Load Model ------------------ #
    if model_version.lower() == "v4":
        from .SwimTFTv2 import TemporalFusionTransformer
        model_path = "../models/swim_tft_v4_best.pt"
        params = {
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
                print("Best parameters file not found. Using default parameters.")
        
        model = TemporalFusionTransformer(
            number_of_past_inputs=SEQ_LEN,
            horizon=params["horizon"],
            embedding_size_inputs=1,
            hidden_dimension=params["hidden_dim"],
            dropout=params["dropout"],
            number_of_heads=params["n_heads"],
            past_inputs={c: 1 for c in PAST_COLS},
            future_inputs={c: 1 for c in FUTURE_COLS},
            static_inputs={c: 1 for c in STATIC_COLS},
            batch_size=params["batch_size"],
            device=DEVICE,
            quantiles=[0.1, 0.5, 0.9]
        ).to(DEVICE)
    elif model_version.lower() == "v3":
        from .SwimTFT import SwimTFT
        model_path = "../models/swim_tft_v3.pt"
        model = SwimTFT(input_dim=len(FEATURE_COLS), hidden_dim=256, num_layers=4, dropout=0.2, n_heads=4).to(DEVICE)
    elif model_version.lower() == "v1":
        model_path = "../models/swim_lstm.pt"
        model = SwimLSTM(input_dim=len(FEATURE_COLS), hidden_dim=128, num_layers=4, dropout=0.49726187436364466).to(DEVICE)
    elif model_version.lower() == "v2":
        model_path = "../models/swim_attention_v2.pt"
        model = SwimAttention(input_dim=len(FEATURE_COLS), hidden_dim=128, num_layers=4, dropout=0.2, n_heads=4).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"\nModel {model_version} loaded and ready.\n")

    preds_list, trues_list = [], []
    importances_list = []

    # ------------------ Inference ------------------ #
    with torch.no_grad():
        for batch_data, y_batch in tqdm(test_loader, desc=f"Testing {model_version}"):
            
            if model_version.lower() == "v4":
                past_dicts = {k: v.to(DEVICE) for k, v in batch_data['past'].items()}
                future_dicts = {k: v.to(DEVICE) for k, v in batch_data['future'].items()}
                static_dicts = {k: v.to(DEVICE) for k, v in batch_data['static'].items()}
                
                # Output V4: [Batch, Horizon, 1, Quantiles]
                pred_full, feat_weights = model(past_inputs=past_dicts, future_inputs=future_dicts, static_inputs=static_dicts)
                
                # On squeeze la dimension "target_dimension" (1) -> [Batch, Horizon, Quantiles]
                pred_full = pred_full.squeeze(2)
                
                # On prend l'horizon t+1 pour les métriques globales et les graphes
                pred = pred_full[:, 0, :].cpu().numpy() # [Batch, 3 quantiles]
                y_target = y_batch[:, 0].numpy()        # [Batch]
                
                if isinstance(feat_weights, dict):
                    importances_list.append(feat_weights['past'].mean(dim=(0,1)).cpu().numpy())

            elif model_version.lower() == "v3":
                X_batch = batch_data.to(DEVICE)
                pred_full, feat_weights, _ = model(X_batch)
                pred = pred_full[:, 0, :].cpu().numpy()
                y_target = y_batch[:, 0].numpy()
                importances_list.append(feat_weights.mean(dim=(0,1)).cpu().numpy())
            else:
                X_batch = batch_data.to(DEVICE)
                pred = model(X_batch).cpu().numpy()
                y_target = y_batch.numpy()

            preds_list.append(pred)
            trues_list.append(y_target)

    # ------------------ Post-Processing & Metrics ------------------ #
    preds = np.vstack(preds_list) # [N, 3]
    trues = np.concatenate(trues_list)

    # Inverse scale (sur le premier horizon pour la comparaison)
    q10 = scaler_y.inverse_transform(preds[:, 0:1]).squeeze()
    q50 = scaler_y.inverse_transform(preds[:, 1:2]).squeeze()
    q90 = scaler_y.inverse_transform(preds[:, 2:3]).squeeze()
    true = scaler_y.inverse_transform(trues.reshape(-1, 1)).squeeze()

    # Correction croisement quantiles
    q10 = np.minimum(q10, q50)
    q90 = np.maximum(q90, q50)
    
    # ------------------ Metrics ------------------ #
    def pinball(y, q, alpha): # équivaut à la quantile loss
        return np.mean(np.maximum(alpha*(y-q), (alpha-1)*(y-q)))

    mae = np.mean(np.abs(q50 - true))
    rmse = np.sqrt(np.mean((q50 - true)**2))

    pb10 = pinball(true, q10, 0.10)
    pb50 = pinball(true, q50, 0.50)
    pb90 = pinball(true, q90, 0.90)

    coverage = np.mean((true >= q10) & (true <= q90)) * 100
    
    # Winkler Score (pénalise si l'intervalle est trop large ou si ça sort)
    width = q90 - q10
    # Si true < q10 (trop bas)
    under = (q10 - true) * (true < q10)
    # Si true > q90 (trop haut)
    over = (true - q90) * (true > q90)
    alpha = 0.2 # (1 - 0.8) pour notre intervalle 10-90
    winkler = np.mean(width + (2/alpha)*under + (2/alpha)*over)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((true - q50) / true)) * 100

    print("\n===== Test Metrics =====")
    print(f"MAE:          {mae:.4f}")
    print(f"RMSE:         {rmse:.4f}")
    print(f"MAPE:         {mape:.2f} %")
    print(f"Pinball q10:  {pb10:.4f}")
    print(f"Pinball q50:  {pb50:.4f}")
    print(f"Pinball q90:  {pb90:.4f}")
    print(f"Coverage(10-90): {coverage:.2f}% (ideal ≈ 80%)")
    print(f"Winkler Score:   {winkler:.4f}")
    print("========================\n")


    # ------------------ Distribution Summary ------------------ #
    print("\n===== Distribution Summary =====")
    stats = pd.DataFrame({
        "Mean":      [np.mean(true), np.mean(q50)],
        "Median":    [np.median(true), np.median(q50)],
        "Std":       [np.std(true), np.std(q50)],
        "Min":       [np.min(true), np.min(q50)],
        "Max":       [np.max(true), np.max(q50)],
        "P10":       [np.percentile(true,10), np.percentile(q50,10)],
        "P90":       [np.percentile(true,90), np.percentile(q50,90)]
    }, index=["True", "Pred q50"])

    print(stats.round(4), "\n")

    # ------------------ V3 Feature Importance ------------------ #
    if model_version.lower() == "v3" and importances_list:
        avg_imp = np.mean(np.vstack(importances_list), axis=0)
        # Création d'un DataFrame pour l'affichage
        imp_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": avg_imp
        }).sort_values(by="Importance", ascending=False)
        
        print("\n===== Global Feature Importance (TFT) =====")
        print(imp_df)
        print("===========================================\n")
        
        # Plot Feature Importance
        fig = plt.figure(figsize=(10, 6))
        plt.barh(imp_df["Feature"], imp_df["Importance"], color='skyblue')
        plt.xlabel("Importance moyenne")
        plt.title("TFT - Quelles variables comptent le plus ?")
        plt.gca().invert_yaxis() # Meilleure feature en haut
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        path = f"{result_dir}/{timestamp}_feature_importance.png"
        if save_figures: fig.savefig(path, bbox_inches="tight")
        if show_figures: plt.show()
        plt.close(fig)
    if model_version.lower() == "v4" and importances_list:
        avg_imp = np.mean(np.vstack(importances_list), axis=0)
        importances_brutes = {col: float(avg_imp[i]) for i, col in enumerate(PAST_COLS)}
        total = sum(importances_brutes.values()) + 1e-8
        importances_finales = {k: v / total for k, v in importances_brutes.items()}

        imp_df = pd.DataFrame({
            "Feature": list(importances_finales.keys()),
            "Importance": list(importances_finales.values())
        }).sort_values(by="Importance", ascending=False)
        
        print("\n===== Global Feature Importance (TFT V4) =====")
        print(imp_df)
        print("==============================================\n")
        
        fig = plt.figure(figsize=(10, 6))
        plt.barh(imp_df["Feature"], imp_df["Importance"], color='salmon')
        plt.xlabel("Importance moyenne normalisée")
        plt.title("TFT V4 - Quelles variables passées comptent le plus ?")
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        path = f"{result_dir}/{timestamp}_feature_importance_v4.png"
        if save_figures: fig.savefig(path, bbox_inches="tight")
        if show_figures: plt.show()
        plt.close(fig)
    # ------------------ PLOTTING & SAVING ------------------ #

    def save_or_show(fig, name):
        path = f"{result_dir}/{timestamp}_{name}.png"
        if save_figures:
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved: {path}")
        if show_figures:
            plt.show()
        plt.close(fig)

    # 1 – Prediction plot with quantile band
    fig = plt.figure(figsize=(14,6))
    plt.plot(true, label="True Values", alpha=0.9)
    plt.plot(q50, label="Predicted q50 (Median)", linewidth=1.3)
    plt.fill_between(np.arange(len(q50)), q10, q90, alpha=0.25,
                     label="Uncertainty band (q10–q90)")
    plt.legend(fontsize=10)
    plt.title("Prediction vs True – Quantile Regression LSTM")
    plt.xlabel("Samples"); plt.ylabel("Time (seconds)")
    plt.grid(True)
    save_or_show(fig, "prediction_with_quantiles")

    # 2 – True-only visualization
    fig = plt.figure(figsize=(14,5))
    plt.plot(true, linewidth=1)
    plt.title("True Performance Times")
    plt.xlabel("Samples"); plt.ylabel("Time (seconds)")
    plt.grid(True)
    save_or_show(fig, "true_only_plot")

    # 3 – Distribution Histogram
    fig = plt.figure(figsize=(12,5))
    plt.hist(true, bins=70, alpha=0.5, density=True, label="True")
    plt.hist(q50, bins=70, alpha=0.5, density=True, label="Pred q50")
    plt.title("Distribution Comparison – True vs Predicted q50")
    plt.legend(); plt.grid(True)
    save_or_show(fig, "hist_distribution")

    # 4 – KDE Curve
    from scipy.stats import gaussian_kde
    xs = np.linspace(min(true.min(), q50.min()), max(true.max(), q50.max()), 400)
    kde_true = gaussian_kde(true); kde_pred = gaussian_kde(q50)

    fig = plt.figure(figsize=(12,5))
    plt.plot(xs, kde_true(xs), label="True KDE", linewidth=2)
    plt.plot(xs, kde_pred(xs), label="Pred KDE", linewidth=2)
    plt.title("KDE Density Curve Comparison")
    plt.legend(); plt.grid(True)
    save_or_show(fig, "kde_density")

    # 5 – Boxplot
    fig = plt.figure(figsize=(7,4))
    plt.boxplot([true, q50], labels=["True", "Pred q50"])
    plt.title("Boxplot Comparison – Predictions vs Ground Truth")
    plt.grid(True, axis='y')
    save_or_show(fig, "boxplot_comparison")

    print("\n Testing complete. Results saved to /model_result/\n")
