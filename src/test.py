import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .train import SwimLSTM                
from .train import SwimIterableDataset    

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 10
BATCH_SIZE = 128
TARGET_COL = "perf_temps_sec"

FEATURE_COLS = [
    "perf_temps_sec","nageur_age_mois_scaled","perf_nage_encoded",
    "perf_distance_encoded","perf_bassin_encoded","mois_saison_sin",
    "mois_saison_cos","nageur_sexe_encoded"
]


# =============================================================
#  TEST FUNCTION
# =============================================================
def test():
    print("\nTEST MODE SELECTED\n")

    # ------------------ Load dataset ------------------ #
    test_csv_path = "../data/test_data/test_df.csv"
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Saved test data not found: {test_csv_path}")

    test_df = pd.read_csv(test_csv_path)
    print(f"Loaded test data: {len(test_df)} rows")

    # Scale target
    with open("../models/target_scaler.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    test_df[TARGET_COL] = scaler_y.transform(test_df[[TARGET_COL]])

    # Keep only long series
    min_len = SEQ_LEN + 1
    valid_series = test_df.groupby("series_id").size()
    valid_series = valid_series[valid_series >= min_len].index
    test_df = test_df[test_df["series_id"].isin(valid_series)]

    test_loader = DataLoader(
        SwimIterableDataset(test_df, SEQ_LEN),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ------------------ Load Model ------------------ #
    model = SwimLSTM(input_dim=len(FEATURE_COLS)).to(DEVICE)
    model.load_state_dict(torch.load("../models/swim_lstm.pt", map_location=DEVICE))
    model.eval()

    print("\nModel loaded and ready for inference.\n")

    preds_list, trues_list = [], []

    # ------------------ Inference ------------------ #
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Testing"):
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch).cpu().numpy()

            preds_list.append(pred)
            trues_list.append(y_batch.numpy())

    preds = np.vstack(preds_list)
    trues = np.concatenate(trues_list)

    # Inverse scale
    q10 = scaler_y.inverse_transform(preds[:, 0:1]).squeeze()
    q50 = scaler_y.inverse_transform(preds[:, 1:2]).squeeze()
    q90 = scaler_y.inverse_transform(preds[:, 2:3]).squeeze()
    true = scaler_y.inverse_transform(trues.reshape(-1, 1)).squeeze()

    # Safety for quantile crossing
    q10 = np.minimum(q10, q50)
    q90 = np.maximum(q90, q50)

    # ------------------ Plot ------------------ #
    plt.figure(figsize=(14, 6))
    plt.plot(true, label="True")
    plt.plot(q50, label="q50 median")
    plt.fill_between(np.arange(len(q50)), q10, q90, alpha=0.2, label="q10-q90 band")
    plt.title("Test - Quantile LSTM Predictions")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n Testing complete.\n")

