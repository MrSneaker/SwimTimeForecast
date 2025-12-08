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
def test(save_figures=True, show_figures=True):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # Create result directory
    result_dir = "../model_result"
    os.makedirs(result_dir, exist_ok=True)

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

    # ------------------ Metrics ------------------ #
    def pinball(y, q, alpha):
        return np.mean(np.maximum(alpha*(y-q), (alpha-1)*(y-q)))

    mae = np.mean(np.abs(q50 - true))
    rmse = np.sqrt(np.mean((q50 - true)**2))

    pb10 = pinball(true, q10, 0.10)
    pb50 = pinball(true, q50, 0.50)
    pb90 = pinball(true, q90, 0.90)

    coverage = np.mean((true >= q10) & (true <= q90)) * 100
    winkler = np.mean((q90-q10) + (2/0.8)*np.maximum(q10-true,0) + (2/0.8)*np.maximum(true-q90,0))

    print("\n===== Test Metrics =====")
    print(f"MAE:          {mae:.4f}")
    print(f"RMSE:         {rmse:.4f}")
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

    print("\n Testing complete. Results saved to /model_result/ 📁\n")
