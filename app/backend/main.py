import math
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from typing import List

from src.SwimTFT import SwimTFT


# === Model Definition ===
class SwimLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=4, dropout=0.2):
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
        return self.fc(out[:, -1, :])


# === Config ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 20
FEATURE_COLS = ["perf_temps_sec","nageur_age_mois_scaled","perf_nage_encoded",
             "perf_distance_encoded","perf_bassin_encoded","mois_saison_sin",
             "mois_saison_cos","nageur_sexe_encoded"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "swim_lstm.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "target_scaler.pkl")

# Encoders/scalers
LE_NAGE_PATH = os.path.join(MODEL_DIR, "encoder_perf_nage.pkl")
LE_SEXE_PATH = os.path.join(MODEL_DIR, "encoder_sexe.pkl")
LE_DISTANCE_PATH = os.path.join(MODEL_DIR, "encoder_perf_distance.pkl")
LE_BASSIN_PATH = os.path.join(MODEL_DIR, "encoder_perf_bassin.pkl")
SCALER_AGE_PATH = os.path.join(MODEL_DIR, "scaler_age.pkl")

with open(SCALER_PATH, "rb") as f:
    scaler_y = pickle.load(f)

# Load each encoder separately
with open(LE_NAGE_PATH, "rb") as f: le_nage = pickle.load(f)
with open(LE_SEXE_PATH, "rb") as f: le_sexe = pickle.load(f)
with open(LE_DISTANCE_PATH, "rb") as f: le_distance = pickle.load(f)
with open(LE_BASSIN_PATH, "rb") as f: le_bassin = pickle.load(f)
with open(SCALER_AGE_PATH, "rb") as f: scaler_age = pickle.load(f)

# === FastAPI app ===
app = FastAPI(title="Swim Time Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Utils ===
def pad_sequence(seq, seq_len=SEQ_LEN):
    if len(seq) >= seq_len:
        return seq[-seq_len:]
    else:
        padding = [seq[-1]] * (seq_len - len(seq))
        return padding + seq

def month_to_sin_cos(month):
    """Convert 0-11 month to sin/cos for cyclical encoding"""
    m0 = int(month) % 12
    angle = 2 * math.pi * (m0 / 12.0)
    return math.sin(angle), math.cos(angle)

# === Input schemas ===
class SwimInputReadable(BaseModel):
    perf_nage: str
    nageur_sexe: str
    nageur_age_mois: float
    perf_distance: int
    perf_bassin: int
    mois_saison: int
    perf_temps_sec: float

class SwimSequenceInputReadable(BaseModel):
    sequence: List[SwimInputReadable]

# === API ===
@app.get("/options")
def get_options():
    return {
        "perf_nage": le_nage.classes_.tolist(),
        "nageur_sexe": le_sexe.classes_.tolist(),
        "perf_distance": le_distance.classes_.tolist(),
        "perf_bassin": le_bassin.classes_.tolist(),
        "mois_saison": [
            "Janvier","Février","Mars","Avril","Mai","Juin",
            "Juillet","Août","Septembre","Octobre","Novembre","Décembre"
        ]
    }

# === Load model V3 ===
model = SwimTFT(input_dim=len(FEATURE_COLS), hidden_dim=256, num_layers=4, n_heads=8, dropout=0.3)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "swim_tft_v3.pt"), map_location=DEVICE))
model.to(DEVICE).eval()

@app.post("/predict_seq")
def predict_seq(input_data: SwimSequenceInputReadable):
    seq = pad_sequence(input_data.sequence, SEQ_LEN)
    df = pd.DataFrame([item.model_dump() for item in seq])
    
    df["perf_nage_encoded"] = le_nage.transform(df["perf_nage"])
    df["nageur_sexe_encoded"] = le_sexe.transform(df["nageur_sexe"])
    df["perf_distance_encoded"] = le_distance.transform(df["perf_distance"])
    df["perf_bassin_encoded"] = le_bassin.transform(df["perf_bassin"])
    df["nageur_age_mois_scaled"] = scaler_age.transform(df[["nageur_age_mois"]])
    sin_cos = df["mois_saison"].apply(month_to_sin_cos)
    df["mois_saison_sin"] = [s for s,c in sin_cos]
    df["mois_saison_cos"] = [c for s,c in sin_cos]
    df["perf_temps_sec"] = scaler_y.transform(df[["perf_temps_sec"]])

    X = torch.tensor(df[FEATURE_COLS].values[None, :, :], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        # Le TFT renvoie : (predictions, feature_weights, attention_weights)
        preds, feat_weights, _ = model(X)
        preds = preds.cpu().numpy() # Shape: (1, Horizon, 3)
        feat_weights = feat_weights.cpu().numpy().squeeze() # Shape: (8,)

    # Préparation de la réponse Multi-Horizon (3 étapes d'un coup)
    horizons = []
    for h in range(preds.shape[1]):
        q10 = scaler_y.inverse_transform(preds[0, h, 0:1].reshape(-1, 1)).item()
        q50 = scaler_y.inverse_transform(preds[0, h, 1:2].reshape(-1, 1)).item()
        q90 = scaler_y.inverse_transform(preds[0, h, 2:3].reshape(-1, 1)).item()
        horizons.append({"q10": q10, "q50": q50, "q90": q90})

    print(f'feat_weights: {feat_weights}') # Debug : Affiche les poids d'importance des features
    # On renvoie aussi l'importance des variables pour le front
    feat_weights_avg = feat_weights.mean(axis=0)
    importances = {FEATURE_COLS[i]: float(feat_weights_avg[i]) for i in range(len(FEATURE_COLS))}

    return {
        "predictions": horizons,
        "feature_importance": importances
    }