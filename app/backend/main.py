import math
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import pickle
import pandas as pd
from typing import List
import numpy as np


# Import du modèle V4
from .SwimTFTv2 import TemporalFusionTransformer

# === Config & Constantes ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 20
HORIZON = 3

# Définition stricte des colonnes V4 (doit matcher train_v4.py)
PAST_COLS = ["perf_temps_sec", "days_since_last_log", "perf_nage_encoded"]
FUTURE_COLS = ["mois_saison_sin", "mois_saison_cos", "perf_distance_encoded", "perf_bassin_encoded", "nageur_age_mois_scaled"]
STATIC_COLS = ["nageur_sexe_encoded"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# === Chargement des Encoders/Scalers ===
SCALER_PATH = os.path.join(MODEL_DIR, "target_scaler.pkl")
LE_NAGE_PATH = os.path.join(MODEL_DIR, "encoder_perf_nage.pkl")
LE_SEXE_PATH = os.path.join(MODEL_DIR, "encoder_sexe.pkl")
LE_DISTANCE_PATH = os.path.join(MODEL_DIR, "encoder_perf_distance.pkl")
LE_BASSIN_PATH = os.path.join(MODEL_DIR, "encoder_perf_bassin.pkl")
SCALER_AGE_PATH = os.path.join(MODEL_DIR, "scaler_age.pkl")

with open(SCALER_PATH, "rb") as f: scaler_y = pickle.load(f)
with open(LE_NAGE_PATH, "rb") as f: le_nage = pickle.load(f)
with open(LE_SEXE_PATH, "rb") as f: le_sexe = pickle.load(f)
with open(LE_DISTANCE_PATH, "rb") as f: le_distance = pickle.load(f)
with open(LE_BASSIN_PATH, "rb") as f: le_bassin = pickle.load(f)
with open(SCALER_AGE_PATH, "rb") as f: scaler_age = pickle.load(f)

# === FastAPI app ===
app = FastAPI(title="Swim Time Predictor V4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Utils ===
def pad_sequence(seq, seq_len):
    """Pad une liste d'objets en dupliquant le dernier élément si nécessaire"""
    if len(seq) >= seq_len:
        return seq[-seq_len:]
    padding = [seq[-1]] * (seq_len - len(seq))
    return padding + seq

def month_to_sin_cos(month):
    m0 = int(month) % 12
    angle = 2 * math.pi * (m0 / 12.0)
    return math.sin(angle), math.cos(angle)

# === Input Schemas ===
class PastSwimInput(BaseModel):
    perf_nage: str
    nageur_sexe: str
    nageur_age_mois: float
    perf_distance: int
    perf_bassin: int
    mois_saison: int
    perf_temps_sec: float
    days_since_last: float

class FutureSwimInput(BaseModel):
    nageur_age_mois: float
    perf_distance: int
    perf_bassin: int
    mois_saison: int

class SwimPredictRequest(BaseModel):
    past_sequence: List[PastSwimInput]
    future_sequence: List[FutureSwimInput] = Field(..., min_items=HORIZON, max_items=HORIZON)

# === Chargement du Modèle V4 ===
def load_v4_model():
    # Chargement des hyperparamètres optimisés s'ils existent
    params = {"hidden_dim": 128, "n_heads": 4, "dropout": 0.2} # Valeurs par défaut
    try:
        with open(os.path.join(MODEL_DIR, "best_params_v4.json"), "r") as f:
            params.update(json.load(f))
    except FileNotFoundError:
        pass

    model = TemporalFusionTransformer(
        number_of_past_inputs=SEQ_LEN,
        horizon=HORIZON,
        embedding_size_inputs=1,
        hidden_dimension=params["hidden_dim"],
        dropout=params["dropout"],
        number_of_heads=params["n_heads"],
        past_inputs={c: 1 for c in PAST_COLS},
        future_inputs={c: 1 for c in FUTURE_COLS},
        static_inputs={c: 1 for c in STATIC_COLS},
        batch_size=1, # Inférence = batch de 1
        device=DEVICE,
        quantiles=[0.1, 0.5, 0.9]
    ).to(DEVICE)
    
    model_path = os.path.join(MODEL_DIR, "swim_tft_v4.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    return model.eval()

model = load_v4_model()

# === API Endpoints ===
@app.get("/options")
def get_options():
    return {
        "perf_nage": le_nage.classes_.tolist(),
        "nageur_sexe": le_sexe.classes_.tolist(),
        "perf_distance": le_distance.classes_.tolist(),
        "perf_bassin": le_bassin.classes_.tolist(),
        "mois_saison": ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
    }

@app.post("/predict_v4")
def predict_v4(input_data: SwimPredictRequest):
    if len(input_data.future_sequence) != HORIZON:
        raise HTTPException(status_code=400, detail=f"future_sequence must have exactly {HORIZON} items.")

    # 1. Traitement du Passé
    past_seq = pad_sequence(input_data.past_sequence, SEQ_LEN)
    df_past = pd.DataFrame([item.model_dump() for item in past_seq])
    
    df_past["perf_temps_sec"] = scaler_y.transform(df_past[["perf_temps_sec"]])
    df_past["perf_nage_encoded"] = le_nage.transform(df_past["perf_nage"])
    df_past["nageur_sexe_encoded"] = le_sexe.transform(df_past["nageur_sexe"])
    df_past["days_since_last_log"] = np.log1p(df_past["days_since_last"])
    
    # 2. Traitement du Futur
    df_future = pd.DataFrame([item.model_dump() for item in input_data.future_sequence])
    df_future["perf_distance_encoded"] = le_distance.transform(df_future["perf_distance"])
    df_future["perf_bassin_encoded"] = le_bassin.transform(df_future["perf_bassin"])
    df_future["nageur_age_mois_scaled"] = scaler_age.transform(df_future[["nageur_age_mois"]])
    
    sin_cos = df_future["mois_saison"].apply(month_to_sin_cos)
    df_future["mois_saison_sin"] = [s for s,c in sin_cos]
    df_future["mois_saison_cos"] = [c for s,c in sin_cos]

    # 3. Création des dictionnaires de tenseurs [Batch(1), SeqLen/Horizon, 1]
    past_tensors = {
        col: torch.tensor(df_past[col].values, dtype=torch.float32).view(1, SEQ_LEN, 1).to(DEVICE)
        for col in PAST_COLS
    }
    
    future_tensors = {
        col: torch.tensor(df_future[col].values, dtype=torch.float32).view(1, HORIZON, 1).to(DEVICE)
        for col in FUTURE_COLS
    }
    
    # Static prend juste la première valeur du passé [Batch(1), 1]
    static_tensors = {
        col: torch.tensor([df_past[col].iloc[0]], dtype=torch.float32).view(1, 1).to(DEVICE)
        for col in STATIC_COLS
    }

    # 4. Inférence
    with torch.no_grad():
        preds, feat_weights = model(past_inputs=past_tensors, future_inputs=future_tensors, static_inputs=static_tensors)
        if preds.dim() == 4:
            preds = preds.squeeze(2) 
        preds = preds.cpu().numpy()
    
    importances_brutes = {}
        
    if isinstance(feat_weights, dict):
        # 1. Moyenne sur la séquence pour le passé
        if 'past' in feat_weights:
            avg_past = feat_weights['past'].mean(dim=(0, 1)).cpu().numpy()
            for i, col in enumerate(PAST_COLS):
                importances_brutes[col] = float(avg_past[i])
        
        # 2. Moyenne sur l'horizon pour le futur
        if 'future' in feat_weights:
            avg_future = feat_weights['future'].mean(dim=(0, 1)).cpu().numpy()
            for i, col in enumerate(FUTURE_COLS):
                importances_brutes[col] = float(avg_future[i])
        
        # 3. Importance statique
        if 'static' in feat_weights:
            avg_static = feat_weights['static'].mean(dim=0).cpu().numpy()
            for i, col in enumerate(STATIC_COLS):
                importances_brutes[col] = float(avg_static[i])
        
        if 'temporal_attention' in feat_weights:
            attn_t = feat_weights['temporal_attention']
            
            temporal_scores = attn_t.mean(dim=(1, 2)).squeeze().cpu().numpy().tolist()
            res_temporal = [round(float(s), 4) for s in temporal_scores]
        else:
            res_temporal = []

        total = sum(importances_brutes.values()) + 1e-8
        importances_finales = {k: v / total for k, v in importances_brutes.items()}
    else:
        print("Attention: feat_weights n'est pas un dictionnaire, feature importance brute non calculée.")
        importances_finales = {col: None for col in PAST_COLS + FUTURE_COLS + STATIC_COLS}
        
    # 5. Formatage de la réponse
    horizons_res = []
    for h in range(HORIZON):
        q10 = scaler_y.inverse_transform(preds[0, h, 0:1].reshape(-1, 1)).item()
        q50 = scaler_y.inverse_transform(preds[0, h, 1:2].reshape(-1, 1)).item()
        q90 = scaler_y.inverse_transform(preds[0, h, 2:3].reshape(-1, 1)).item()
        
        # Sécurité pour ne pas avoir de quantiles qui se croisent
        q10, q90 = min(q10, q50), max(q90, q50)
        
        horizons_res.append({"horizon": h+1, "q10": round(q10, 2), "q50": round(q50, 2), "q90": round(q90, 2)})

    print(f">>> [PREDICTION] Horizons: {horizons_res}, Feature Importance: {feat_weights}")
    return {
        "predictions": horizons_res,
        "feature_importance": importances_finales,
        "temporal_attention": res_temporal,
        "message": "Prédiction V4 réussie."
    }