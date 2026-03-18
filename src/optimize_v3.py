import optuna
import json
import os
import sys

try:
    from .train_v3 import load_and_prep_data, run_training, FEATURE_COLS
except ImportError:
    from train_v3 import load_and_prep_data, run_training, FEATURE_COLS

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- CONFIG ---
N_TRIALS = 30
SAMPLE_FRAC = 0.2
EPOCHS_PER_TRIAL = 15

def objective(trial):
    params = {
        "epochs": EPOCHS_PER_TRIAL,
        
        # Batch Size: On reste raisonnable pour la VRAM
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
        
        # Learning Rate: Echelle logarithmique
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        
        # TFT Params
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8])
    }
    
    # Contrainte de divisibilité pour Attention (Hidden dim doit être divisible par n_heads)
    if params["hidden_dim"] % params["n_heads"] != 0:
        # On force hidden_dim à être un multiple
        params["hidden_dim"] = (params["hidden_dim"] // params["n_heads"]) * params["n_heads"]

    # 2. Lancement de l'entraînement
    # Note: 'train_data_cache' et 'val_data_cache' doivent être définis globalement
    # pour éviter de recharger à chaque fois (voir bloc main ci-dessous)
    val_loss = run_training(params, train_data_cache, val_data_cache, trial=trial)
    
    return val_loss

if __name__ == "__main__":
    # 1. Chargement des données UNE SEULE FOIS avant l'optimisation
    print(">>> Pré-chargement des données en RAM...")
    train_data_cache, val_data_cache, _ = load_and_prep_data(seq_len=20, horizon=3, sample_frac=SAMPLE_FRAC)
    
    # 2. Setup Optuna
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    
    print("\n>>> Démarrage de l'optimisation...")
    try:
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        print("Optimisation interrompue par l'utilisateur.")

    # 3. Résultats
    print("\n" + "="*40)
    print(" MEILLEURS PARAMÈTRES (TFT V3)")
    print("="*40)
    print(study.best_params)
    print(f"Best Validation Loss: {study.best_value:.4f}")

    # 4. Sauvegarde
    os.makedirs("../models", exist_ok=True)
    with open("../models/best_params_v3.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    print("Paramètres sauvegardés dans models/best_params_v3.json")