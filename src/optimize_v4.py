import optuna
import json
import os
import pickle 

try:
    from .train_v4 import load_and_prep_data, run_training
except ImportError:
    from train_v4 import load_and_prep_data, run_training

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- CONFIG ---
N_TRIALS = 30
SAMPLE_FRAC = 0.2  # On garde 20% des nageurs pour aller vite pendant la recherche
EPOCHS_PER_TRIAL = 10

# --- CACHE ---
CACHE_DIR = "../optuna/data_cache"
TRAIN_CACHE_PATH = os.path.join(CACHE_DIR, f"train_cache_v4_{SAMPLE_FRAC}.pkl")
VAL_CACHE_PATH = os.path.join(CACHE_DIR, f"val_cache_v4_{SAMPLE_FRAC}.pkl")

def objective(trial):
    params = {
        "epochs": EPOCHS_PER_TRIAL,
        
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
        
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.6),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        
        "seq_len": 20,
        "horizon": 3
    }
    
    # Contrainte de divisibilité (Hidden dim doit être divisible par n_heads)
    if params["hidden_dim"] % params["n_heads"] != 0:
        # On force hidden_dim à être un multiple
        params["hidden_dim"] = (params["hidden_dim"] // params["n_heads"]) * params["n_heads"]

    # Lancement de l'entraînement avec le trial pour le pruning
    val_loss = run_training(
        params=params, 
        train_df=train_df_cache, 
        val_df=val_df_cache, 
        trial=trial,
        save_model=False, # On ne sauvegarde pas les modèles intermédiaires
        plot_loss=False   # On ne plot pas pour ne pas bloquer la boucle
    )
    
    return val_loss

if __name__ == "__main__":
    os.makedirs("../optuna", exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- LOGIQUE DE CACHE DES DONNÉES ---
    if os.path.exists(TRAIN_CACHE_PATH) and os.path.exists(VAL_CACHE_PATH):
        print(f">>> [CACHE] Chargement des données d'optimisation strictes depuis {CACHE_DIR}...")
        with open(TRAIN_CACHE_PATH, "rb") as f:
            train_df_cache = pickle.load(f)
        with open(VAL_CACHE_PATH, "rb") as f:
            val_df_cache = pickle.load(f)
    else:
        print(">>> [INIT] Pré-chargement et échantillonnage initial des données pour la V4...")
        train_df_cache, val_df_cache = load_and_prep_data(seq_len=20, horizon=3, sample_frac=SAMPLE_FRAC)
        
        print(">>> [CACHE] Sauvegarde de l'échantillon pour garantir l'équité des prochains lancements...")
        with open(TRAIN_CACHE_PATH, "wb") as f:
            pickle.dump(train_df_cache, f)
        with open(VAL_CACHE_PATH, "wb") as f:
            pickle.dump(val_df_cache, f)
    # -----------------------------------
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    db_path = os.path.join(os.path.dirname(__file__), "../optuna/optuna_v4.db")
    storage_url = f"sqlite:///{db_path}"
    study_name = "tft_v4_optimization"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize", 
        pruner=pruner
    )
    
    print(f"\n>>> Démarrage/Reprise de l'optimisation Optuna (TFT V4)")
    print(f"Base de données : {db_path}")
    print(f"Essais déjà complétés dans la BDD : {len(study.trials)}")
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n[!] Optimisation interrompue par l'utilisateur (Ctrl+C).")
        print("[!] Progression sauvegardée en toute sécurité dans la BDD.")

    #  Résultats
    if len(study.trials) > 0:
        print("\n" + "="*40)
        print(" MEILLEURS PARAMÈTRES GLOBAUX (TFT V4)")
        print("="*40)
        print(study.best_params)
        print(f"Best Validation Loss: {study.best_value:.5f}")

        best_params_final = study.best_params
        if best_params_final["hidden_dim"] % best_params_final["n_heads"] != 0:
            best_params_final["hidden_dim"] = (best_params_final["hidden_dim"] // best_params_final["n_heads"]) * best_params_final["n_heads"]
        
        with open("../models/best_params_v4.json", "w") as f:
            json.dump(best_params_final, f, indent=4)
            
        print("Paramètres sauvegardés dans ../models/best_params_v4.json")