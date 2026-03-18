import optuna
import json
import os
import sys

# Import de la V4
try:
    from .train_v4 import load_and_prep_data, run_training
except ImportError:
    from train_v4 import load_and_prep_data, run_training

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- CONFIG ---
N_TRIALS = 30
SAMPLE_FRAC = 0.2  # On garde 20% des nageurs pour aller vite pendant la recherche
EPOCHS_PER_TRIAL = 10

def objective(trial):
    # Paramètres suggérés par Optuna
    params = {
        "epochs": EPOCHS_PER_TRIAL,
        
        # Batch Size: Attention à la VRAM avec les dictionnaires du V4
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
        
        # Learning Rate: Echelle logarithmique
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        
        # TFT V4 Params
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.6),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        
        # Séquence (Fixe pour le moment, mais Optuna pourrait les chercher)
        "seq_len": 20,
        "horizon": 3
    }
    
    # Contrainte de divisibilité (Hidden dim doit être divisible par n_heads)
    if params["hidden_dim"] % params["n_heads"] != 0:
        # On force hidden_dim à être un multiple
        params["hidden_dim"] = (params["hidden_dim"] // params["n_heads"]) * params["n_heads"]

    # Lancement de l'entraînement avec le trial pour le Pruning (arrêt précoce)
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
    print(">>> Pré-chargement des données en RAM pour la V4...")
    train_df_cache, val_df_cache = load_and_prep_data(seq_len=20, horizon=3, sample_frac=SAMPLE_FRAC)
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    os.makedirs("../optuna", exist_ok=True)
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

    #  Résultats (Affiche les meilleurs résultats globaux, depuis le tout premier lancement)
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