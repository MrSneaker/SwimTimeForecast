# FFNBoardProject

Projet combinant un backend FastAPI pour le traitement des données et un frontend React pour l'affichage.

## Arborescence du projet

```bash
.
├── app
│   ├── backend
│   │   ├── main.py
│   │   └── models
│   │  
│   └── frontend
│       ├── eslint.config.js
│       ├── index.html
│       ├── package.json
│       ├── package-lock.json
│       ├── public
│       ├── src
│       └── vite.config.js
├── data
├── logs
├── model_result
├── models
├── notebooks
├── pids
├── scripts
└── src
```

## Prérequis

- [Conda](https://docs.conda.io/en/latest/miniconda.html) installé
- [Node.js](https://nodejs.org/) version 18+ et npm
- [Git](https://git-scm.com/)

## Installation des dépendances

### 1. Créer un environnement Conda

Il est recommandé de créer un environnement dédié pour le projet :

```bash
conda create -n ffnboard python=3.12 -y
conda activate ffnboard
```

### 2. Installer les dépendances Python

Depuis la racine du projet :

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Vérifiez que `torch` et `fastapi` sont bien installés :

```bash
python -c "import torch; print(torch.__version__)"
python -c "import fastapi; print('FastAPI OK')"
```

### 3. Installer les dépendances frontend (Node.js / npm)

Depuis le dossier frontend :

```bash
cd app/frontend
npm install
```

Vous pouvez tester que le frontend fonctionne :

```bash
npm run dev
```

Le frontend sera accessible par défaut sur [http://localhost:5173](http://localhost:5173).

## Lancer l’application

Un script bash `run_app.sh` est fourni pour démarrer ou arrêter le backend et le frontend simultanément.

### Commandes disponibles

```bash
bash scripts/run_app.sh start     # Lancer backend + frontend
bash scripts/run_app.sh stop      # Arrêter backend + frontend
bash scripts/run_app.sh restart   # Redémarrer backend + frontend
bash scripts/run_app.sh status    # Vérifier le statut des processus
```

### Notes

- Les logs sont enregistrés dans le dossier `logs/` (`backend.log` et `frontend.log`).
- Les PID des processus sont sauvegardés dans `pids/`.
- Le backend écoute par défaut sur le port __8000__.
- Le frontend écoute par défaut sur le port __5173__.

## Structure des dossiers importants

- `app/backend` : code FastAPI et modèles Python.
- `app/frontend` : application React avec Vite.
- `data` : données CSV utilisées par le projet.
- `models` : modèles pré-entraînés et scalers.
- `model_result` : résultats graphiques générés.
- `scripts/run_app.sh` : script pour lancer/arrêter l’application.
- `logs` : logs backend et frontend.
- `pids` : stockage des PID pour le contrôle des processus.
