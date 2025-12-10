# API Swim Time Predictor - Documentation

## Build & Run avec Docker

```bash
# Option 1: docker-compose (depuis la racine)
docker-compose up --build

# Option 2: docker build (depuis app/backend/)
cd app/backend
docker build -t swim-predictor .
docker run -p 8000:8000 swim-predictor
```

Le serveur sera accessible sur `http://localhost:8000`

---

## Endpoints

### GET `/options`

Retourne les valeurs valides pour les champs catégoriels.

**Réponse:**
```json
{
  "perf_nage": ["Nage Libre", "Dos", "Brasse", "Papillon", "4 Nages"],
  "nageur_sexe": ["M", "F"],
  "perf_distance": [50, 100, 200, 400, 800, 1500],
  "perf_bassin": [25, 50],
  "mois_saison": ["Janvier", "Février", ...]
}
```

---

### POST `/predict_seq`

Prédit le temps de nage basé sur une séquence de performances passées.

**Corps de la requête:**
```json
{
  "sequence": [
    {
      "perf_nage": "Nage Libre",
      "nageur_sexe": "M",
      "nageur_age_mois": 180,
      "perf_distance": 100,
      "perf_bassin": 50,
      "mois_saison": 6,
      "perf_temps_sec": 65.5
    }
  ]
}
```

**Champs requis par entrée:**

| Champ | Type | Description | Exemple |
|-------|------|-------------|---------|
| `perf_nage` | string | Type de nage | `"Nage Libre"`, `"Dos"`, `"Brasse"`, `"Papillon"`, `"4 Nages"` |
| `nageur_sexe` | string | Sexe du nageur | `"M"` ou `"F"` |
| `nageur_age_mois` | float | Âge en mois | `180` (15 ans) |
| `perf_distance` | int | Distance en mètres | `50`, `100`, `200`, `400`, `800`, `1500` |
| `perf_bassin` | int | Taille du bassin | `25` ou `50` |
| `mois_saison` | int | Mois (0-11) | `0` = Janvier, `11` = Décembre |
| `perf_temps_sec` | float | Temps en secondes | `65.5` |

**Note sur la séquence:**
- Le modèle attend **10 entrées** (séquence temporelle)
- Si moins de 10 entrées sont fournies, la dernière sera dupliquée pour compléter
- Si plus de 10, seules les 10 dernières sont utilisées
- Les entrées doivent être dans l'**ordre chronologique** (plus ancien → plus récent)

**Réponse:**
```json
{
  "q10": 64.2,
  "q50": 65.1,
  "q90": 66.8
}
```

| Champ | Description |
|-------|-------------|
| `q10` | Borne basse (10e percentile) - scénario optimiste |
| `q50` | Prédiction médiane |
| `q90` | Borne haute (90e percentile) - scénario pessimiste |

---

## Exemple avec curl

```bash
curl -X POST http://localhost:8000/predict_seq \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [
      {"perf_nage": "Nage Libre", "nageur_sexe": "M", "nageur_age_mois": 168, "perf_distance": 100, "perf_bassin": 50, "mois_saison": 0, "perf_temps_sec": 68.0},
      {"perf_nage": "Nage Libre", "nageur_sexe": "M", "nageur_age_mois": 170, "perf_distance": 100, "perf_bassin": 50, "mois_saison": 2, "perf_temps_sec": 67.2},
      {"perf_nage": "Nage Libre", "nageur_sexe": "M", "nageur_age_mois": 172, "perf_distance": 100, "perf_bassin": 50, "mois_saison": 4, "perf_temps_sec": 66.8}
    ]
  }'
```

---

## Exemple Python

```python
import requests

url = "http://localhost:8000/predict_seq"
data = {
    "sequence": [
        {
            "perf_nage": "Nage Libre",
            "nageur_sexe": "M",
            "nageur_age_mois": 180,
            "perf_distance": 100,
            "perf_bassin": 50,
            "mois_saison": 6,
            "perf_temps_sec": 65.5
        }
    ] * 3  # 3 entrées identiques (sera paddé à 10)
}

response = requests.post(url, json=data)
print(response.json())
# {"q10": 64.2, "q50": 65.1, "q90": 66.8}
```

---

## Swagger UI

Documentation interactive disponible sur: `http://localhost:8000/docs`
