import React, { useState, useEffect, useMemo, useRef } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import "chart.js/auto";
import "./App.css";

// --- UTILS ---
const calculateAgeMonths = (birthDate, eventDate) => {
  const birth = new Date(birthDate);
  const event = new Date(eventDate);
  let months = (event.getFullYear() - birth.getFullYear()) * 12;
  months -= birth.getMonth();
  months += event.getMonth();
  return months > 0 ? months : 0;
};

const formatAgeReadable = (totalMonths) => {
  if (!totalMonths) return "";
  const years = Math.floor(totalMonths / 12);
  const months = totalMonths % 12;
  return `${years} ans${months > 0 ? ` ${months} mois` : ''}`;
};

const formatDate = (dateStr) => {
  if (!dateStr) return "";
  return new Date(dateStr).toLocaleDateString("fr-FR");
};

// --- COMPOSANTS ---

const UserProfile = ({ user, setUser, options }) => {
  const handleChange = (e) => setUser({ ...user, [e.target.name]: e.target.value });

  return (
    <div className="profile-section">
      <h2>👤 Profil Nageur</h2>
      <div className="form-group">
        <label>Date de naissance</label>
        <input type="date" name="dob" value={user.dob} onChange={handleChange} />
      </div>
      <div className="form-group">
        <label>Sexe</label>
        <select name="sexe" value={user.sexe} onChange={handleChange}>
          {options?.nageur_sexe?.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      </div>
      <div className="profile-info">
        L'âge est calculé automatiquement en fonction de la date de la course.
      </div>
    </div>
  );
};

export default function App() {
  const [options, setOptions] = useState(null);

  // Référence pour l'input file caché
  const fileInputRef = useRef(null);

  const [user, setUser] = useState({
    dob: "2005-01-01",
    sexe: "M"
  });

  const [fullHistory, setFullHistory] = useState([]);

  const [selectedNage, setSelectedNage] = useState("Nage Libre");
  const [selectedDistance, setSelectedDistance] = useState("100");
  const [selectedBassin, setSelectedBassin] = useState("50");

  const [addForm, setAddForm] = useState({
    date: new Date().toISOString().split('T')[0],
    time: "",
  });

  const [predictionCache, setPredictionCache] = useState({});

  const currentDisciplineKey = useMemo(() => {
    return `${selectedNage}-${selectedDistance}-${selectedBassin}`;
  }, [selectedNage, selectedDistance, selectedBassin]);

  const results = predictionCache[currentDisciplineKey] || null;

  useEffect(() => {
    axios.get("http://localhost:8000/options").then((res) => {
      setOptions(res.data);
      if (res.data.nageur_sexe) setUser(u => ({ ...u, sexe: res.data.nageur_sexe[0] }));
    });
  }, []);

  // --- LOGIQUE IMPORT JSON ---

  const handleJsonImport = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const jsonData = JSON.parse(event.target.result);

        // Validation basique et transformation
        if (!Array.isArray(jsonData)) throw new Error("Le format doit être un tableau JSON");

        const enrichedHistory = jsonData.map(item => {
          // Calculs dérivés basés sur le profil actuel
          const ageMois = calculateAgeMonths(user.dob, item.date);
          const dateObj = new Date(item.date);
          const moisSaison = dateObj.getMonth();

          return {
            perf_nage: item.nage,
            nageur_sexe: user.sexe,
            perf_distance: parseInt(item.distance),
            perf_bassin: parseInt(item.bassin),
            mois_saison: moisSaison,
            nageur_age_mois: ageMois,
            perf_temps_sec: parseFloat(item.temps),
            date: item.date
          };
        });

        // Fusion avec l'historique existant
        setFullHistory(prev => [...prev, ...enrichedHistory]);
        alert(`${enrichedHistory.length} performances importées !`);

      } catch (err) {
        console.error(err);
        alert("Erreur lors de la lecture du fichier JSON. Vérifiez le format.");
      }
    };
    reader.readAsText(file);
    // Reset input value pour permettre de réimporter le même fichier si besoin
    e.target.value = null;
  };

  // --- LOGIQUE FILTRAGE ---

  const filteredHistory = useMemo(() => {
    return fullHistory.filter(h =>
      h.perf_nage === selectedNage &&
      String(h.perf_distance) === String(selectedDistance) &&
      String(h.perf_bassin) === String(selectedBassin)
    ).sort((a, b) => new Date(a.date) - new Date(b.date));
  }, [fullHistory, selectedNage, selectedDistance, selectedBassin]);

  useEffect(() => {
    if (predictionCache[currentDisciplineKey]) return;
    if (filteredHistory.length > 0) {
      predictSequence(filteredHistory, currentDisciplineKey);
    }
  }, [currentDisciplineKey, filteredHistory.length]); // eslint-disable-line

  // --- LOGIQUE AJOUT & PREDICTION ---

  const handleAddPerformance = async () => {
    if (!addForm.time || !user.dob) {
      alert("Veuillez remplir le temps et vérifier votre date de naissance.");
      return;
    }

    const ageMois = calculateAgeMonths(user.dob, addForm.date);
    const dateObj = new Date(addForm.date);
    const moisSaison = dateObj.getMonth();

    const newEntry = {
      perf_nage: selectedNage,
      nageur_sexe: user.sexe,
      perf_distance: parseInt(selectedDistance),
      perf_bassin: parseInt(selectedBassin),
      mois_saison: moisSaison,
      nageur_age_mois: ageMois,
      perf_temps_sec: parseFloat(addForm.time),
      date: addForm.date
    };

    const updatedFiltered = [...filteredHistory, newEntry];
    setFullHistory(prevHistory => [...prevHistory, newEntry]);
    setAddForm({ ...addForm, time: "" });

    await predictSequence(updatedFiltered, currentDisciplineKey);
  };

  const predictSequence = async (sequenceData, keyToCache) => {
    if (sequenceData.length === 0) return;

    try {
      const res = await axios.post("http://localhost:8000/predict_seq", {
        sequence: sequenceData,
      });

      setPredictionCache(prevCache => ({
        ...prevCache,
        [keyToCache]: {
          predictions: res.data.predictions, // Tableau de 3 objets {q10, q50, q90}
          importance: res.data.feature_importance
        },
      }));
    } catch (err) {
      console.error("Erreur prédiction:", err);
    }
  };

  const ImportanceWidget = ({ importance }) => {
    if (!importance) return null;

    const featToLabel = {
      "perf_temps_sec": "Chronos précédents",
      "nageur_age_mois_scaled": "Âge & Croissance",
      "nageur_sexe_encoded": "Genre",
      "perf_distance_encoded": "Distance de l'épreuve",
      "perf_bassin_encoded": "Taille du bassin",
      "perf_nage_encoded": "Type de nage"
    };

    const processedImportance = () => {
      const entries = Object.entries(importance);
      const result = {};
      let saisonSum = 0;

      entries.forEach(([key, val]) => {
        if (key === 'mois_saison_sin' || key === 'mois_saison_cos') {
          saisonSum += val;
        } else {
          result[key] = val;
        }
      });

      // On ajoute l'entrée unique pour la saison
      result['saison_grouped'] = saisonSum;
      return Object.entries(result);
    };

    // On trie par importance décroissante
    const sorted = processedImportance()
      .sort(([, a], [, b]) => b - a);

    return (
      <div className="card" style={{ background: '#1a1a1a' }}>
        <h4>🧠 Pourquoi cette prédiction ?</h4>
        <h5 style={{ color: '#ccc', marginBottom: '10px', fontSize: '0.85rem' }}>
          Facteurs d'influence identifiés par l'IA :
        </h5>
        <div style={{ marginTop: '10px' }}>
          {sorted.map(([key, val]) => (
            <div key={key} style={{ marginBottom: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                <span style={{ color: '#aaa' }}>
                  {key === 'saison_grouped' ? "Saisonnalité" : (featToLabel[key] || key)}
                </span>
                <span style={{ color: '#00d4ff', fontWeight: 'bold' }}>
                  {(val * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{ height: '6px', background: '#333', borderRadius: '3px', marginTop: '6px' }}>
                <div
                  style={{
                    width: `${val * 100}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, #005f73, #00d4ff)',
                    borderRadius: '3px',
                    transition: 'width 0.5s ease-out'
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // --- CHART DATA ---
  const getChartData = () => {
    if (!filteredHistory.length && !results) return null;
    const realData = filteredHistory.map(h => h.perf_temps_sec);
    const labels = filteredHistory.map(h => formatDate(h.date));
    const predLabels = ["Prochaine", "J+2", "J+3"];
    console.log('results:', results);
    const predDataQ50 = results ? results.predictions.map(r => r.q50) : [];
    const predDataQ10 = results ? results.predictions.map(r => r.q10) : [];
    const predDataQ90 = results ? results.predictions.map(r => r.q90) : [];
    const padding = Array(realData.length - 1).fill(null);
    const lastRealVal = realData[realData.length - 1] || 0;

    return {
      labels: [...labels, ...predLabels],
      datasets: [
        {
          label: "Performance Réelle",
          data: [...realData, null, null, null],
          borderColor: "#00d4ff",
          backgroundColor: "#00d4ff",
          tension: 0.2,
          pointRadius: 6,
        },
        {
          label: "Q10 (Meilleur)",
          data: [...padding, lastRealVal, ...predDataQ10],
          borderColor: "rgba(255, 0, 122, 0.3)",
          borderDash: [5, 5],
          pointRadius: 4,
          fill: false,
        },
        {
          label: "Intervalle 10-90%",
          data: [...padding, lastRealVal, ...predDataQ90],
          borderColor: "rgba(255, 0, 122, 0.3)",
          backgroundColor: "rgba(255, 0, 122, 0.15)",
          borderDash: [5, 5],
          pointRadius: 4,
          fill: 1,
        },
        {
          label: "Prédiction",
          data: [...padding, lastRealVal, ...predDataQ50],
          borderColor: "#ff007a",
          borderWidth: 3,
          borderDash: [2, 2],
          pointRadius: 5,
          fill: false,
        }
      ],
    };
  };

  if (!options) return <div className="app-container">Chargement...</div>;

  return (
    <div className="app-container">
      {/* SIDEBAR GAUCHE */}
      <aside className="sidebar">
        <h1 style={{ color: 'white' }}>🏊 SwimAI</h1>
        <UserProfile user={user} setUser={setUser} options={options} />

        <div className="card" style={{ background: 'rgba(255,255,255,0.1)', marginTop: '1rem' }}>
          <h4>📁 Importer JSON</h4>
          <p style={{ fontSize: '0.75rem', color: '#ccc', marginBottom: '0.5rem' }}>
            Assurez-vous que la date de naissance ci-dessus est correcte avant d'importer.
          </p>
          <input
            type="file"
            accept=".json"
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={handleJsonImport}
          />
          <button
            className="btn-primary"
            style={{ width: '100%', fontSize: '0.8rem', background: '#333', border: '1px solid #555' }}
            onClick={() => fileInputRef.current.click()}
          >
            Choisir un fichier
          </button>
        </div>

        <div className="card" style={{ marginTop: 'auto', background: 'rgba(255,255,255,0.05)' }}>
          <h3>Astuce</h3>
          <p style={{ fontSize: '0.85rem', color: '#aaa' }}>
            Sélectionnez une épreuve ci-dessus pour voir l'historique spécifique.
          </p>
        </div>
      </aside>

      {/* CONTENU PRINCIPAL */}
      <main className="main-content">
        {/* 1. SELECTION DU CONTEXTE (ONLGETS) */}
        <div className="tabs">
          {options.perf_nage.map(nage => (
            <button
              key={nage}
              className={`tab-btn ${selectedNage === nage ? "active" : ""}`}
              onClick={() => { setSelectedNage(nage); }}
            >
              {nage}
            </button>
          ))}
        </div>

        <div className="controls-bar">
          <div style={{ flex: 1 }}>
            <label style={{ color: '#aaa', fontSize: '0.8rem' }}>Distance</label>
            <select value={selectedDistance} onChange={e => { setSelectedDistance(e.target.value); }}>
              {options.perf_distance.map(d => <option key={d} value={d}>{d}m</option>)}
            </select>
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ color: '#aaa', fontSize: '0.8rem' }}>Bassin</label>
            <select value={selectedBassin} onChange={e => { setSelectedBassin(e.target.value); }}>
              {options.perf_bassin.map(b => <option key={b} value={b}>{b}m</option>)}
            </select>
          </div>
        </div>

        {/* 2. GRAPHIQUE & STATS */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3>📈 Progression : {selectedDistance}m {selectedNage} ({selectedBassin}m)</h3>
            {filteredHistory.length > 0 && (
              <span style={{ color: '#00d4ff' }}>Record: {Math.min(...filteredHistory.map(h => h.perf_temps_sec))}s</span>
            )}
          </div>

          {results && (
            <>
              <div className="prediction-stats">
                <div className="stat-box">
                  <div className="stat-value">{results.predictions[0].q10.toFixed(2)}s</div>
                  <div className="stat-label">Best Case (Q10)</div>
                </div>
                <div className="stat-box">
                  <div className="stat-value" style={{ color: '#fff' }}>{results.predictions[0].q50.toFixed(2)}s</div>
                  <div className="stat-label">Attendu (Moyenne)</div>
                </div>
                <div className="stat-box">
                  <div className="stat-value">{results.predictions[0].q90.toFixed(2)}s</div>
                  <div className="stat-label">Worst Case (Q90)</div>
                </div>
              </div>

              <ImportanceWidget importance={results.importance} />
            </>
          )}

          <div style={{ height: "300px", width: "100%" }}>
            {filteredHistory.length > 0 ? (
              <Line
                data={getChartData()}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: { legend: { display: true } },
                  scales: {
                    x: { grid: { color: '#333' } },
                    y: { grid: { color: '#333' } }
                  }
                }}
              />
            ) : (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666' }}>
                Aucune donnée pour cette épreuve. Ajoutez une performance manuellement ou importez un JSON.
              </div>
            )}
          </div>
        </div>

        {/* 3. FORMULAIRE D'AJOUT MANUEL */}
        <div className="card" style={{ borderLeft: '4px solid #00d4ff' }}>
          <h3>⏱️ Ajouter une performance</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: '1rem', alignItems: 'end' }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label>Date</label>
              <input type="date" value={addForm.date} onChange={(e) => setAddForm({ ...addForm, date: e.target.value })} />
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label>Temps (sec)</label>
              <input type="number" step="0.01" value={addForm.time} onChange={(e) => setAddForm({ ...addForm, time: e.target.value })} />
            </div>
            <button className="btn-primary" style={{ height: '42px' }} onClick={handleAddPerformance}>Ajouter</button>
          </div>
        </div>

        {/* 4. TABLEAU HISTORIQUE */}
        {filteredHistory.length > 0 && (
          <div className="card">
            <h3>Historique Détaillé</h3>
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Age</th>
                  <th>Temps</th>
                </tr>
              </thead>
              <tbody>
                {filteredHistory.slice().reverse().map((h, i) => (
                  <tr key={i}>
                    <td>{formatDate(h.date)}</td>
                    <td style={{ color: '#aaa' }}>{formatAgeReadable(h.nageur_age_mois)}</td>
                    <td style={{ color: '#fff', fontWeight: 'bold' }}>{h.perf_temps_sec} s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  );
}