import React, { useState, useEffect, useMemo } from "react";
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

  const result = predictionCache[currentDisciplineKey] || null;

  useEffect(() => {
    axios.get("http://localhost:8000/options").then((res) => {
      setOptions(res.data);
      // Set defaults
      if (res.data.nageur_sexe) setUser(u => ({ ...u, sexe: res.data.nageur_sexe[0] }));
    });
  }, []);

  // --- LOGIQUE FILTRAGE ---

  const filteredHistory = useMemo(() => {
    return fullHistory.filter(h =>
      h.perf_nage === selectedNage &&
      String(h.perf_distance) === String(selectedDistance) &&
      String(h.perf_bassin) === String(selectedBassin)
    ).sort((a, b) => new Date(a.date) - new Date(b.date)); // Tri chronologique
  }, [fullHistory, selectedNage, selectedDistance, selectedBassin]);

  useEffect(() => {
    if (predictionCache[currentDisciplineKey]) {
      return;
    }

    if (filteredHistory.length > 0) {
      predictSequence(filteredHistory, currentDisciplineKey);
    }

  }, [currentDisciplineKey, filteredHistory.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- LOGIQUE AJOUT & PREDICTION ---

  const handleAddPerformance = async () => {
    if (!addForm.time || !user.dob) {
      alert("Veuillez remplir le temps et vérifier votre date de naissance.");
      return;
    }

    const ageMois = calculateAgeMonths(user.dob, addForm.date);
    const dateObj = new Date(addForm.date);
    const moisSaison = dateObj.getMonth(); // 0-11

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

    let currentSequence = [...sequenceData];
    const steps = 3;
    const multiPredictionResult = [];

    try {
      for (let i = 0; i < steps; i++) {
        const res = await axios.post("http://localhost:8000/predict_seq", {
          sequence: currentSequence,
        });

        const pred = res.data;
        multiPredictionResult.push(pred);

        const lastItem = currentSequence[currentSequence.length - 1];
        currentSequence = [
          ...currentSequence,
          {
            ...lastItem,
            perf_temps_sec: pred.q50,
            nageur_age_mois: lastItem.nageur_age_mois + 1,
            mois_saison: (lastItem.mois_saison + 1) % 12
          },
        ];
      }

      setPredictionCache(prevCache => ({
        ...prevCache,
        [keyToCache]: multiPredictionResult,
      }));

    } catch (err) {
      console.error("Erreur prédiction:", err);
      setPredictionCache(prevCache => {
        const newCache = { ...prevCache };
        delete newCache[keyToCache];
        return newCache;
      });
    }
  };

  // --- CHART DATA PREPARATION ---

  const getChartData = () => {
    if (!filteredHistory.length && !result) return null;

    const realData = filteredHistory.map(h => h.perf_temps_sec);
    const labels = filteredHistory.map(h => formatDate(h.date));

    // Ajout des labels prédictifs
    const predLabels = ["Prochaine", "J+2", "J+3"];

    const predDataQ50 = result ? result.map(r => r.q50) : [];
    const predDataQ10 = result ? result.map(r => r.q10) : [];
    const predDataQ90 = result ? result.map(r => r.q90) : [];

    // Padding pour aligner les prédictions
    const padding = Array(realData.length - 1).fill(null);
    const lastRealVal = realData[realData.length - 1] || 0;

    return {
      labels: [...labels, ...predLabels],
      datasets: [
        // Index 0: Historique réel
        {
          label: "Performance Réelle",
          data: [...realData, null, null, null],
          borderColor: "#00d4ff",
          backgroundColor: "#00d4ff",
          tension: 0.2,
          pointRadius: 6,
          pointHoverRadius: 8,
        },
        // Index 1: Q10
        {
          label: "Q10 (Meilleur cas)",
          data: [...padding, lastRealVal, ...predDataQ10],
          borderColor: "rgba(255, 0, 122, 0.3)",
          backgroundColor: "transparent",
          borderDash: [5, 5],
          pointRadius: 4,
          pointBackgroundColor: "rgba(255, 0, 122, 0.3)",
          fill: false,
        },
        // Index 2: Q90
        {
          label: "Intervalle 10-90%",
          data: [...padding, lastRealVal, ...predDataQ90],
          borderColor: "rgba(255, 0, 122, 0.3)",
          backgroundColor: "rgba(255, 0, 122, 0.15)",
          borderDash: [5, 5],
          pointRadius: 4,
          pointBackgroundColor: "rgba(255, 0, 122, 0.3)",
          fill: 1,
        },
        // Index 3: Q50 (Médiane)
        {
          label: "Prédiction (Médiane)",
          data: [...padding, lastRealVal, ...predDataQ50],
          borderColor: "#ff007a",
          borderWidth: 3,
          borderDash: [2, 2],
          pointRadius: 5,
          pointBackgroundColor: "#ff007a",
          pointBorderColor: "#fff",
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

        <div className="card" style={{ marginTop: 'auto', background: 'rgba(255,255,255,0.05)' }}>
          <h3>Astuce</h3>
          <p style={{ fontSize: '0.85rem', color: '#aaa' }}>
            Sélectionnez une épreuve ci-dessus pour voir l'historique spécifique et lancer une prédiction.
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
              onClick={() => {
                setSelectedNage(nage);
              }}
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

          {result && (
            <div className="prediction-stats">
              <div className="stat-box">
                <div className="stat-value">{result[0].q10.toFixed(2)}s</div>
                <div className="stat-label">Best Case (Q10)</div>
              </div>
              <div className="stat-box">
                <div className="stat-value" style={{ color: '#fff' }}>{result[0].q50.toFixed(2)}s</div>
                <div className="stat-label">Attendu (Moyenne)</div>
              </div>
              <div className="stat-box">
                <div className="stat-value">{result[0].q90.toFixed(2)}s</div>
                <div className="stat-label">Worst Case (Q90)</div>
              </div>
            </div>
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
                Aucune donnée pour cette épreuve. Ajoutez une performance ci-dessous.
              </div>
            )}
          </div>
        </div>

        {/* 3. FORMULAIRE D'AJOUT CONTEXTUEL */}
        <div className="card" style={{ borderLeft: '4px solid #00d4ff' }}>
          <h3>⏱️ Ajouter une performance</h3>
          <p style={{ fontSize: '0.9rem', color: '#888', marginBottom: '1rem' }}>
            Ajout pour : <b>{selectedDistance}m {selectedNage}</b> en bassin de <b>{selectedBassin}m</b>
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: '1rem', alignItems: 'end' }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label>Date de la course</label>
              <input
                type="date"
                value={addForm.date}
                onChange={(e) => setAddForm({ ...addForm, date: e.target.value })}
              />
            </div>

            <div className="form-group" style={{ marginBottom: 0 }}>
              <label>Temps (secondes)</label>
              <input
                type="number"
                step="0.01"
                placeholder="Ex: 65.40"
                value={addForm.time}
                onChange={(e) => setAddForm({ ...addForm, time: e.target.value })}
              />
            </div>

            <button className="btn-primary" style={{ height: '42px' }} onClick={handleAddPerformance}>
              Ajouter & Prédire
            </button>
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

                    <td style={{ color: '#aaa' }}>
                      {formatAgeReadable(h.nageur_age_mois)}
                    </td>

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