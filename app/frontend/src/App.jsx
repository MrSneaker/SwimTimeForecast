import React, { useState, useEffect, useMemo, useRef } from "react";
import axios from "axios";
import { Chart } from "react-chartjs-2";
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

const getDaysBetween = (date1, date2) => {
  const d1 = new Date(date1);
  const d2 = new Date(date2);
  const diffTime = Math.abs(d2 - d1);
  return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
};

// --- COMPOSANTS INTERNES ---

const ImportanceWidget = ({ importance }) => {
  if (!importance || Object.keys(importance).length === 0) return null;

  const featToLabel = {
    "perf_temps_sec": "Chronos précédents",
    "days_since_last_log": "Rythme de compétition",
    "nageur_age_mois_scaled": "Progression (Âge)",
    "nageur_sexe_encoded": "Genre",
    "perf_distance_encoded": "Distance",
    "perf_bassin_encoded": "Type de Bassin",
    "perf_nage_encoded": "Style de Nage"
  };

  const processedImportance = () => {
    const result = {};
    let saisonSum = 0;
    Object.entries(importance).forEach(([key, val]) => {
      if (key === 'mois_saison_sin' || key === 'mois_saison_cos' || key === 'saison_grouped') {
        saisonSum += val;
      } else {
        result[key] = val;
      }
    });
    if (saisonSum > 0) result['Saisonnalité'] = saisonSum;
    return Object.entries(result);
  };

  const sorted = processedImportance().sort(([, a], [, b]) => b - a);

  return (
    <div className="card" style={{ background: '#1a1a1a', border: '1px solid #333', height: '78.35%' }}>
      <h3 style={{ color: '#fff', marginBottom: '15px' }}>🧠 Importances des facteurs pour les prédictions </h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '15px' }}>
        {sorted.map(([key, val]) => (
          <div key={key}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '1.3rem' }}>
              <span style={{ color: '#aaa' }}>{featToLabel[key] || key}</span>
              <span style={{ color: '#00d4ff', fontWeight: 'bold' }}>{(val * 100).toFixed(1)}%</span>
            </div>
            <div style={{ height: '6px', background: '#333', borderRadius: '3px', marginTop: '5px' }}>
              <div
                style={{
                  width: `${val * 100}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, #005f73, #00d4ff)',
                  borderRadius: '3px',
                  transition: 'width 1s ease-in-out'
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

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
    </div>
  );
};

// --- MAIN APP ---

export default function App() {
  const [options, setOptions] = useState(null);
  const fileInputRef = useRef(null);

  const [user, setUser] = useState({ dob: "2005-01-01", sexe: "M" });
  const [fullHistory, setFullHistory] = useState([]);
  const [selectedNage, setSelectedNage] = useState("Nage Libre");
  const [selectedDistance, setSelectedDistance] = useState("100");
  const [selectedBassin, setSelectedBassin] = useState("50");

  const [addForm, setAddForm] = useState({ date: new Date().toISOString().split('T')[0], time: "" });
  const [predictionCache, setPredictionCache] = useState({});

  const currentDisciplineKey = useMemo(() => {
    return `${selectedNage}-${selectedDistance}-${selectedBassin}`;
  }, [selectedNage, selectedDistance, selectedBassin]);

  const results = predictionCache[currentDisciplineKey] || null;

  useEffect(() => {
    axios.get("http://localhost:8000/options").then((res) => {
      setOptions(res.data);
    }).catch(err => console.error("Erreur options:", err));
  }, []);

  // --- LOGIQUE IMPORT JSON ---
  const handleJsonImport = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const jsonData = JSON.parse(event.target.result);
        if (!Array.isArray(jsonData)) throw new Error("Le format doit être un tableau JSON");

        const enrichedHistory = jsonData.map(item => {
          const ageMois = calculateAgeMonths(user.dob, item.date);
          const dateObj = new Date(item.date);
          return {
            perf_nage: item.nage,
            nageur_sexe: user.sexe,
            perf_distance: parseInt(item.distance),
            perf_bassin: parseInt(item.bassin),
            mois_saison: dateObj.getMonth(),
            nageur_age_mois: ageMois,
            perf_temps_sec: parseFloat(item.temps),
            date: item.date
          };
        });

        setFullHistory(prev => [...prev, ...enrichedHistory]);
        alert(`${enrichedHistory.length} performances importées !`);
      } catch (err) {
        console.error(err);
        alert("Erreur lors de la lecture du fichier JSON. Vérifiez le format.");
      }
    };
    reader.readAsText(file);
    e.target.value = null;
  };

  const filteredHistory = useMemo(() => {
    return fullHistory.filter(h =>
      h.perf_nage === selectedNage &&
      String(h.perf_distance) === String(selectedDistance) &&
      String(h.perf_bassin) === String(selectedBassin)
    ).sort((a, b) => new Date(a.date) - new Date(b.date));
  }, [fullHistory, selectedNage, selectedDistance, selectedBassin]);

  const predictSequence = async (sequenceData, keyToCache) => {
    if (sequenceData.length === 0) return;

    // 1. On ne garde que les 20 dernières courses pour le modèle
    const last20Sequences = sequenceData.slice(-20);

    const past_sequence = last20Sequences.map((item, index) => {
      // Calcul du delta temps (le backend fera le log1p)
      let days = index > 0 ? getDaysBetween(last20Sequences[index - 1].date, item.date) : 0;
      
      return { 
        ...item, 
        days_since_last: days,
        nageur_sexe: user.sexe
      };
    });

    const lastItem = past_sequence[past_sequence.length - 1];
    const future_sequence = [];
    const lastDate = new Date(lastItem.date);

    // 2. Préparation du futur (Horizon 3)
    for (let i = 1; i <= 3; i++) {
      const futureDate = new Date(lastDate);
      futureDate.setMonth(futureDate.getMonth() + i);
      future_sequence.push({
        nageur_age_mois: calculateAgeMonths(user.dob, futureDate),
        perf_distance: lastItem.perf_distance,
        perf_bassin: lastItem.perf_bassin,
        mois_saison: futureDate.getMonth()
      });
    }

    try {
      const res = await axios.post("http://localhost:8000/predict_v4", { 
        past_sequence, 
        future_sequence 
      });
      
      setPredictionCache(prev => ({
        ...prev,
        [keyToCache]: {
          predictions: res.data.predictions,
          importance: res.data.feature_importance,
          temporal: res.data.temporal_attention
        },
      }));
    } catch (err) {
      console.error("Erreur API:", err);
    }
  };

  useEffect(() => {
    if (filteredHistory.length > 0 && !predictionCache[currentDisciplineKey]) {
      predictSequence(filteredHistory, currentDisciplineKey);
    }
  }, [currentDisciplineKey, filteredHistory]);

  const handleAddPerformance = async () => {
    if (!addForm.time) return;
    const newEntry = {
      perf_nage: selectedNage,
      nageur_sexe: user.sexe,
      perf_distance: parseInt(selectedDistance),
      perf_bassin: parseInt(selectedBassin),
      mois_saison: new Date(addForm.date).getMonth(),
      nageur_age_mois: calculateAgeMonths(user.dob, addForm.date),
      perf_temps_sec: parseFloat(addForm.time),
      date: addForm.date
    };
    const updated = [...filteredHistory, newEntry].sort((a, b) => new Date(a.date) - new Date(b.date));
    setFullHistory(prev => [...prev, newEntry]);
    setAddForm({ ...addForm, time: "" });
    await predictSequence(updated, currentDisciplineKey);
  };

  // --- CREATION DU GRAPHIQUE MIXTE (Temps + Attention) ---
  const getChartData = () => {
    const realData = filteredHistory.map(h => h.perf_temps_sec);
    const labels = filteredHistory.map(h => formatDate(h.date));
    const predLabels = ["M+1", "M+2", "M+3"];
    
    const predQ10 = results?.predictions.map(r => r.q10) || [];
    const predQ50 = results?.predictions.map(r => r.q50) || [];
    const predQ90 = results?.predictions.map(r => r.q90) || [];
    
    const paddingLength = Math.max(0, realData.length - 1);
    const padding = Array(paddingLength).fill(null);
    const lastVal = realData[realData.length - 1] || null;

    const basePredData = realData.length > 0 ? [...padding, lastVal] : [];

    let mappedAttention = [];
    if (results?.temporal && realData.length > 0) {
      const fullAttn = results.temporal.map(v => v * 100);
      
      // Si on a plus de 20 courses, l'attention ne couvre que les 20 dernières.
      // Il faut donc padder mappedAttention avec des nulls pour les vieilles courses.
      const leadingNulls = Math.max(0, realData.length - 20);
      const relevantAttn = fullAttn.slice(-Math.min(realData.length, 20));
      
      mappedAttention = [
        ...Array(leadingNulls).fill(null), 
        ...relevantAttn, 
        ...Array(predLabels.length).fill(null)
      ];
    }

    return {
      labels: [...labels, ...predLabels],
      datasets: [
        // 1. BARRES EN ARRIÈRE-PLAN (Attention)
        {
          type: 'bar',
          label: "Importance de cette course pour la prédiction",
          data: mappedAttention,
          backgroundColor: "rgba(0, 213, 255, 0.16)",
          hoverBackgroundColor: "rgba(0, 213, 255, 0.76)",
          yAxisID: 'y1',
          barPercentage: 1.0,
          categoryPercentage: 1.0,
          order: 10
        },
        // 2. COURBES DES TEMPS
        { 
          type: 'line',
          label: "Réel", 
          data: [...realData, ...Array(predLabels.length).fill(null)], 
          borderColor: "#00d4ff", 
          pointRadius: 5,
          yAxisID: 'y',
          order: 1
        },
        { 
          type: 'line',
          label: "Meilleur Cas (Q10)", 
          data: [...basePredData, ...predQ10], 
          borderColor: "rgba(255, 0, 122, 0.5)", 
          borderDash: [5, 5],
          pointRadius: 4,
          yAxisID: 'y',
          order: 2
        },
        { 
          type: 'line',
          label: "Pire Cas (Q90)", 
          data: [...basePredData, ...predQ90], 
          backgroundColor: "rgba(255, 0, 122, 0.15)",
          borderColor: "rgba(255, 0, 122, 0.5)",
          borderDash: [5, 5],
          fill: '-1',
          pointRadius: 4,
          yAxisID: 'y',
          order: 3
        },
        { 
          type: 'line',
          label: "Attendu (Q50)", 
          data: [...basePredData, ...predQ50], 
          borderColor: "#ff007a",
          borderWidth: 3,
          pointRadius: 5,
          yAxisID: 'y',
          order: 4
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
    tooltip: {
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      titleColor: '#fff',
      bodyColor: '#ccc',
      borderColor: '#333',
      borderWidth: 1,
      padding: 10,
      callbacks: {
        // --- CONFIGURATION DES UNITÉS AU SURVOL ---
        label: (context) => {
          let label = context.dataset.label || '';
          const value = context.parsed.y;

          if (value === null) return null;

          if (context.dataset.type === 'bar') {
            // Unité pour l'attention temporelle
            return `${label} : ${value.toFixed(2)}%`;
          } else {
            // Unité pour les chronos (Réel, Q10, Q50, Q90)
            return `${label} : ${value.toFixed(2)}s`;
          }
        }
      }
    },
    legend: {
      labels: { color: '#aaa', font: { size: 11 } }
    }
  },
    scales: {
      x: { grid: { color: '#333' } },
      y: { 
        type: 'linear',
        display: true,
        position: 'left',
        grid: { color: '#333' },
        title: { display: true, text: 'Temps (secondes)' }
      },
      y1: {
        type: 'linear',
        display: false,
        position: 'right',
        min: 0,
        suggestedMax: 1
      }
    }
  };

  if (!options) return <div className="app-container">Initialisation de l'IA...</div>;

  return (
    <div className="app-container">
      <aside className="sidebar">
        <h1 style={{ color: 'white', letterSpacing: '-1px' }}>🏊 SWIM.AI <span style={{fontSize: '0.6rem', color: '#ff007a'}}>V4</span></h1>
        <UserProfile user={user} setUser={setUser} options={options} />
        
        <div className="card" style={{ background: 'rgba(255,255,255,0.05)', marginTop: '20px' }}>
          <h4>📁 Importer</h4>
          <p style={{ fontSize: '0.75rem', color: '#ccc', marginBottom: '10px' }}>
            Importer l'historique complet d'un nageur à partir d'un fichier JSON.
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
      </aside>

      <main className="main-content">
        <div className="tabs">
          {options.perf_nage.map(n => (
            <button key={n} className={`tab-btn ${selectedNage === n ? "active" : ""}`} onClick={() => setSelectedNage(n)}>{n}</button>
          ))}
        </div>

        <div className="controls-bar">
          <select value={selectedDistance} onChange={e => setSelectedDistance(e.target.value)}>
            {options.perf_distance.map(d => <option key={d} value={d}>{d}m</option>)}
          </select>
          <select value={selectedBassin} onChange={e => setSelectedBassin(e.target.value)}>
            {options.perf_bassin.map(b => <option key={b} value={b}>{b}m</option>)}
          </select>
        </div>

        {/* SECTION PRINCIPALE : STATS + EXPLICABILITÉ */}
        {results && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px', marginBottom: '20px' }}>
            <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <div className="stat-box" style={{ marginBottom: '15px' }}>
                <div className="stat-label">Cible prochaine course</div>
                <div className="stat-value" style={{ fontSize: '1.8rem' }}>{results.predictions[0].q50.toFixed(2)}s</div>
              </div>
              <div className="stat-box" style={{ borderTop: '1px solid #333', paddingTop: '15px', marginBottom: '15px' }}>
                <div className="stat-label">Potentiel maximum prochaine course</div>
                <div className="stat-value" style={{ color: '#00d4ff' }}>{results.predictions[0].q10.toFixed(2)}s</div>
              </div>
              <div className="stat-box" style={{ borderTop: '1px solid #333', paddingTop: '15px' }}>
                <div className="stat-label">Pire cas prochaine course</div>
                <div className="stat-value" style={{ color: '#d36464' }}>{results.predictions[0].q90.toFixed(2)}s</div>
              </div>
            </div>
            
            <ImportanceWidget importance={results.importance} />
          </div>
        )}

        <div className="card">
          <h3 style={{ marginBottom: '5px' }}>Progression & Analyse Temporelle</h3>
          <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '15px' }}>
            Les barres bleues en fond indiquent quelles courses passées le modèle regarde le plus pour calculer le futur.
          </p>
          <div style={{ height: "400px" }}>
            <Chart data={getChartData()} options={chartOptions} />
          </div>
        </div>

        <div className="card" style={{ borderTop: '2px solid #00d4ff' }}>
          <h4>⏱️ Ajouter un Chrono</h4>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-end' }}>
            <input type="date" value={addForm.date} onChange={e => setAddForm({ ...addForm, date: e.target.value })} />
            <input type="number" placeholder="Secondes" value={addForm.time} onChange={e => setAddForm({ ...addForm, time: e.target.value })} />
            <button className="btn-primary" onClick={handleAddPerformance}>Enregistrer</button>
          </div>
        </div>

        {filteredHistory.length > 0 && (
          <div className="card">
            <h4>Historique</h4>
            <table style={{ width: '100%', textAlign: 'left', fontSize: '0.85rem' }}>
              <thead><tr style={{ color: '#666' }}><th>Date</th><th>Âge</th><th>Temps</th></tr></thead>
              <tbody>
                {filteredHistory.slice().reverse().map((h, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #222' }}>
                    <td style={{ padding: '8px 0' }}>{formatDate(h.date)}</td>
                    <td>{formatAgeReadable(h.nageur_age_mois)}</td>
                    <td style={{ color: '#00d4ff', fontWeight: 'bold' }}>{h.perf_temps_sec}s</td>
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