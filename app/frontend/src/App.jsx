import React, { useState, useEffect } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import "chart.js/auto";

export default function App() {
  const [options, setOptions] = useState(null);
  const [form, setForm] = useState({});
  const [history, setHistory] = useState([]);
  const [result, setResult] = useState(null);

  useEffect(() => {
    axios.get("http://localhost:8000/options").then((res) => {
      setOptions(res.data);
      setForm({
        perf_nage: res.data.perf_nage[0],
        nageur_sexe: res.data.nageur_sexe[0],
        perf_distance: res.data.perf_distance[0],
        perf_bassin: res.data.perf_bassin[0],
        mois_saison: 0,
        nageur_age_mois: 180,
        perf_temps_sec: 60,
      });
    });
  }, []);

  if (!options) return <div style={{ padding: 30 }}>Chargement...</div>;

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm({
      ...form,
      [name]: ["nageur_age_mois", "perf_temps_sec"].includes(name)
        ? parseFloat(value)
        : value,
    });
  };

  /**
   * multi-step prediction:
   * Predict n steps ahead by feeding back each prediction
   * into the input sequence for the next prediction.
   */
  const addPerformance = async () => {
    const updatedHistory = [...history, { ...form }];
    setHistory(updatedHistory);
    setForm({ ...form, perf_temps_sec: 60 });

    let currentSequence = updatedHistory;

    const steps = 3;
    const multiPredictionResult = [];

    try {
      for (let i = 0; i < steps; i++) {
        const res = await axios.post("http://localhost:8000/predict_seq", {
          sequence: currentSequence,
        });

        const newPrediction = res.data;
        multiPredictionResult.push(newPrediction);

        currentSequence = [
          ...currentSequence,
          {
            ...currentSequence[currentSequence.length - 1],
            perf_temps_sec: newPrediction.q50,
            nageur_age_mois: (currentSequence[currentSequence.length - 1].nageur_age_mois + 1)
          },
        ];
      }

      setResult(multiPredictionResult);

    } catch (err) {
      console.error("Erreur lors de la multi-prédiction:", err);
      setResult(null);
    }
  };


  const lastValue = history.length > 0 ? history[history.length - 1].perf_temps_sec : 0;
  
  const predictionQ50s = result ? result.map(p => p.q50) : [];
  const predictionQ10s = result ? result.map(p => p.q10) : [];
  const predictionQ90s = result ? result.map(p => p.q90) : [];
  
  const numPredictions = predictionQ50s.length; 

  const nullPadding = history.length > 1
    ? Array(history.length - 1).fill(null)
    : [];

  
  const dataQ50 = [...nullPadding, lastValue, ...predictionQ50s];
  const dataQ10 = [...nullPadding, lastValue, ...predictionQ10s];
  const dataQ90 = [...nullPadding, lastValue, ...predictionQ90s];

  const predictionPoints = Array(numPredictions).fill(4);
  const predictionHoverPoints = Array(numPredictions).fill(7);

  const pointRadiusArray = history.length > 0
    ? [...Array(history.length).fill(0), ...predictionPoints]
    : [];
  const pointHoverRadiusArray = history.length > 0
    ? [...Array(history.length).fill(0), ...predictionHoverPoints]
    : [];
  
  // Labels for the 3 steps (J+1, J+2, J+3)
  const predictionLabels = Array.from({length: numPredictions}, (_, i) => `Pred J+${i + 1}`);

  // ---------------------------------------------

  const chartData =
    result && history.length > 0
      ? {
        labels: [...history.map((_, i) => `Run ${i + 1}`), ...predictionLabels],
        datasets: [
          {
            label: "True Performance",
            // We fill with null value to align with J+1, J+2, J+3
            data: [...history.map((h) => h.perf_temps_sec), ...Array(numPredictions).fill(null)],
            borderColor: "cyan",
            backgroundColor: "cyan",
            borderWidth: 2,
            tension: 0.2,
            pointRadius: 4,
          },
          {
            label: "10th percentile (best case)",
            data: dataQ10,
            borderColor: "rgba(255, 165, 0, 0.4)",
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            tension: 0.6,
            pointRadius: pointRadiusArray,
            pointBackgroundColor: "rgba(255, 165, 0, 0.6)",
            pointBorderColor: "#fff",
            pointHoverRadius: pointHoverRadiusArray,
          },
          {
            label: "Confidence Interval (10-90%)",
            data: dataQ90,
            borderColor: "rgba(255, 165, 0, 0.4)",
            borderWidth: 1,
            borderDash: [5, 5],
            backgroundColor: "rgba(255, 165, 0, 0.15)",
            fill: 1,
            tension: 0.6,
            pointRadius: pointRadiusArray,
            pointBackgroundColor: "rgba(255, 165, 0, 0.6)",
            pointBorderColor: "#fff",
            pointHoverRadius: pointHoverRadiusArray,
          },
          // Median prediction
          {
            label: "Predicted median (50th percentile)",
            data: dataQ50,
            borderColor: "orange",
            borderWidth: 3,
            borderDash: [7, 7],
            fill: false,
            tension: 0, 
            pointRadius: pointRadiusArray,
            pointBackgroundColor: "orange",
            pointBorderColor: "#fff",
            pointHoverRadius: pointHoverRadiusArray,
          },
        ],
      }
      : null;

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>🏊 Swim Time Predictor</h1>

      {/* Form Card */}
      <div style={styles.card}>
        <h2>Ajouter une performance</h2>

        {[
          { label: "Nage", name: "perf_nage", options: options.perf_nage },
          { label: "Sexe", name: "nageur_sexe", options: options.nageur_sexe },
          { label: "Distance", name: "perf_distance", options: options.perf_distance },
          { label: "Bassin", name: "perf_bassin", options: options.perf_bassin },
          { label: "Mois saison", name: "mois_saison", options: options.mois_saison },
        ].map((field) => (
          <div style={styles.row} key={field.name}>
            <label>{field.label}</label>
            <select
              name={field.name}
              value={form[field.name]}
              onChange={handleChange}
            >
              {field.options.map((opt, i) => (
                <option key={i} value={field.name === "mois_saison" ? i : opt}>
                  {opt}
                </option>
              ))}
            </select>
          </div>
        ))}

        <div style={styles.row}>
          <label>Âge (mois)</label>
          <input
            type="number"
            name="nageur_age_mois"
            value={form.nageur_age_mois}
            onChange={handleChange}
            min={48}
          />
        </div>

        <div style={styles.row}>
          <label>Temps (sec)</label>
          <input
            type="number"
            name="perf_temps_sec"
            value={form.perf_temps_sec}
            onChange={handleChange}
          />
        </div>

        <button style={styles.btnAdd} onClick={addPerformance}>
          ➕ Ajouter & prédire
        </button>
      </div>

      {/* History */}
      {history.length > 0 && (
        <div style={styles.card}>
          <h2>📄 Historique</h2>
          <table style={styles.table}>
            <thead>
              <tr>
                <th>Temps</th>
                <th>Nage</th>
                <th>Sexe</th>
                <th>Age</th>
                <th>Distance</th>
                <th>Bassin</th>
                <th>Mois</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h, i) => (
                <tr key={i}>
                  <td>{h.perf_temps_sec}s</td>
                  <td>{h.perf_nage}</td>
                  <td>{h.nageur_sexe}</td>
                  <td>{h.nageur_age_mois}</td>
                  <td>{h.perf_distance}</td>
                  <td>{h.perf_bassin}</td>
                  <td>{h.mois_saison}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Prediction */}
      {result && result.length > 0 && (
        <div style={{ ...styles.card, height: "50em" }}>
          <div style={{marginBottom: 20 }}>
            <h2>📊 Prédiction (Prochaine Course)</h2>
            <p>
              <b>Q10 (J+1):</b> {result[0].q10.toFixed(2)} sec
            </p>
            <p>
              <b>Q50 (J+1):</b> {result[0].q50.toFixed(2)} sec (médian)
            </p>
            <p>
              <b>Q90 (J+1):</b> {result[0].q90.toFixed(2)} sec
            </p>
          </div>

          <div style={{ width: "100%", height: "70%", marginTop: 5 }}>
            <Line
              data={chartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: "top" } },
                scales: {
                  x: { title: { display: true, text: "Performance #" } },
                  y: { title: { display: true, text: "Temps (sec)" } },
                },
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ---------- STYLES ----------
const styles = {
  container: {
    width: "100vw",
    minHeight: "100vh",
    padding: "30px",
    boxSizing: "border-box",
    fontFamily: "Arial",
  },
  title: { textAlign: "center", marginBottom: 20 },
  card: {
    width: "100%",
    maxWidth: "1200px",
    margin: "20px auto",
    background: "#303030ff",
    padding: 20,
    borderRadius: 10,
    boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
  },
  row: { display: "flex", justifyContent: "space-between", marginTop: 10 },
  btnAdd: {
    marginTop: 15,
    padding: "8px 20px",
    background: "#007bff",
    color: "#fff",
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
  },
  table: { width: "100%", borderCollapse: "collapse", marginTop: 10 },
};
