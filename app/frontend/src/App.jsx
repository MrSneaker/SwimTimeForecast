import React, { useState, useEffect } from "react";
import axios from "axios";

export default function App() {
  const [options, setOptions] = useState(null);
  const [form, setForm] = useState({});
  const [history, setHistory] = useState([]); // tableau des performances
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

  if (!options) return <div>Chargement des options...</div>;

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm({
      ...form,
      [name]:
        name === "nageur_age_mois" || name === "perf_temps_sec"
          ? parseFloat(value)
          : value,
    });
  };

  const addPerformance = () => {
    setHistory([...history, { ...form }]);
    setForm({ ...form, perf_temps_sec: 60 }); // reset pour prochaine saisie
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // On envoie toute la séquence (history + entrée courante)
      const sequence = [...history, form];
      const res = await axios.post(
        "http://localhost:8000/predict_seq",
        { sequence }
      );
      setResult(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ padding: 20, maxWidth: 600 }}>
      <h1>Swim Time Predictor</h1>

      <form onSubmit={handleSubmit}>
        {/* Champs de formulaire */}
        <div>
          <label>Nage : </label>
          <select name="perf_nage" value={form.perf_nage} onChange={handleChange}>
            {options.perf_nage.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>

        <div>
          <label>Sexe : </label>
          <select name="nageur_sexe" value={form.nageur_sexe} onChange={handleChange}>
            {options.nageur_sexe.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>

        <div>
          <label>Âge (mois) : </label>
          <input type="number" name="nageur_age_mois" value={form.nageur_age_mois} onChange={handleChange} min={48}/>
        </div>

        <div>
          <label>Distance : </label>
          <select name="perf_distance" value={form.perf_distance} onChange={handleChange}>
            {options.perf_distance.map((d) => <option key={d} value={d}>{d} m</option>)}
          </select>
        </div>

        <div>
          <label>Bassin : </label>
          <select name="perf_bassin" value={form.perf_bassin} onChange={handleChange}>
            {options.perf_bassin.map((b) => <option key={b} value={b}>{b} m</option>)}
          </select>
        </div>

        <div>
          <label>Mois : </label>
          <select name="mois_saison" value={form.mois_saison} onChange={handleChange}>
            {options.mois_saison.map((m, idx) => <option key={idx} value={idx}>{m}</option>)}
          </select>
        </div>

        <div>
          <label>Dernière performance (sec) : </label>
          <input type="number" name="perf_temps_sec" value={form.perf_temps_sec} onChange={handleChange} min={0}/>
        </div>

        <div style={{ marginTop: 10 }}>
          <button type="button" onClick={addPerformance}>Ajouter à la séquence</button>
          <button type="submit" style={{ marginLeft: 10 }}>Predict</button>
        </div>
      </form>

      {/* Affichage de la séquence */}
      {history.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h3>Historique des performances</h3>
          <table border="1" cellPadding="5">
            <thead>
              <tr>
                <th>Nage</th>
                <th>Sexe</th>
                <th>Âge (mois)</th>
                <th>Distance</th>
                <th>Bassin</th>
                <th>Mois</th>
                <th>Temps (sec)</th>
              </tr>
            </thead>
            <tbody>
              {history.map((perf, idx) => (
                <tr key={idx}>
                  <td>{perf.perf_nage}</td>
                  <td>{perf.nageur_sexe}</td>
                  <td>{perf.nageur_age_mois}</td>
                  <td>{perf.perf_distance}</td>
                  <td>{perf.perf_bassin}</td>
                  <td>{perf.mois_saison}</td>
                  <td>{perf.perf_temps_sec}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Résultat prédiction */}
      {result && (
        <div style={{ marginTop: 20 }}>
          <h2>Predictions (seconds)</h2>
          <p>Q10: {result.q10.toFixed(2)}</p>
          <p>Q50: {result.q50.toFixed(2)}</p>
          <p>Q90: {result.q90.toFixed(2)}</p>
        </div>
      )}
    </div>
  );
}
