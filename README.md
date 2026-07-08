# iot-anomaly-detection-xai

Human-Centred IoT Sensor Anomaly Detection with Explainable AI and Natural Language Explanations.

---

## Overview

This project builds a human-in-the-loop (HITL) system for IoT sensor anomaly detection using the AI4I 2020 Predictive Maintenance dataset. When an anomaly is flagged by a multi-detector ensemble, a local or cloud LLM generates a plain-language explanation grounded in temporal and domain context. A Streamlit dashboard lets operators confirm, reject, or snooze alerts — closing the feedback loop.

**Two operating modes:**
- **Batch mode** — LangGraph pipeline processes all anomaly records, generates three explanation types in parallel, and presents them for operator review one by one
- **Stream mode** — CSV replay simulates a live sensor feed; each anomaly is explained asynchronously in a background thread while the stream continues

---

## Research Questions

**RQ1:** How do different prompting strategies (zero-shot, contextualised, RAG) affect explanation quality, measured via LLM-as-Judge?

**RQ2:** Does grounding LLM explanations in temporal sensor context and domain-rule evidence improve explanation specificity compared to zero-shot prompting?

---

## System Architecture

```
Raw CSV
  └─ Preprocessing (normalise, feature engineering, rule flags)
       └─ Multi-Detector Ensemble
            ├─ Z-score (global + rolling-50, threshold 3.0σ)
            ├─ Isolation Forest (200 trees, contamination 0.034)
            └─ Autoencoder (5→10→4→10→5, p95 MSE threshold)
                 └─ Score Fusion (⅓×Z + ⅓×IF + ⅓×AE)
                      ├─ ML path  — combined_score > 0.70 → anomaly
                      └─ Rule path — HDF ∨ TWF ∨ OSF ∨ PWF → anomaly
                           └─ Context Assembly (temporal window, rolling stats, domain rules)
                                └─ LLM Explainer
                                     ├─ Zero-Shot
                                     ├─ Contextualised
                                     └─ RAG (InMemoryVectorStore, 12 KB entries)
                                          └─ SHAP (TreeExplainer on IsolationForest)
                                               └─ Streamlit Dashboard (Batch + Stream tabs)
```

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — UCI Machine Learning Repository  
10,000 synthetic sensor readings with 5 failure modes.

| Sensor | Description |
|---|---|
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Machine process temperature |
| Rotational speed [rpm] | Spindle rotational speed |
| Torque [Nm] | Applied torque |
| Tool wear [min] | Cumulative tool wear |

**Failure modes:** HDF (Heat Dissipation), TWF (Tool Wear), OSF (Overstrain), PWF (Power), RNF (Random)

Source: AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

---

## Anomaly Detection

Three statistical detectors run in parallel; their normalised scores are fused with equal weights:

| Detector | Method | Score weight |
|---|---|---|
| Z-score | Max of per-sensor global Z and rolling-50 Z | ⅓ |
| Isolation Forest | Normalised anomaly score (200 trees, contam=0.034) | ⅓ |
| Autoencoder | Per-row MSE; threshold at p95 of normal-row MSE | ⅓ |

The anomaly gate has two independent paths:
- **ML path** — `combined_score > 0.70` flags the row
- **Rule path** — any of four deterministic domain rules fires independently

| Rule | Condition |
|---|---|
| HDF | `temp_diff_k < 8.6 K` AND `rot_speed_rpm < 1380 rpm` |
| TWF | `200 ≤ tool_wear_min ≤ 240 min` |
| OSF | `wear_torque > threshold(type)` — L: 11,000 / M: 12,000 / H: 13,000 Nm·min |
| PWF | `power_w < 3,500 W` OR `power_w > 9,000 W` |

Rule-triggered rows are routed to low-confidence monitoring by the LangGraph confidence layer unless a statistical detector also agrees.

---

## Anomaly Context Payload

Each anomaly record carries:
- **Temporal window** — ±10 readings around the anomaly row
- **Rolling statistics** — median and std over the preceding 50 rows per sensor
- **Global percentile bands** — p5/p25/p75/p95 over the full dataset
- **AE attribution** — per-sensor reconstruction error ranked as a % of total MSE
- **Rule explanation** — plain-English text for each triggered domain rule (HDF, TWF, OSF, PWF) with sensor values and thresholds
- **Detector summary** — which statistical detectors and domain rules flagged the row, with agreement level

This replaces raw SHAP values with information that is interpretable without ML expertise.

---

## LLM Explanation Strategies

| Strategy | Prompt content | Use case |
|---|---|---|
| Zero-shot | Worst sensor, Z-score, detector agreement | Fastest; no context needed |
| Contextualised | All sensors, AE attribution, rules, detector summary | Grounded in measurement context |
| RAG | Contextualised prompt + retrieved KB passages | Adds domain knowledge (failure mechanisms, sensor correlations) |

The RAG knowledge base contains 12 entries covering HDF/TWF/OSF/PWF/RNF mechanisms, autoencoder interpretation guidance, and sensor correlation patterns.

**LLM providers:**
- **Ollama (local)** — model name contains `:` (e.g. `gemma3:4b`, `llama3.2:3b`)
- **Groq (cloud)** — model name without `:` (e.g. `llama-3.3-70b-versatile`)

---

## Project Structure

```
.
├── src/
│   ├── config.py           # central config — all thresholds, hyperparameters, model names
│   ├── preprocessing/      # load, clean, feature engineering, normalise
│   ├── detector/           # zscore, isolation_forest, autoencoder, fusion
│   ├── explainer/          # prompts, context, rag, llm, shap, export, KB entries
│   ├── streaming/          # CSVStreamSimulator, OnlineDetector, StreamingWorker
│   ├── agents/             # LangGraph nodes, graph definitions, state, store
│   └── ui/                 # app.py — Streamlit dashboard (entry point)
├── data/
│   ├── input/
│   │   ├── ai4i2020.csv        # Raw source dataset
│   │   ├── ai4i_clean.csv      # Preprocessed — input to detection & streaming
│   │   └── ai4i_ranges.json    # Sensor min/max/mean/std for denormalisation
│   └── output/
│       ├── ae_ai4i.pt          # Saved autoencoder weights
│       ├── ai4i_anomaly_records.json
│       ├── ai4i_explanations.json
│       └── *.png               # Diagnostic plots
├── test_notebooks/
│   ├── data_preprocessing.ipynb
│   └── anomaly_detection.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally **or** a [Groq](https://console.groq.com) API key

### 1. Clone and install

```bash
git clone <repo-url>
cd iot-anomaly-detection-xai
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Local Ollama (default)
OLLAMA_BASE_URL=http://localhost:11434
EXPLAINER_MODEL=gemma3:4b
EMBED_MODEL=nomic-embed-text:latest

# Groq cloud (optional)
GROQ_API_KEY=your_key_here
GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant,gemma2-9b-it
```

### 3. Pull Ollama models (if using local LLM)

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text:latest
```

---

## Running the Pipeline

### Streamlit dashboard (recommended)

```bash
streamlit run src/ui/app.py
```

Opens at `http://localhost:8501`. Use the **Batch Mode** tab for full LangGraph pipeline with operator review, or **Stream Simulation** to watch row-by-row anomaly detection and live explanations.

### Docker (all-in-one with Ollama)

```bash
docker compose up --build
```

This starts three services: `ollama` (model server), `ollama-init` (pulls models once), and `streamlit` (app on port 8501). Models are cached in a named volume across restarts.

### Batch pipeline (CLI)

Run the full batch pipeline from the command line without the UI:

```bash
python -m src.explainer.pipeline
```

Outputs `data/output/ai4i_explanations.json` with all anomaly records and their three explanations.

### Preprocessing only

```bash
python -m src.preprocessing.pipeline
```

Generates `data/input/ai4i_clean.csv` and `data/input/ai4i_ranges.json` from the raw `data/input/ai4i2020.csv`.

### Notebooks

```bash
jupyter notebook test_notebooks/
```

- `data_preprocessing.ipynb` — EDA, feature engineering exploration
- `anomaly_detection.ipynb` — detector development, threshold sweeps, ensemble tuning

--

## References

- Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3).
- Chatterjee, A., & Ahmed, B. S. (2022). IoT anomaly detection methods and applications: A survey. *Internet of Things*, 19.
- Li, Z., Zhu, Y., & van Leeuwen, M. (2023). A survey on explainable anomaly detection. *ACM TKDD*, 18(1).
- Gummadi, N., et al. (2024). XAI-IoT: An explainable AI framework for enhancing anomaly detection in IoT systems. *IEEE Access*.
- Padín-Torrente, H., et al. (2026). Toward human-centered explainability: Natural language explanations for anomaly detection. *Information Systems Frontiers*.
