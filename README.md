# 🚆 KALANTAR: Railway Intelligence System

> **An Explainable and Probabilistic AI-Based Railway Intelligence System for Dynamic Train ETA Prediction, Delay Propagation Analysis, and Passenger Decision Support**

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/ML%20Engine-XGBoost-eb4034?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Tailwind CSS](https://img.shields.io/badge/Frontend-TailwindCSS-38bdf8?logo=tailwindcss&logoColor=white)](https://tailwindcss.com)

---

## 📌 Project Overview
**KALANTAR** bridges the gap between machine learning delay estimation and operational railway dynamics. Rather than treating delay prediction as an isolated black-box point estimate, KALANTAR introduces a three-tiered architecture:

1. **⚡ Dynamic Delay Predictor:** Trained on a chronological corpus of **1.5 million train journeys** (2018–2024), predicting arrival delays with an empirical Mean Absolute Error (MAE) of **34.65 minutes** (~34.2% relative error reduction over operational baselines).
2. **🔍 Explainable AI (XAI) & Uncertainty:** Provides local and global feature attribution via **TreeSHAP** alongside empirical 80% prediction confidence intervals ($[-53.4\text{ min}, +58.8\text{ min}]$).
3. **🚆 Network Route Delay Propagation:** Simulates station-by-station delay absorption and cascade dynamics across a directed multigraph of **175,002 route edges** constructed from 186,114 cleaned official timetable records.
4. **🛡️ Passenger Decision Support:** Evaluates transfer feasibility at junction stations, calculating the **Connection Risk Index** ($R_c$) and automatically searching for downstream alternative train connections when a miss is likely.

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│               FRONTEND (Modern Web Dashboard)          │
│  - Tailwind CSS + Glassmorphism UI                     │
│  - 3 Tabs: Live Predictor, Route Map, Decision Support │
└───────────────────────────▲────────────────────────────┘
                            │ (JSON REST API)
                            ▼
┌────────────────────────────────────────────────────────┐
│               BACKEND (FastAPI - Python)               │
│  - /api/predict-delay        (XGBoost + TreeSHAP)      │
│  - /api/simulate-route       (Buffer Absorption Engine)│
│  - /api/evaluate-connection  (Connection Risk & Alts)  │
│  - Interactive Swagger Docs at /docs                   │
└───────────────────────────▲────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│     ML Model Engine       │ │    Timetable Network      │
│ - XGBoost Regressor       │ │ - 8,151 Stations          │
│ - OneHot Preprocessor     │ │ - 175,002 Validated Edges │
│ - 38 Operational Features │ │ - Real Buffer Simulation  │
└───────────────────────────┘ └───────────────────────────┘
```

---

## 📊 Experimental Results

| Model Partition | Split Year | Historical Baseline MAE | KALANTAR XGBoost MAE | Relative Error Reduction |
| :--- | :--- | :--- | :--- | :--- |
| **Validation** | 2023 | 52.65 min | **34.65 min** | **34.2%** |
| **Held-out Test** | 2024 | 52.30 min | **34.92 min** | **33.2%** |

### Top Predictive Features (TreeSHAP)
1. **Late Incoming Rake ($\delta_{\text{rake}}$):** Explains ~32% of total delay variance. Turnaround delays are rarely recoverable along congested trunk corridors.
2. **Seasonal Severity Score ($\psi_{\text{season}}$):** Regional weather friction (dense winter fog, heavy monsoons).
3. **Historical Route On-Time Percentage ($\mu_{\text{ontime}}$):** Captures chronic corridor congestion.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Harsh2438/Kalantar.git
cd Kalantar
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Application
```bash
uvicorn backend.main:app --reload --port 8000
```
Open your browser to:
* **Web Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive API Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📁 Repository Structure

```
kalantar/
├── backend/
│   └── main.py                   # FastAPI application & REST endpoints
├── frontend/
│   └── index.html                # Responsive Web UI (Tailwind CSS)
├── models/
│   ├── model.pkl                 # Trained XGBoost Regressor
│   ├── preprocessor.pkl          # ColumnTransformer feature pipeline
│   └── metadata.json             # Model training metadata & quantiles
├── src/
│   ├── models/
│   │   ├── train_and_save.py     # Training and evaluation pipeline
│   │   ├── predict.py            # CLI prediction script
│   │   └── simulator.py          # Route propagation & connection risk engine
├── data/
│   ├── processed/                # Cleaned timetable & network edges
│   └── raw/                      # Data dictionaries & raw schemas
├── docs/
│   ├── data_limitations.md       # Empirical dataset boundary documentation
│   ├── kalantar_chapter.tex      # Full 8-page academic research manuscript
│   └── experiment_*.txt          # Experiment audit logs
├── notebooks/                    # Research and exploration notebooks
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## 📜 Academic Chapter Manuscript
The complete 8-page academic research paper detailing theoretical formulations, distance discrepancy audits, and mathematical modeling is available at [`docs/kalantar_chapter.tex`](docs/kalantar_chapter.tex).

---

## ⚖️ License
This project is open-source under the MIT License.
