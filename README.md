# TITANIC-SIM

> An AI-driven historical survival simulator. An LLM places you aboard the RMS Titanic in 1912 and quietly collects your decisions. A trained machine learning model then decides if you survive.

---

## What this is

Most Titanic projects stop at the notebook. This one asks: what if the prediction was the ending of a game?

A Large Language Model acts as a "Story Master" — generating period-authentic 1912 narrative, managing social dynamics, and collecting passenger features through natural dialogue. When enough information is gathered, it passes a structured JSON payload to a FastAPI backend running a trained logistic regression pipeline. The model returns a survival verdict. The Story Master delivers it as a closing scene.

This project was built to demonstrate end-to-end ML engineering: from raw data and feature engineering, through a production-style API, to LLM prompt engineering and system integration.

---

## Architecture
```
Player decisions
      │
      ▼
Story Master (LLM + System Prompt)
      │  Period-authentic narrative
      │  Hidden feature collection
      │
      ▼
Feature State (JSON)
      │
      ▼
FastAPI Backend (/predict endpoint)
      │  Pydantic validation
      │  Model pipeline inference
      │
      ▼
Logistic Regression Pipeline
      │  StandardScaler + one-hot encoded Title
      │  FamilySize composite feature
      │
      ▼
Survival Verdict + Confidence Score
      │
      ▼
Story Master delivers closing scene
```

See `docs/workflow.pdf` and `docs/dataflow1.5.png` for full diagrams.

---

## Tech stack

| Layer | Technology |
|---|---|
| ML model | scikit-learn — Logistic Regression Pipeline |
| Feature scaling | StandardScaler |
| API framework | FastAPI |
| Input validation | Pydantic v2 — BaseModel, Field, model_validator |
| LLM integration | OpenAI-compatible API (Story Master system prompt) |
| Model serialisation | joblib |

---

## Model performance

| Metric | Score |
|---|---|
| Cross-validation accuracy | 83.8% |
| ROC-AUC | 88.7% |
| Algorithm | Logistic Regression |

Key features: `Pclass`, `Age`, `Fare`, `FamilySize`, `Has_Cabin`, `Sex_male`, `Title` (one-hot encoded: Miss, Mr, Mrs, Rare)

---

## Project structure
```
TITANIC-SIM/
├── api/
│   └── main.py              # FastAPI app — /predict endpoint, Pydantic validation, CORS
├── docs/
│   ├── workflow.pdf          # System architecture diagram
│   ├── dataflow1.5.png       # Detailed dataflow diagram
│   └── game_design_document.md
├── model/
│   └── README.md             # Reproduction instructions
├── prompts/
│   └── System prompts.pdf    # Story Master system prompt (13+ restrictions)
├── notebooks/                # Training notebook (add after export)
├── .gitignore
└── README.md
```

---

## Running the API

**Requirements**
```bash
pip install fastapi uvicorn scikit-learn pandas joblib pydantic
```

**Start the server**
```bash
uvicorn api.main:app --reload
```

**Sample request**
```json
POST /predict
{
  "Pclass": 3,
  "Age": 22.0,
  "Fare": 7.25,
  "FamilySize": 1,
  "Has_Cabin": 0,
  "Sex_male": 1,
  "Title_Miss": 0,
  "Title_Mr": 1,
  "Title_Mrs": 0,
  "Title_Rare": 0
}
```

**Sample response**
```json
{
  "survived": 0,
  "probability": 0.18,
  "verdict": "Did not survive",
  "confidence": "high"
}
```

---

## Reproducing the model

The trained model binary is not stored in this repository. To reproduce it:

1. Add your training notebook to `notebooks/`
2. Run with Restart & Run All
3. The pipeline exports `titanic_model.joblib` to `model/`

---

## Known issues

These bugs were identified during integration testing and are documented here intentionally — they represent the next engineering iteration:

- **FamilySize miscounting** — the player is not included in the FamilySize count, causing the feature to be off by one
- **Fare inconsistency** — the fare narrated in the story does not always match the value passed in the JSON payload
- **Closing delimiter** — the Story Master occasionally uses an em dash instead of the specified hyphen delimiter, breaking the closing scene parser

---

## Background

Built as a solo portfolio project by an Electrical and Electronics Engineering graduate self-teaching machine learning. Started in the final year of a degree. Encouraged by a brother who made the case that the world was moving and it was time to move with it.

The goal was never just to predict Titanic survival. It was to build something that combined genuine ML engineering with a product idea worth remembering.

---

*Project status: v1.0 complete. Frontend and bug fixes planned for v2.*

