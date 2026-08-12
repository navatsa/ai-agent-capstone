# Capstone Project: AI/Automation Agents & Support Case Outcomes

## Assignment 20.1 — Initial Report and Exploratory Data Analysis (EDA)

Notebook: [`notebooks/eda_report.ipynb`](notebooks/eda_report.ipynb)

1 --> Summary of Findings

This report explores `data/sample_support_cases.csv`, a set of 1,500
synthetic customer-support case records covering severity, product area,
agent/automation involvement, SLA outcomes, and customer satisfaction (CSAT).

- Missing data is structural, not random. ~31% of cases (466 rows) never
  had an AI/automation agent involved, which explains why `agent_type`,
  `lifecycle_stage_touched`, `handoff_time_hours`, and `automation_success`
  are blank for exactly those rows. These were imputed (`"None"` / `0` /
  `False`) rather than dropped, preserving the full sample.
- No duplicate cases were found (checked both on full-row and `case_id`
  uniqueness).
- SLA performance varies by severity, complexity, and effort level:
  higher-severity, higher-complexity, higher-effort cases miss SLA more
  often. Numeric load metrics (`queue_depth`, `engineer_workload`) show
  essentially no linear correlation with `days_to_close` or SLA outcomes in
  this sample.
- Outliers in `days_to_close` correspond almost entirely to cases with
  `reopen_flag` or `escalation_flag` set — these were flagged
  (`is_outlier_dtc`) rather than removed, since they represent real,
  operationally important cases.
- Baseline model: a Logistic Regression classifier predicting `sla_met`
  outperforms a majority-class baseline on precision, recall, and F1
  (accuracy alone is misleading here because SLA outcomes are imbalanced).
  The strongest predictors are `complexity`, `effort_level`, and whether an
  agent was involved (`has_agent`); `queue_depth` and `engineer_workload`
  show essentially no relationship with SLA outcomes in this sample.

2 --> Repository Layout

```
capstone/
├── README.md                  <- this file
├── data/
│   └── sample_support_cases.csv
├── notebooks/
│   ├── eda_report.ipynb       <- Assignment 20.1 deliverable (this report)
│   ├── capstone_analysis.ipynb<- extended cohort / time-series / modeling analysis
│   └── generate_sample_data.py
├── models/
│   ├── sla_met_logreg_balanced.joblib      <- persisted best model (Sec. 7 of eda_report.ipynb)
│   ├── sla_met_knn.joblib                  <- persisted KNN (k=5) model (Sec. 7.1)
│   ├── sla_met_decision_tree_balanced.joblib <- persisted Decision Tree (balanced) model (Sec. 7.1)
│   └── sla_met_random_forest_balanced.joblib <- persisted Random Forest (balanced) model (Sec. 7.1)
├── api/
│   ├── main.py                <- FastAPI service serving the persisted model
│   └── requirements.txt
└── src/
    ├── data_loader.js
    ├── cohort_analysis.js
    ├── time_series_analysis.js
    ├── predictive_modeling.js
    └── generate_sample_data.js
```

3 --> How to Run

**Notebook / EDA report:**
```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace eda_report.ipynb
# or open eda_report.ipynb directly in Jupyter / VS Code
```

Requires: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`.

**Prediction API:**

Running the notebook's "Model Persistence" section (Sec. 7) saves the best
model to `models/sla_met_logreg_balanced.joblib`. Once that file exists, serve
it via FastAPI:

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `http://127.0.0.1:8000/docs` for interactive Swagger UI, or call it
directly:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "severity": "Sev-C", "product_area": "Networking",
        "issue_type": "Configuration", "complexity": "Medium",
        "effort_level": "High", "has_agent": true,
        "sla_target_days": 12, "engineer_workload": 20,
        "queue_depth": 44, "kb_coverage_score": 0.89, "num_actions": 8
      }'
```

`GET /health` reports whether the model loaded successfully.
