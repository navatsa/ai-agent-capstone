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
└── src/
    ├── data_loader.js
    ├── cohort_analysis.js
    ├── time_series_analysis.js
    ├── predictive_modeling.js
    └── generate_sample_data.js
```

3 --> How to Run

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace eda_report.ipynb
# or open eda_report.ipynb directly in Jupyter / VS Code
```

Requires: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
