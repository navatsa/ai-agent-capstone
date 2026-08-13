# AI/Automation Agents & Support Case Outcomes — Capstone Final Report

**Jupyter Notebooks:** [`notebooks/eda_report.ipynb`](notebooks/eda_report.ipynb) (primary technical analysis) · [`notebooks/capstone_analysis.ipynb`](notebooks/capstone_analysis.ipynb) (cohort, time-series & modeling deep dive)

## Executive Summary (for a non-technical reader)

Customer support teams are increasingly using AI/automation "agents" (bots
that triage, route, or resolve cases) alongside human engineers. This project
asks a simple business question: **do these AI agents actually help cases
get resolved faster and within their promised deadline (SLA), and can we
predict — before a case is even worked — whether it's likely to miss its
SLA?** Being able to flag likely-to-miss cases early would let a support
organization proactively add staffing or escalate before a customer is
impacted, instead of reacting after the deadline has already passed.

Using 1,500 synthetic (sample) support case records, we found that
**AI-agent-assisted cases close about 2.7 days faster on average and meet
their SLA more than twice as often** as cases with no agent involvement
(42% vs. 20%). We built and compared several machine learning models to
predict SLA outcomes, and a **Logistic Regression model correctly identifies
whether a case will meet its SLA about 3 times out of 4 (F1 ≈ 0.77)**,
using only information available when the case is opened (severity,
complexity, effort level, whether an agent is assigned, workload). This
means the model could realistically be used to flag at-risk cases the
moment they're created.

---

## 1. Problem Statement

Support organizations want to know two things: (1) whether AI/automation
agents measurably improve case outcomes (time-to-close, SLA adherence,
customer satisfaction), and (2) whether a case's SLA outcome can be
*predicted at intake*, so that at-risk cases can be prioritized before they
breach their deadline. Missing an SLA has real costs — customer trust,
contractual penalties, and rework — so a reliable early-warning signal is
directly actionable. The goal of this project is to quantify the agent
effect and to build a classifier that predicts `sla_met` (whether a case
closes within its SLA target) from information known at case-open time.

## 2. Model Outcomes or Predictions

This is a **supervised, binary classification** problem: the model predicts
one of two outcomes for each support case — **`sla_met = True`** (the case
will close within its SLA target) or **`sla_met = False`** (it will miss
it). Four classification algorithms were trained and compared: **K-Nearest
Neighbors (KNN), Logistic Regression, Decision Tree, and Random Forest**
(see [Modeling](#5-modeling) below). No unsupervised learning was used, since
the goal is to predict a known, labeled outcome rather than discover
unlabeled structure in the data.

## 3. Data Acquisition

The analysis uses `data/sample_support_cases.csv`, a set of 1,500 synthetic
customer-support case records generated to resemble a real support-case
export (`notebooks/generate_sample_data.py`), with case-level fields
covering: severity, product area, issue type, complexity/effort estimates,
agent/automation involvement (`agent_type`, `has_agent`), workload context
(`queue_depth`, `engineer_workload`), and outcomes (`days_to_close`,
`sla_met`, `csat_score`, `reopen_flag`). A single source was used for this
capstone iteration; the notebooks are structured so a real CRM/ticketing
export (e.g., ServiceNow, Zendesk, Salesforce Service Cloud) can be dropped
in as a second, real-world data source in a future iteration (see
[Next Steps](#recommendations--next-steps)). The correlation heatmap and
class-balance/outlier plots in `notebooks/eda_report.ipynb` (Sections 4 and 6)
visualize each field's potential to predict SLA outcomes before any
modeling is done.

## 4. Data Preprocessing / Preparation

- **Missing values:** ~31% of cases (466 rows) never had an AI/automation
  agent involved, which is *why* `agent_type`, `lifecycle_stage_touched`,
  `handoff_time_hours`, and `automation_success` are blank for exactly those
  rows — this is structural missingness, not random. These fields were
  imputed (`"None"` / `0` / `False`) rather than dropped, to preserve the
  full 1,500-row sample.
- **Duplicates:** none found, checked on both full-row and `case_id`
  uniqueness.
- **Outliers:** extreme `days_to_close` values correspond almost entirely to
  cases with `reopen_flag` or `escalation_flag` set. These were flagged
  (`is_outlier_dtc`) rather than removed, since they represent real,
  operationally important cases rather than data errors.
- **Encoding:** categorical fields (`severity`, `product_area`, `issue_type`,
  `complexity`, `effort_level`, `has_agent`) were one-hot encoded via
  `ColumnTransformer` + `OneHotEncoder`; numeric fields were passed through
  unchanged.
- **Train/test split:** an 80/20 stratified split (`train_test_split(...,
  stratify=y)`) was used so the ~65/35 SLA-met/missed class balance is
  preserved in both sets.

## 5. Modeling

Four classifiers were trained on the same preprocessed features to predict
`sla_met`: **KNN, Logistic Regression, Decision Tree, and Random Forest**,
each compared with and without `class_weight="balanced"` (KNN excepted,
which has no such parameter) to address the ~65/35 class imbalance. All
models were wrapped in a single `sklearn` `Pipeline` (preprocessing +
classifier) to prevent data leakage between train and test folds. The two
strongest families (Logistic Regression and Random Forest, both balanced)
were then tuned with **`GridSearchCV` over 5-fold stratified
cross-validation**, searching Logistic Regression's regularization strength
(`C`) and Random Forest's `n_estimators`, `max_depth`, and
`min_samples_leaf`.

## 6. Model Evaluation

**Evaluation metric:** F1 score (harmonic mean of precision and recall) on
the positive ("SLA Met") class, alongside accuracy, precision, and recall.
**Why F1 and not accuracy:** `sla_met` is imbalanced (~65% missed / ~35%
met), so a model that always predicts "missed" would score a misleadingly
high ~65% accuracy while providing zero business value. F1 forces the model
to balance catching true SLA-met cases (recall) against not over-predicting
them (precision), which is the more honest measure of a genuinely useful
classifier here.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Majority-class baseline (Dummy) | 0.000 | 0.000 | 0.000 |
| Logistic Regression (balanced) | — | — | **~0.773 (best)** |
| Random Forest (balanced) | 0.699 | 0.819 | 0.754 |
| Decision Tree (balanced) | 0.636 | 0.667 | 0.651 |
| KNN (k=5) | 0.351 | 0.257 | 0.297 |

5-fold cross-validated `GridSearchCV` tuning did **not** meaningfully beat
the untuned balanced models above (best cross-validated F1 ≈ 0.775 for both
Logistic Regression and Random Forest), confirming the Section-6 comparison
wasn't a fluke of one particular train/test split. **Logistic Regression
(balanced) was selected as the final model** — it matches the tuned Random
Forest on F1 while being simpler, faster to retrain, and directly
interpretable via its coefficients (important when a support manager asks
*why* a case was flagged). It is persisted to
`models/sla_met_logreg_balanced.joblib`; the other three candidates (KNN,
Decision Tree, Random Forest) are also persisted alongside it for future
comparison or swapping (see [Repository Layout](#repository-layout)).

---

## Key Findings (plain language)

- **AI agents help.** Cases with any AI/automation agent involved close
  ~2.7 days faster on average and meet their SLA more than twice as often
  (42% vs. 20% for no-agent cases) — a statistically significant difference
  (p < 0.001).
- **Not all agents are equal.** "Blended" (multi-agent) handling produces the
  fastest resolutions and highest SLA-met rate (67%); specialized agents
  (e.g., "Scoping") beat generic ones (e.g., "Routing") on high-complexity
  cases by about 4.6 days on average.
- **SLA risk is predictable at intake.** Severity, complexity, effort level,
  and whether an agent is assigned are the strongest predictors of SLA
  outcome — meaningfully more predictive than current team workload
  (`queue_depth`, `engineer_workload`), which shows little relationship to
  outcomes in this sample.
- **Resolution times are trending up.** A slight but statistically
  significant upward trend (~0.25 extra days/month) was observed in the
  time-series analysis, worth monitoring as case volume or complexity grows.
- **The model is accurate enough to act on.** The chosen model correctly
  classifies SLA outcomes with F1 ≈ 0.77, well above the 0.00 baseline of
  always guessing "missed" — a real, usable early-warning signal rather
  than noise.

## Recommendations & Next Steps

- **Pilot an early-warning workflow:** score new cases with the persisted
  model at intake and route predicted at-risk cases to expedited handling.
- **Replace synthetic data with a real export** (e.g., from a live CRM/
  ticketing system) to validate that findings hold outside this sample.
- **Add time-based validation** (train on earlier months, test on later
  ones) to check the model holds up as case patterns drift over time.
- **Investigate the upward DTC trend** — is it complexity growth, staffing,
  or agent coverage? A follow-up cohort analysis could isolate the cause.
- **Try gradient boosting** (e.g., XGBoost/LightGBM) as an additional
  candidate now that a solid baseline and tuning pipeline exist.
- **Extend to secondary targets** (`csat_score`, `resolution_accuracy`) for
  a fuller picture of case quality, not just speed/SLA.

---

## Repository Layout

```
capstone/
├── README.md                  <- this file
├── data/
│   └── sample_support_cases.csv
├── notebooks/
│   ├── eda_report.ipynb       <- Main technical notebook: EDA, cleaning, modeling,
│   │                             cross-validation/GridSearchCV, model persistence
│   ├── capstone_analysis.ipynb<- Extended cohort / time-series / modeling analysis
│   └── generate_sample_data.py<- Generates data/sample_support_cases.csv
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
    └── generate_sample_data.js  <- JS port of the sample-data generator
```

## How to Run

**Notebooks:**
```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace eda_report.ipynb
jupyter nbconvert --to notebook --execute --inplace capstone_analysis.ipynb
# or open either .ipynb directly in Jupyter / VS Code
```

Requires: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`.

**Prediction API:**

Running `eda_report.ipynb`'s "Model Persistence" sections (Sec. 7 and 7.1)
saves all four candidate models to `models/`. The API serves the chosen best
model, `sla_met_logreg_balanced.joblib`; once that file exists, serve it via
FastAPI:

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
