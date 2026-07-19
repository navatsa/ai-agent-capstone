"""
Generates a synthetic support-case dataset for the AI/automation agent DTC
(Days to Close) research project. Designed to mimic the fields described in
the project's Expected Data Sources section, with realistic-ish underlying
relationships baked in:

- Higher complexity/effort -> longer baseline DTC
- Agents reduce DTC, but the effect size depends on agent type vs. complexity:
    * Routing agents help a little across the board (faster triage)
    * Knowledge agents help more on medium complexity / KB-rich areas
    * Scoping/Troubleshooting agents help the most on HIGH complexity cases
    * Blended (multiple agents across lifecycle) generally outperforms any
      single agent, especially on high-priority/high-complexity cases
- Quality metrics (CSAT, reopen rate, SLA attainment) are correlated with DTC
  and automation success so the "maintain/improve quality" angle has signal.

Output: ../data/sample_support_cases.csv
"""

import numpy as np
import pandas as pd
from datetime import timedelta

RNG = np.random.default_rng(42)
N = 1500

product_areas = ["Networking", "Storage", "Identity", "Compute", "Data Platform", "Security", "Collaboration"]
issue_types = ["Bug", "Configuration", "Performance", "How-To", "Outage", "Feature Gap"]
severities = ["Sev-A", "Sev-B", "Sev-C", "Sev-D"]
complexities = ["Low", "Medium", "High"]
effort_levels = ["Low", "Medium", "High"]
agent_types = ["None", "Routing", "Knowledge", "Scoping", "CaseHealth", "Blended"]
lifecycle_stages = ["Creation", "Scoping/Troubleshooting", "Closure"]

agent_probs = [0.30, 0.15, 0.15, 0.15, 0.10, 0.15]  # ~70% of cases have some agent touch

rows = []
start_date = pd.Timestamp("2024-01-01")

for i in range(1, N + 1):
    severity = RNG.choice(severities, p=[0.10, 0.30, 0.40, 0.20])
    complexity = RNG.choice(complexities, p=[0.35, 0.40, 0.25])
    effort = RNG.choice(effort_levels, p=[0.40, 0.35, 0.25])
    product_area = RNG.choice(product_areas)
    issue_type = RNG.choice(issue_types)
    agent_type = RNG.choice(agent_types, p=agent_probs)

    # ---- Baseline DTC driven by severity/complexity/effort ----
    sev_base = {"Sev-A": 3, "Sev-B": 6, "Sev-C": 10, "Sev-D": 14}[severity]
    complexity_add = {"Low": 0, "Medium": 4, "High": 9}[complexity]
    effort_add = {"Low": 0, "Medium": 3, "High": 7}[effort]
    baseline_dtc = sev_base + complexity_add + effort_add + RNG.normal(0, 2)

    # ---- Agent effect (reduction in DTC, in days) ----
    if agent_type == "None":
        reduction = 0
    elif agent_type == "Routing":
        reduction = RNG.normal(1.0, 0.5)
    elif agent_type == "Knowledge":
        reduction = RNG.normal(2.0, 0.7) + (1.5 if complexity == "Medium" else 0)
    elif agent_type == "Scoping":
        reduction = RNG.normal(2.5, 0.8) + (3.5 if complexity == "High" else 0.5)
    elif agent_type == "CaseHealth":
        reduction = RNG.normal(1.5, 0.6)
    else:  # Blended
        reduction = RNG.normal(3.5, 1.0) + (4.0 if complexity == "High" else 1.5) + (1.0 if severity in ("Sev-A", "Sev-B") else 0)

    reduction = max(reduction, 0)
    dtc = max(baseline_dtc - reduction, 0.5)

    # ---- Automation / intervention detail ----
    automation_success = RNG.random() < (0.55 if agent_type == "None" else 0.85)
    num_actions = RNG.integers(1, 4) if agent_type == "None" else RNG.integers(2, 9)
    handoff_hours = round(abs(RNG.normal(6, 4)), 1) if agent_type != "None" else np.nan
    lifecycle_stage_touched = RNG.choice(lifecycle_stages) if agent_type != "None" else None

    # ---- Quality / outcome metrics (correlated with DTC & automation) ----
    first_response_hours = round(max(RNG.normal(4 if agent_type != "None" else 7, 2), 0.2), 1)
    escalation_flag = RNG.random() < (0.25 if complexity == "High" and agent_type == "None" else 0.08)
    reopen_flag = RNG.random() < (0.15 if not automation_success else 0.05)
    resolution_accuracy = round(np.clip(RNG.normal(0.9 if automation_success else 0.75, 0.08), 0, 1), 2)
    csat = round(np.clip(RNG.normal(4.3 if automation_success and dtc < baseline_dtc else 3.6, 0.6), 1, 5), 1)
    sla_target_days = {"Sev-A": 5, "Sev-B": 8, "Sev-C": 12, "Sev-D": 18}[severity]
    sla_met = dtc <= sla_target_days
    repeat_contact_flag = RNG.random() < (0.12 if not automation_success else 0.04)

    # ---- Knowledge / ops context ----
    kb_article_used = RNG.random() < (0.7 if agent_type in ("Knowledge", "Blended") else 0.35)
    kb_coverage_score = round(np.clip(RNG.normal(0.75 if kb_article_used else 0.4, 0.15), 0, 1), 2)
    engineer_workload = RNG.integers(3, 25)
    queue_depth = RNG.integers(5, 80)

    open_date = start_date + timedelta(days=int(RNG.integers(0, 500)), hours=int(RNG.integers(0, 24)))
    close_date = open_date + timedelta(days=float(dtc))

    rows.append({
        "case_id": f"CASE-{100000 + i}",
        "open_date": open_date,
        "close_date": close_date,
        "days_to_close": round(dtc, 2),
        "severity": severity,
        "product_area": product_area,
        "issue_type": issue_type,
        "complexity": complexity,
        "effort_level": effort,
        "escalation_flag": escalation_flag,
        "reopen_flag": reopen_flag,
        "agent_type": agent_type,
        "lifecycle_stage_touched": lifecycle_stage_touched,
        "handoff_time_hours": handoff_hours,
        "num_actions": num_actions,
        "automation_success": automation_success if agent_type != "None" else np.nan,
        "first_response_time_hours": first_response_hours,
        "resolution_accuracy": resolution_accuracy,
        "csat_score": csat,
        "sla_target_days": sla_target_days,
        "sla_met": sla_met,
        "repeat_contact_flag": repeat_contact_flag,
        "kb_article_used": kb_article_used,
        "kb_coverage_score": kb_coverage_score,
        "engineer_workload": engineer_workload,
        "queue_depth": queue_depth,
    })

df = pd.DataFrame(rows)
out_path = "../data/sample_support_cases.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
print(df.groupby("agent_type")["days_to_close"].mean().sort_values())
