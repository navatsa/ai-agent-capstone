/**
 * Generates a synthetic support-case dataset for the AI/automation agent DTC
 * (Days to Close) research project. See generate_sample_data.py for the full
 * rationale/comments on the underlying relationships baked into the data.
 *
 * Output: ../data/sample_support_cases.csv
 */
const fs = require("fs");
const path = require("path");

// ---- simple seeded RNG (mulberry32) for reproducibility ----
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(42);

function randNormal(mean = 0, sd = 1) {
  // Box-Muller transform
  const u1 = Math.max(rand(), 1e-9);
  const u2 = rand();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + z * sd;
}
function randInt(min, maxExclusive) {
  return Math.floor(rand() * (maxExclusive - min)) + min;
}
function choice(arr, probs) {
  if (!probs) return arr[randInt(0, arr.length)];
  const r = rand();
  let cum = 0;
  for (let i = 0; i < arr.length; i++) {
    cum += probs[i];
    if (r <= cum) return arr[i];
  }
  return arr[arr.length - 1];
}
function clip(v, lo, hi) {
  return Math.min(Math.max(v, lo), hi);
}

const N = 1500;
const productAreas = ["Networking", "Storage", "Identity", "Compute", "Data Platform", "Security", "Collaboration"];
const issueTypes = ["Bug", "Configuration", "Performance", "How-To", "Outage", "Feature Gap"];
const severities = ["Sev-A", "Sev-B", "Sev-C", "Sev-D"];
const complexities = ["Low", "Medium", "High"];
const effortLevels = ["Low", "Medium", "High"];
const agentTypes = ["None", "Routing", "Knowledge", "Scoping", "CaseHealth", "Blended"];
const lifecycleStages = ["Creation", "Scoping/Troubleshooting", "Closure"];
const agentProbs = [0.3, 0.15, 0.15, 0.15, 0.1, 0.15];

const sevBase = { "Sev-A": 3, "Sev-B": 6, "Sev-C": 10, "Sev-D": 14 };
const slaTarget = { "Sev-A": 5, "Sev-B": 8, "Sev-C": 12, "Sev-D": 18 };
const complexityAdd = { Low: 0, Medium: 4, High: 9 };
const effortAdd = { Low: 0, Medium: 3, High: 7 };

const startDate = new Date("2024-01-01T00:00:00Z");
const rows = [];

for (let i = 1; i <= N; i++) {
  const severity = choice(severities, [0.1, 0.3, 0.4, 0.2]);
  const complexity = choice(complexities, [0.35, 0.4, 0.25]);
  const effort = choice(effortLevels, [0.4, 0.35, 0.25]);
  const productArea = choice(productAreas);
  const issueType = choice(issueTypes);
  const agentType = choice(agentTypes, agentProbs);

  const baselineDtc = sevBase[severity] + complexityAdd[complexity] + effortAdd[effort] + randNormal(0, 2);

  let reduction;
  switch (agentType) {
    case "None":
      reduction = 0;
      break;
    case "Routing":
      reduction = randNormal(1.0, 0.5);
      break;
    case "Knowledge":
      reduction = randNormal(2.0, 0.7) + (complexity === "Medium" ? 1.5 : 0);
      break;
    case "Scoping":
      reduction = randNormal(2.5, 0.8) + (complexity === "High" ? 3.5 : 0.5);
      break;
    case "CaseHealth":
      reduction = randNormal(1.5, 0.6);
      break;
    default: // Blended
      reduction =
        randNormal(3.5, 1.0) +
        (complexity === "High" ? 4.0 : 1.5) +
        (severity === "Sev-A" || severity === "Sev-B" ? 1.0 : 0);
  }
  reduction = Math.max(reduction, 0);
  const dtc = Math.max(baselineDtc - reduction, 0.5);

  const automationSuccess = agentType === "None" ? null : rand() < 0.85;
  const automationAttemptSuccess = agentType === "None" ? rand() < 0.55 : automationSuccess;
  const numActions = agentType === "None" ? randInt(1, 4) : randInt(2, 9);
  const handoffHours = agentType !== "None" ? Math.round(Math.abs(randNormal(6, 4)) * 10) / 10 : "";
  const lifecycleStageTouched = agentType !== "None" ? choice(lifecycleStages) : "";

  const firstResponseHours = Math.round(Math.max(randNormal(agentType !== "None" ? 4 : 7, 2), 0.2) * 10) / 10;
  const escalationFlag = rand() < (complexity === "High" && agentType === "None" ? 0.25 : 0.08);
  const reopenFlag = rand() < (!automationAttemptSuccess ? 0.15 : 0.05);
  const resolutionAccuracy = Math.round(clip(randNormal(automationAttemptSuccess ? 0.9 : 0.75, 0.08), 0, 1) * 100) / 100;
  const csat = Math.round(clip(randNormal(automationAttemptSuccess && dtc < baselineDtc ? 4.3 : 3.6, 0.6), 1, 5) * 10) / 10;
  const slaMet = dtc <= slaTarget[severity];
  const repeatContactFlag = rand() < (!automationAttemptSuccess ? 0.12 : 0.04);

  const kbArticleUsed = rand() < (agentType === "Knowledge" || agentType === "Blended" ? 0.7 : 0.35);
  const kbCoverageScore = Math.round(clip(randNormal(kbArticleUsed ? 0.75 : 0.4, 0.15), 0, 1) * 100) / 100;
  const engineerWorkload = randInt(3, 25);
  const queueDepth = randInt(5, 80);

  const openDate = new Date(startDate.getTime() + randInt(0, 500) * 86400000 + randInt(0, 24) * 3600000);
  const closeDate = new Date(openDate.getTime() + dtc * 86400000);

  rows.push({
    case_id: `CASE-${100000 + i}`,
    open_date: openDate.toISOString(),
    close_date: closeDate.toISOString(),
    days_to_close: Math.round(dtc * 100) / 100,
    severity,
    product_area: productArea,
    issue_type: issueType,
    complexity,
    effort_level: effort,
    escalation_flag: escalationFlag,
    reopen_flag: reopenFlag,
    agent_type: agentType,
    lifecycle_stage_touched: lifecycleStageTouched,
    handoff_time_hours: handoffHours,
    num_actions: numActions,
    automation_success: agentType === "None" ? "" : automationSuccess,
    first_response_time_hours: firstResponseHours,
    resolution_accuracy: resolutionAccuracy,
    csat_score: csat,
    sla_target_days: slaTarget[severity],
    sla_met: slaMet,
    repeat_contact_flag: repeatContactFlag,
    kb_article_used: kbArticleUsed,
    kb_coverage_score: kbCoverageScore,
    engineer_workload: engineerWorkload,
    queue_depth: queueDepth,
  });
}

const headers = Object.keys(rows[0]);
const csvLines = [headers.join(",")];
for (const row of rows) {
  csvLines.push(headers.map((h) => row[h]).join(","));
}

const outPath = path.join(__dirname, "..", "data", "sample_support_cases.csv");
fs.writeFileSync(outPath, csvLines.join("\n"));
console.log(`Wrote ${rows.length} rows to ${outPath}`);

// quick summary: avg DTC by agent type
const byAgent = {};
for (const r of rows) {
  if (!byAgent[r.agent_type]) byAgent[r.agent_type] = [];
  byAgent[r.agent_type].push(r.days_to_close);
}
for (const [agent, vals] of Object.entries(byAgent)) {
  const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
  console.log(`${agent}: avg DTC = ${avg.toFixed(2)} (n=${vals.length})`);
}
