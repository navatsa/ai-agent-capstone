/**
 * Comparative Cohort Analysis (A/B-style)
 * Compares DTC and quality metrics for agent-assisted vs. non-agent cases,
 * segmented by agent type and by complexity, to test:
 *   - Do agents reduce DTC overall?
 *   - Do specialized agents (Scoping) help more on High complexity cases
 *     than generic agents (Routing)?
 *   - Does a Blended strategy outperform single-agent approaches?
 */
const ss = require("simple-statistics");
const { loadCases } = require("./data_loader");

function summarize(rows, label) {
  if (rows.length === 0) return null;
  const dtc = rows.map((r) => r.days_to_close);
  const csat = rows.map((r) => r.csat_score);
  const slaMetRate = rows.filter((r) => r.sla_met).length / rows.length;
  const reopenRate = rows.filter((r) => r.reopen_flag).length / rows.length;
  return {
    segment: label,
    n: rows.length,
    avg_dtc: round(ss.mean(dtc)),
    median_dtc: round(ss.median(dtc)),
    std_dtc: round(ss.standardDeviation(dtc)),
    avg_csat: round(ss.mean(csat)),
    sla_met_rate: round(slaMetRate),
    reopen_rate: round(reopenRate),
  };
}

function round(x, d = 2) {
  const f = 10 ** d;
  return Math.round(x * f) / f;
}

// Welch's t-test (does not assume equal variances) — good enough for an
// exploratory A/B comparison between two independent cohorts.
function welchTTest(a, b) {
  const n1 = a.length, n2 = b.length;
  const m1 = ss.mean(a), m2 = ss.mean(b);
  const v1 = ss.variance(a), v2 = ss.variance(b);
  const se = Math.sqrt(v1 / n1 + v2 / n2);
  const t = (m1 - m2) / se;
  // approximate two-tailed p-value via normal approximation (large n)
  const p = 2 * (1 - normalCdf(Math.abs(t)));
  return { t: round(t, 3), approx_p: round(p, 4), mean_diff: round(m1 - m2) };
}

function normalCdf(x) {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

function run() {
  const rows = loadCases();

  console.log("=== Overall: Agent-assisted vs. No-agent ===");
  const withAgent = rows.filter((r) => r.has_agent);
  const noAgent = rows.filter((r) => !r.has_agent);
  console.table([summarize(withAgent, "With Agent"), summarize(noAgent, "No Agent")]);
  const overallTest = welchTTest(withAgent.map((r) => r.days_to_close), noAgent.map((r) => r.days_to_close));
  console.log("Welch t-test (With Agent vs No Agent) on DTC:", overallTest);

  console.log("\n=== By Agent Type ===");
  const agentTypes = [...new Set(rows.map((r) => r.agent_type))];
  const byAgentType = agentTypes
    .map((a) => summarize(rows.filter((r) => r.agent_type === a), a))
    .sort((x, y) => x.avg_dtc - y.avg_dtc);
  console.table(byAgentType);

  console.log("\n=== By Agent Type x Complexity (avg DTC) ===");
  const complexities = ["Low", "Medium", "High"];
  const matrix = agentTypes.map((a) => {
    const entry = { agent_type: a };
    for (const c of complexities) {
      const subset = rows.filter((r) => r.agent_type === a && r.complexity === c);
      entry[`DTC_${c}`] = subset.length ? round(ss.mean(subset.map((r) => r.days_to_close))) : null;
      entry[`n_${c}`] = subset.length;
    }
    return entry;
  });
  console.table(matrix);

  console.log("\n=== Specialized (Scoping) vs Generic (Routing) on High-complexity cases ===");
  const scopingHigh = rows.filter((r) => r.agent_type === "Scoping" && r.complexity === "High");
  const routingHigh = rows.filter((r) => r.agent_type === "Routing" && r.complexity === "High");
  console.table([summarize(scopingHigh, "Scoping/High"), summarize(routingHigh, "Routing/High")]);
  console.log(
    "Welch t-test (Scoping vs Routing, High complexity):",
    welchTTest(scopingHigh.map((r) => r.days_to_close), routingHigh.map((r) => r.days_to_close))
  );

  console.log("\n=== Blended vs Best Single Agent per Priority (Severity) ===");
  const severities = ["Sev-A", "Sev-B", "Sev-C", "Sev-D"];
  const bySeverity = severities.map((sev) => {
    const blended = rows.filter((r) => r.severity === sev && r.agent_type === "Blended");
    const singleAgentTypes = agentTypes.filter((a) => a !== "None" && a !== "Blended");
    const singleAvgs = singleAgentTypes.map((a) => {
      const subset = rows.filter((r) => r.severity === sev && r.agent_type === a);
      return { agent_type: a, avg_dtc: subset.length ? ss.mean(subset.map((r) => r.days_to_close)) : Infinity };
    });
    const bestSingle = singleAvgs.sort((x, y) => x.avg_dtc - y.avg_dtc)[0];
    return {
      severity: sev,
      blended_avg_dtc: blended.length ? round(ss.mean(blended.map((r) => r.days_to_close))) : null,
      best_single_agent: bestSingle.agent_type,
      best_single_avg_dtc: round(bestSingle.avg_dtc),
    };
  });
  console.table(bySeverity);
}

run();
