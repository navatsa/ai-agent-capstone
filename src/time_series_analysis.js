/**
 * Time Series Analysis / Forecasting
 * Aggregates DTC by month, computes a rolling average trend line, applies a
 * simple linear regression to detect drift/direction, and compares average
 * DTC before vs. after a chosen "agent rollout" cutoff date to estimate
 * whether improvements persist over time.
 */
const ss = require("simple-statistics");
const { loadCases } = require("./data_loader");

function monthKey(d) {
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`;
}

function round(x, d = 2) {
  const f = 10 ** d;
  return Math.round(x * f) / f;
}

function rollingAverage(values, window = 3) {
  const out = [];
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = values.slice(start, i + 1);
    out.push(round(ss.mean(slice)));
  }
  return out;
}

function run() {
  const rows = loadCases();

  // Group by close month
  const byMonth = new Map();
  for (const r of rows) {
    const key = monthKey(r.close_date);
    if (!byMonth.has(key)) byMonth.set(key, []);
    byMonth.get(key).push(r);
  }
  const months = [...byMonth.keys()].sort();

  const series = months.map((m) => {
    const caseRows = byMonth.get(m);
    return {
      month: m,
      n_cases: caseRows.length,
      avg_dtc: round(ss.mean(caseRows.map((r) => r.days_to_close))),
      avg_dtc_no_agent: (() => {
        const s = caseRows.filter((r) => !r.has_agent);
        return s.length ? round(ss.mean(s.map((r) => r.days_to_close))) : null;
      })(),
      avg_dtc_with_agent: (() => {
        const s = caseRows.filter((r) => r.has_agent);
        return s.length ? round(ss.mean(s.map((r) => r.days_to_close))) : null;
      })(),
    };
  });

  const rolling = rollingAverage(
    series.map((s) => s.avg_dtc),
    3
  );
  series.forEach((s, i) => (s.rolling_avg_3mo = rolling[i]));

  console.log("=== Monthly DTC Trend (with 3-month rolling average) ===");
  console.table(series);

  // Linear regression on month index vs avg_dtc to detect trend direction
  const xy = series.map((s, i) => [i, s.avg_dtc]);
  const regression = ss.linearRegression(xy);
  const slopeDirection = regression.m < 0 ? "IMPROVING (DTC trending down)" : "WORSENING (DTC trending up)";
  console.log("\n=== Linear Trend ===");
  console.log(`Slope (days/month): ${round(regression.m, 4)} -> ${slopeDirection}`);
  console.log(`Intercept: ${round(regression.b, 2)}`);
  console.log(`R^2: ${round(ss.rSquared(xy, ss.linearRegressionLine(regression)), 4)}`);

  // Before/after split at the midpoint month as a stand-in "agent rollout" date
  const midpoint = Math.floor(months.length / 2);
  const cutoffMonth = months[midpoint];
  const before = rows.filter((r) => monthKey(r.close_date) < cutoffMonth);
  const after = rows.filter((r) => monthKey(r.close_date) >= cutoffMonth);
  console.log(`\n=== Before/After Rollout Cutoff (${cutoffMonth}) ===`);
  console.table([
    { period: `Before ${cutoffMonth}`, n: before.length, avg_dtc: round(ss.mean(before.map((r) => r.days_to_close))) },
    { period: `After ${cutoffMonth}`, n: after.length, avg_dtc: round(ss.mean(after.map((r) => r.days_to_close))) },
  ]);
  console.log(
    "NOTE: Replace the midpoint cutoff with the actual agent-rollout date once real production data is loaded."
  );
}

run();
