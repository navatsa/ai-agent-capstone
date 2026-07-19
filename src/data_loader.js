/**
 * Shared CSV loader + typed row parsing for the capstone dataset.
 */
const fs = require("fs");
const path = require("path");
const { parse } = require("csv-parse/sync");

function loadCases(csvPath = path.join(__dirname, "..", "data", "sample_support_cases.csv")) {
  const raw = fs.readFileSync(csvPath, "utf8");
  const records = parse(raw, { columns: true, skip_empty_lines: true });

  return records.map((r) => ({
    case_id: r.case_id,
    open_date: new Date(r.open_date),
    close_date: new Date(r.close_date),
    days_to_close: parseFloat(r.days_to_close),
    severity: r.severity,
    product_area: r.product_area,
    issue_type: r.issue_type,
    complexity: r.complexity,
    effort_level: r.effort_level,
    escalation_flag: r.escalation_flag === "true",
    reopen_flag: r.reopen_flag === "true",
    agent_type: r.agent_type,
    has_agent: r.agent_type !== "None",
    lifecycle_stage_touched: r.lifecycle_stage_touched || null,
    handoff_time_hours: r.handoff_time_hours ? parseFloat(r.handoff_time_hours) : null,
    num_actions: parseInt(r.num_actions, 10),
    automation_success: r.automation_success === "" ? null : r.automation_success === "true",
    first_response_time_hours: parseFloat(r.first_response_time_hours),
    resolution_accuracy: parseFloat(r.resolution_accuracy),
    csat_score: parseFloat(r.csat_score),
    sla_target_days: parseInt(r.sla_target_days, 10),
    sla_met: r.sla_met === "true",
    repeat_contact_flag: r.repeat_contact_flag === "true",
    kb_article_used: r.kb_article_used === "true",
    kb_coverage_score: parseFloat(r.kb_coverage_score),
    engineer_workload: parseInt(r.engineer_workload, 10),
    queue_depth: parseInt(r.queue_depth, 10),
  }));
}

module.exports = { loadCases };
