/**
 * Predictive Modeling (Classification & Regression)
 * Trains KNN, Logistic Regression, and a Decision Tree (CART) classifier to
 * predict "high DTC-reduction potential" cases (i.e., where agent
 * intervention is likely to yield the greatest DTC improvement), using
 * case attributes (severity, complexity, effort, product area, etc.).
 *
 * Also ranks which case attributes are most predictive via the Decision
 * Tree's feature usage, giving a simple stand-in for feature importance.
 *
 * NOTE: This uses lightweight, dependency-free-ish JS ML libraries
 * (ml-knn, ml-logistic-regression, ml-cart) suitable for demonstrating the
 * approach end-to-end. For production-scale modeling, port this pipeline to
 * Azure ML (as noted in the project plan) using the same feature set.
 */
const KNN = require("ml-knn");
const { Matrix } = require("ml-matrix");
const LogisticRegression = require("ml-logistic-regression");
const { DecisionTreeClassifier } = require("ml-cart");
const ss = require("simple-statistics");
const { loadCases } = require("./data_loader");

function round(x, d = 3) {
  const f = 10 ** d;
  return Math.round(x * f) / f;
}

// ---- Encoding helpers ----
const severityMap = { "Sev-A": 0, "Sev-B": 1, "Sev-C": 2, "Sev-D": 3 };
const complexityMap = { Low: 0, Medium: 1, High: 2 };
const effortMap = { Low: 0, Medium: 1, High: 2 };
function oneHot(value, categories) {
  return categories.map((c) => (c === value ? 1 : 0));
}

function buildFeatures(rows, productAreas, issueTypes) {
  return rows.map((r) => [
    severityMap[r.severity],
    complexityMap[r.complexity],
    effortMap[r.effort_level],
    r.engineer_workload,
    r.queue_depth,
    r.kb_coverage_score,
    ...oneHot(r.product_area, productAreas),
    ...oneHot(r.issue_type, issueTypes),
  ]);
}

function trainTestSplit(X, y, testRatio = 0.25, seed = 7) {
  // deterministic shuffle
  let s = seed;
  const rand = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  const idx = X.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const nTest = Math.floor(X.length * testRatio);
  const testIdx = new Set(idx.slice(0, nTest));
  const XTrain = [], yTrain = [], XTest = [], yTest = [], testRowIdx = [];
  idx.forEach((i, pos) => {
    if (testIdx.has(i)) {
      XTest.push(X[i]);
      yTest.push(y[i]);
      testRowIdx.push(i);
    } else {
      XTrain.push(X[i]);
      yTrain.push(y[i]);
    }
  });
  return { XTrain, yTrain, XTest, yTest, testRowIdx };
}

function accuracy(yTrue, yPred) {
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) if (yTrue[i] === yPred[i]) correct++;
  return round(correct / yTrue.length);
}

function precisionRecallF1(yTrue, yPred, positiveClass = 1) {
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yPred[i] === positiveClass && yTrue[i] === positiveClass) tp++;
    else if (yPred[i] === positiveClass && yTrue[i] !== positiveClass) fp++;
    else if (yPred[i] !== positiveClass && yTrue[i] === positiveClass) fn++;
  }
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { precision: round(precision), recall: round(recall), f1: round(f1) };
}

function run() {
  const rows = loadCases();
  const productAreas = [...new Set(rows.map((r) => r.product_area))];
  const issueTypes = [...new Set(rows.map((r) => r.issue_type))];

  // ---- Label: "high DTC-reduction potential" ----
  // Approximate the *counterfactual* by comparing each case's DTC to the
  // median DTC of similar no-agent cases (same severity+complexity+effort).
  // Label = 1 if an agent was present AND DTC beat that baseline by >=25%.
  const noAgentRows = rows.filter((r) => !r.has_agent);
  function baselineFor(r) {
    const peers = noAgentRows.filter(
      (p) => p.severity === r.severity && p.complexity === r.complexity && p.effort_level === r.effort_level
    );
    if (peers.length === 0) return null;
    return ss.median(peers.map((p) => p.days_to_close));
  }

  const labeled = rows
    .filter((r) => r.has_agent)
    .map((r) => {
      const baseline = baselineFor(r);
      if (baseline === null) return null;
      const improvementPct = (baseline - r.days_to_close) / baseline;
      return { row: r, label: improvementPct >= 0.25 ? 1 : 0, improvementPct };
    })
    .filter(Boolean);

  console.log(`Labeled ${labeled.length} agent-assisted cases for classification.`);
  console.log(
    `Positive class ("high DTC-reduction potential", >=25% improvement): ${labeled.filter((l) => l.label === 1).length}`
  );

  const X = buildFeatures(labeled.map((l) => l.row), productAreas, issueTypes);
  const y = labeled.map((l) => l.label);

  const { XTrain, yTrain, XTest, yTest } = trainTestSplit(X, y);

  // ---- KNN ----
  const knn = new KNN(XTrain, yTrain, { k: 5 });
  const knnPred = XTest.map((x) => knn.predict(x));
  console.log("\n=== KNN (k=5) ===");
  console.log("Accuracy:", accuracy(yTest, knnPred));
  console.log(precisionRecallF1(yTest, knnPred));

  // ---- Logistic Regression ----
  const logreg = new LogisticRegression({ numSteps: 1000, learningRate: 0.05 });
  logreg.train(new Matrix(XTrain), Matrix.columnVector(yTrain));
  const lrPredRaw = logreg.predict(new Matrix(XTest));
  const lrPred = Array.from(lrPredRaw);
  console.log("\n=== Logistic Regression ===");
  console.log("Accuracy:", accuracy(yTest, lrPred));
  console.log(precisionRecallF1(yTest, lrPred));

  // ---- Decision Tree ----
  const tree = new DecisionTreeClassifier({ maxDepth: 5, minNumSamples: 10 });
  tree.train(XTrain, yTrain);
  const treePred = tree.predict(XTest);
  console.log("\n=== Decision Tree (CART) ===");
  console.log("Accuracy:", accuracy(yTest, treePred));
  console.log(precisionRecallF1(yTest, treePred));

  // ---- Feature importance proxy: permutation-style drop in accuracy ----
  const featureNames = [
    "severity", "complexity", "effort_level", "engineer_workload", "queue_depth", "kb_coverage_score",
    ...productAreas.map((p) => `product_area=${p}`),
    ...issueTypes.map((t) => `issue_type=${t}`),
  ];
  const baseAcc = accuracy(yTest, tree.predict(XTest));
  const importances = featureNames.map((name, fIdx) => {
    const XPerturbed = XTest.map((row) => row.slice());
    // shuffle this column across test rows
    const col = XPerturbed.map((r) => r[fIdx]);
    for (let i = col.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [col[i], col[j]] = [col[j], col[i]];
    }
    XPerturbed.forEach((r, i) => (r[fIdx] = col[i]));
    const permAcc = accuracy(yTest, tree.predict(XPerturbed));
    return { feature: name, importance_drop: round(baseAcc - permAcc) };
  });
  importances.sort((a, b) => b.importance_drop - a.importance_drop);
  console.log("\n=== Top Predictive Features (Decision Tree, permutation importance) ===");
  console.table(importances.slice(0, 10));

  console.log(
    "\nNext step per project plan: port this feature pipeline to Azure ML for scaled training/deployment once real Zebra AI data is loaded."
  );
}

run();
